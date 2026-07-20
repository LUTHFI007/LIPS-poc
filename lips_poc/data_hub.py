import configparser
import io
import json
import pathlib
import re
import shutil
import tempfile
import zipfile

from huggingface_hub import HfApi

from lips_poc import hub_versioning
from lips_poc import lakefs_store

_api = HfApi()

_REPO_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-.]+$")

# A dataset ZIP must contain all four benchmark splits. Only the public ones go
# to the HF repo (browsing + training data); the FULL tree goes to lakeFS and is
# what evaluation runs against — so the test splits never touch HF.
PUBLIC_SPLITS   = ("train", "val")
REQUIRED_SPLITS = ("train", "val", "test", "test_ood_topo")

_HF_ALLOW_PATTERNS = [f"{s}/**" for s in PUBLIC_SPLITS]

# The one LIPS benchmark config every registered dataset currently shares — it
# already has a section per benchmark (Benchmark1, Benchmark2, ...), so a new
# dataset never needs its own config file, only its own section name.
_BENCH_CONFIG_PATH = pathlib.Path(__file__).parent.parent / "configurations/powergrid/benchmarks/benchmark.ini"
_REGISTRY_PATH = pathlib.Path(__file__).parent.parent / "dataset_registry.json"


# benchmark.ini also has sections for Benchmark4, Benchmark5, DoNothing, and
# Benchmark_competition, but only Benchmark1-3 have real local data and are
# supported end-to-end today — capped here rather than offering options that
# would fail at publish/evaluate time.
_SUPPORTED_BENCHMARKS = {"Benchmark1", "Benchmark2", "Benchmark3"}


def available_benchmark_names() -> list[str]:
    """Supported benchmark.ini section names — what a newly published dataset
    can be registered against."""
    parser = configparser.ConfigParser()
    parser.read(_BENCH_CONFIG_PATH)
    return [s for s in parser.sections() if s in _SUPPORTED_BENCHMARKS]


def register_dataset(name: str, benchmark_name: str) -> None:
    """Add (or overwrite) this dataset's entry in dataset_registry.json so it
    becomes evaluable — publishing to HF/lakeFS alone isn't enough, main.py
    checks this file before Evaluate will run. `lakefs_repo` and `config_path`
    need no input: every dataset shares the one benchmark.ini, and the lakeFS
    repo name is always the published name lowercased (repo_name_for)."""
    registry = {}
    if _REGISTRY_PATH.exists():
        with _REGISTRY_PATH.open() as f:
            registry = json.load(f)

    registry[name] = {
        "benchmark_name": benchmark_name,
        "lakefs_repo": lakefs_store.repo_name_for(name),
        "config_path": str(_BENCH_CONFIG_PATH),
    }

    with _REGISTRY_PATH.open("w") as f:
        json.dump(registry, f, indent=2)
        f.write("\n")


def search_datasets(keyword: str = "") -> list[dict]:
    results = _api.list_datasets(author="lips-poc", search=keyword or None, limit=100)
    rows = []
    for ds in results:
        dataset_id = ds.id
        last_modified = str(ds.lastModified)[:10] if ds.lastModified else ""
        url = f"https://huggingface.co/datasets/{dataset_id}"
        rows.append({
            "Dataset ID": dataset_id,
            "Last Modified": last_modified,
            "URL": url,
        })
    return rows


def list_dataset_versions(repo_id: str) -> list[dict]:
    """A dataset's versions from its HF git tags (v0, v1, ...), newest first."""
    return hub_versioning.list_versions(repo_id, repo_type="dataset")


def next_dataset_version(repo_id: str) -> str:
    return hub_versioning.next_version(repo_id, repo_type="dataset")


def get_dataset_version_metadata(repo_id: str, revision: "str | None" = None) -> dict:
    """Read author + parent_version from the dataset card at a given revision.
    Returns {'author': <name, default 'master'>, 'parent_version': <str|None>}.
    Reads the card at the exact revision so each version reports its own author
    (datasets uploaded outside the app fall back to 'master')."""
    tags = []
    try:
        from huggingface_hub import DatasetCard, hf_hub_download
        readme = hf_hub_download(
            repo_id=repo_id, filename="README.md",
            revision=revision, repo_type="dataset",
        )
        tags = DatasetCard.load(readme).data.tags or []
    except Exception:
        tags = []

    author = next(
        (t.split(":", 1)[1] for t in tags
         if isinstance(t, str) and t.startswith("author:")),
        "",
    ) or "master"
    parent = next(
        (t.split(":", 1)[1] for t in tags
         if isinstance(t, str) and t.startswith("parent_version:")),
        None,
    )
    return {"author": author, "parent_version": parent}


def validate_dataset_upload_inputs(repo_name: str, zip_bytes, new_version: bool = False) -> list[str]:
    """Pure validation of dataset upload inputs. Returns a list of error strings
    (empty = all clear). Does not write files, create repos, or upload.

    `new_version=False` (a brand-new dataset): the repo must NOT already exist.
    `new_version=True`  (a new version of an existing dataset): it MUST exist."""
    errors: list[str] = []

    if not repo_name:
        errors.append("Dataset name cannot be empty.")
    else:
        if not _REPO_NAME_RE.match(repo_name):
            errors.append(
                "Dataset name may only contain letters, numbers, "
                "hyphens, underscores, and dots."
            )
        try:
            exists = HfApi().repo_exists(repo_id=f"lips-poc/{repo_name}", repo_type="dataset")
            if new_version and not exists:
                errors.append(f"No dataset named '{repo_name}' to add a version to.")
            elif not new_version and exists:
                errors.append(f"A dataset named '{repo_name}' already exists in lips-poc.")
        except Exception:
            pass

    if not zip_bytes:
        errors.append("Please upload a ZIP file.")
        return errors

    if not zipfile.is_zipfile(io.BytesIO(zip_bytes)):
        errors.append("The uploaded file is not a valid ZIP archive.")
        return errors

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        members = [n for n in zf.namelist() if not n.endswith("/")]
        if not members:
            errors.append("The ZIP archive is empty.")
            return errors
        if any(n.startswith("/") or ".." in pathlib.PurePosixPath(n).parts for n in members):
            errors.append("The ZIP contains unsafe paths (absolute or containing '..').")
            return errors

        missing = [s for s in REQUIRED_SPLITS if s not in _zip_top_dirs(members)]
        if missing:
            errors.append(
                "The ZIP must contain the full dataset — all four split folders "
                f"({', '.join(REQUIRED_SPLITS)}). Missing: {', '.join(missing)}. "
                f"Only {'/'.join(PUBLIC_SPLITS)} are published publicly; the test "
                "splits are stored privately for evaluation."
            )

    return errors


def _strip_wrapper(members: list) -> bool:
    """True when every member sits under one shared top-level folder (the
    natural result of zipping a folder rather than its contents)."""
    roots = {pathlib.PurePosixPath(n).parts[0] for n in members}
    return len(roots) == 1 and all(
        len(pathlib.PurePosixPath(n).parts) > 1 for n in members
    )


def _zip_top_dirs(members: list) -> set:
    """Top-level directory names of the archive, after wrapper stripping."""
    strip = _strip_wrapper(members)
    tops = set()
    for n in members:
        parts = pathlib.PurePosixPath(n).parts
        if strip:
            parts = parts[1:]
        if len(parts) > 1:
            tops.add(parts[0])
    return tops


def _extract_zip(zip_bytes, dest: pathlib.Path) -> None:
    """Extract the archive into dest, streaming file-by-file (never the whole
    archive in memory). If every member sits under one shared top-level folder —
    the natural result of zipping a folder rather than its contents — that
    wrapper is stripped, so the repo root holds train/, val/, ... directly."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        members = [n for n in zf.namelist() if not n.endswith("/")]
        strip = _strip_wrapper(members)
        for member in members:
            parts = pathlib.PurePosixPath(member).parts
            rel = pathlib.Path(*parts[1:]) if strip else pathlib.Path(*parts)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)


def _push_dataset_card(repo_id: str, description: str, author: str,
                       parent_version: "str | None" = None) -> None:
    from huggingface_hub import DatasetCard, DatasetCardData

    tags = [f"author:{author}", "lips", "powergrid"]
    if parent_version:
        tags.insert(1, f"parent_version:{parent_version}")
    card_data = DatasetCardData(tags=tags)
    card_content = f"---\n{card_data.to_yaml()}\n---\n\n{description or ''}"
    DatasetCard(card_content).push_to_hub(repo_id)


def upload_dataset(
    repo_name: str,
    zip_bytes: bytes,
    description: str = "",
    author: str = "master",
) -> str:
    """Publish a brand-new dataset from a full ZIP (all four splits): the public
    splits + card go to a new HF dataset repo, the FULL tree goes to a new
    lakeFS repo, and both are tagged v0 only after both uploads succeed.
    Returns the repo_id. Raises ValueError on any failure (both repos are
    rolled back so the name stays free for a retry)."""
    errors = validate_dataset_upload_inputs(repo_name, zip_bytes)
    if errors:
        raise ValueError("\n".join(errors))

    repo_id = f"lips-poc/{repo_name}"
    lk_name = lakefs_store.repo_name_for(repo_name)
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=False)

    try:
        with tempfile.TemporaryDirectory() as tmp:
            _extract_zip(zip_bytes, pathlib.Path(tmp))
            api.upload_folder(
                repo_id=repo_id, repo_type="dataset", folder_path=tmp,
                allow_patterns=_HF_ALLOW_PATTERNS,
                commit_message="Upload dataset (v0)",
            )
            lakefs_store.ensure_repo(lk_name)
            lk_commit = lakefs_store.upload_tree(
                lk_name, tmp, message="Publish v0", metadata={"author": author},
            )
        _push_dataset_card(repo_id, description, author)
    except Exception:
        # A half-uploaded brand-new dataset is useless — remove both repos so
        # the name stays free for a retry.
        try:
            api.delete_repo(repo_id=repo_id, repo_type="dataset")
        except Exception:
            pass
        try:
            lakefs_store.delete_repo(lk_name)
        except Exception:
            pass
        raise

    # Tag both systems only after everything is uploaded: v0 = HF HEAD (public
    # splits + card) paired with the lakeFS commit (full tree).
    api.create_tag(repo_id=repo_id, tag="v0", repo_type="dataset", exist_ok=True)
    lakefs_store.create_version_tag(lk_name, "v0", lk_commit)
    return repo_id


def upload_new_dataset_version(
    repo_name: str,
    zip_bytes: bytes,
    description: str = "",
    author: str = "master",
    parent_version: "str | None" = None,
) -> "tuple[str, str]":
    """Publish a NEW VERSION of an existing dataset: push the uploaded files as
    new commits on the EXISTING repo, then tag v{n+1} and record author + parent
    version on the card. Returns (repo_id, new_version_tag).

    Unlike upload_dataset, this does NOT create the repo and does NOT delete it
    on failure (older versions live on as tags and must be preserved). Raises
    ValueError on any failure."""
    errors = validate_dataset_upload_inputs(repo_name, zip_bytes, new_version=True)
    if errors:
        raise ValueError("\n".join(errors))

    repo_id = f"lips-poc/{repo_name}"
    lk_name = lakefs_store.repo_name_for(repo_name)
    api = HfApi()

    new_tag = next_dataset_version(repo_id)     # e.g. v0 exists -> v1
    if lakefs_store.tag_exists(lk_name, new_tag):
        raise ValueError(
            f"lakeFS already has a tag '{new_tag}' for this dataset — the two "
            "stores are out of sync. Resolve manually before publishing."
        )
    if parent_version is None:                  # default: builds on the latest version
        versions = list_dataset_versions(repo_id)
        parent_version = versions[0]["version"] if versions else None

    # An empty description means "keep the current card text" — otherwise every
    # version upload would wipe the card body.
    if not description:
        try:
            from huggingface_hub import DatasetCard
            description = DatasetCard.load(repo_id).text
        except Exception:
            pass

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        _extract_zip(zip_bytes, tmp_path)
        uploaded_public = {
            str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")
            if p.is_file() and p.relative_to(tmp_path).parts[0] in PUBLIC_SPLITS
        }
        api.upload_folder(
            repo_id=repo_id, repo_type="dataset", folder_path=tmp,
            allow_patterns=_HF_ALLOW_PATTERNS,
            commit_message=f"Upload dataset ({new_tag})",
        )
        lakefs_store.ensure_repo(lk_name)   # legacy datasets may predate lakeFS
        lk_commit = lakefs_store.upload_tree(
            lk_name, tmp, message=f"Publish {new_tag}",
            metadata={"author": author, "parent_version": parent_version or ""},
        )

    # A version is exactly the uploaded file set: drop files left over from
    # earlier versions so they don't silently linger in the new one. They stay
    # retrievable forever through the older tags. (lakeFS handles its own
    # staleness inside upload_tree.)
    keep = {"README.md", ".gitattributes"}
    stale = [
        f for f in api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        if f not in uploaded_public and f not in keep
    ]
    if stale:
        from huggingface_hub import CommitOperationDelete
        api.create_commit(
            repo_id=repo_id, repo_type="dataset",
            operations=[CommitOperationDelete(path_in_repo=f) for f in stale],
            commit_message=f"Remove files not part of {new_tag}",
        )

    _push_dataset_card(repo_id, description, author, parent_version)

    # Tag both systems as the last step, so a failed publish never yields a
    # claimable half-version.
    api.create_tag(repo_id=repo_id, tag=new_tag, repo_type="dataset")
    lakefs_store.create_version_tag(lk_name, new_tag, lk_commit)
    return repo_id, new_tag
