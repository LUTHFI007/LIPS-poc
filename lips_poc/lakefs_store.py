"""lakeFS-backed private dataset storage (backed by MinIO — see
deploy/lakefs-local/). Holds the FULL dataset tree (train/val/test/test_ood_topo)
for evaluation; the public HF dataset repo carries only the public splits.

Versions follow the same scheme as HF: tags v0, v1, ... pointing at immutable
commit IDs ("tag for selection, commit ID for execution"). Author and parent
version are stored as native commit metadata (the counterpart of HF card tags).

Only the server holds lakeFS credentials (env: LAKEFS_ENDPOINT,
LAKEFS_ACCESS_KEY_ID, LAKEFS_SECRET_ACCESS_KEY) — users never touch storage.
"""
import os
import pathlib
import uuid

from lips_poc.hub_versioning import version_num

_BUCKET = "lakefs-data"
_BRANCH = "main"


def repo_name_for(hf_repo_name: str) -> str:
    """lakeFS repository ids must be lowercase; derive from the HF repo name."""
    return hf_repo_name.lower()


def _client():
    import lakefs_sdk
    from lakefs_sdk.client import LakeFSClient

    endpoint = os.environ.get("LAKEFS_ENDPOINT")
    key      = os.environ.get("LAKEFS_ACCESS_KEY_ID")
    secret   = os.environ.get("LAKEFS_SECRET_ACCESS_KEY")
    if not (endpoint and key and secret):
        raise RuntimeError(
            "lakeFS is not configured: set LAKEFS_ENDPOINT, LAKEFS_ACCESS_KEY_ID "
            "and LAKEFS_SECRET_ACCESS_KEY in .env (see deploy/lakefs-local/README.md), "
            "and make sure the stack is running (deploy/lakefs-local/start.sh)."
        )
    config = lakefs_sdk.Configuration(
        host=endpoint.rstrip("/") + "/api/v1", username=key, password=secret,
    )
    return LakeFSClient(config)


def ensure_repo(name: str) -> None:
    """Create the lakeFS repository if it doesn't exist yet.

    The storage path gets a random suffix (rather than reusing `name` as-is) so
    that deleting a repo and later recreating one under the same name never
    collides with objects the old repo left behind — lakeFS defers actually
    purging storage after a delete, and refuses to create a new repo directly
    on top of leftover objects ("storage namespace already in use")."""
    import lakefs_sdk
    from lakefs_sdk.exceptions import NotFoundException

    client = _client()
    try:
        client.repositories_api.get_repository(name)
    except NotFoundException:
        client.repositories_api.create_repository(
            lakefs_sdk.RepositoryCreation(
                name=name,
                storage_namespace=f"s3://{_BUCKET}/{name}/{uuid.uuid4().hex[:12]}",
                default_branch=_BRANCH,
            )
        )


def delete_repo(name: str) -> None:
    """Remove a repository (rollback of a failed brand-new publish only)."""
    _client().repositories_api.delete_repository(name, force=True)


def _list_paths(client, name: str, ref: str) -> list[str]:
    """All object paths in a repo at a ref (paginated)."""
    paths, after = [], ""
    while True:
        resp = client.objects_api.list_objects(name, ref, after=after, amount=1000)
        paths.extend(o.path for o in resp.results)
        if not resp.pagination.has_more:
            return paths
        after = resp.pagination.next_offset


def upload_tree(name: str, folder: str, message: str, metadata: dict) -> str:
    """Publish the contents of `folder` as the complete new state of `main`:
    reset any leftover uncommitted changes, upload every file, delete objects
    not in the new set, and commit (author/parent_version ride as commit
    metadata). Returns the new commit id — or, when nothing changed vs. the
    current head (e.g. a version bump with identical data), the head's id."""
    import lakefs_sdk
    from lakefs_sdk.exceptions import ApiException

    client = _client()
    folder_path = pathlib.Path(folder)

    # A previously failed publish may have left staged-but-uncommitted objects;
    # reset so every publish starts from the committed head.
    try:
        client.branches_api.reset_branch(
            name, _BRANCH, lakefs_sdk.ResetCreation(type="reset")
        )
    except ApiException:
        pass

    new_paths = set()
    for f in sorted(p for p in folder_path.rglob("*") if p.is_file()):
        rel = str(f.relative_to(folder_path))
        new_paths.add(rel)
        client.objects_api.upload_object(
            name, _BRANCH, rel, content=str(f), force=True
        )

    for stale in set(_list_paths(client, name, _BRANCH)) - new_paths:
        client.objects_api.delete_object(name, _BRANCH, stale)

    try:
        commit = client.commits_api.commit(
            name, _BRANCH,
            lakefs_sdk.CommitCreation(message=message, metadata=metadata),
        )
        return commit.id
    except ApiException as e:
        # Identical content -> lakeFS refuses an empty commit; the version tag
        # can safely point at the current head instead.
        if "no changes" in str(e):
            return client.refs_api.get_branch_head(name, _BRANCH).id \
                if hasattr(client.refs_api, "get_branch_head") \
                else client.branches_api.get_branch(name, _BRANCH).commit_id
        raise


def create_version_tag(name: str, tag: str, commit_id: str) -> None:
    import lakefs_sdk

    _client().tags_api.create_tag(
        name, lakefs_sdk.TagCreation(id=tag, ref=commit_id)
    )


def tag_exists(name: str, tag: str) -> bool:
    from lakefs_sdk.exceptions import NotFoundException

    try:
        _client().tags_api.get_tag(name, tag)
        return True
    except NotFoundException:
        return False


def list_versions(name: str) -> list[dict]:
    """Version tags (v0, v1, ...) newest first, same shape as
    hub_versioning.list_versions: {version, num, revision (commit id)}."""
    try:
        resp = _client().tags_api.list_tags(name)
    except Exception:
        return []
    versions = []
    for ref in resp.results:
        num = version_num(ref.id)
        if num is None:
            continue
        versions.append({"version": ref.id, "num": num, "revision": ref.commit_id})
    versions.sort(key=lambda v: v["num"], reverse=True)
    return versions


def resolve(name: str, tag: str) -> str:
    """Tag -> immutable commit id."""
    return _client().tags_api.get_tag(name, tag).commit_id


def get_version_metadata(name: str, revision: str) -> dict:
    """Author + parent_version from the commit metadata at a tag/commit.
    Mirrors data_hub.get_dataset_version_metadata."""
    try:
        commit = _client().commits_api.get_commit(name, revision)
        meta = commit.metadata or {}
    except Exception:
        meta = {}
    return {
        "author": meta.get("author") or "master",
        "parent_version": meta.get("parent_version") or None,
    }


def download_snapshot(name: str, commit_id: str, dest_dir: str) -> None:
    """Download the full tree at a commit into dest_dir, preserving paths."""
    client = _client()
    dest = pathlib.Path(dest_dir)
    for path in _list_paths(client, name, commit_id):
        target = dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        data = client.objects_api.get_object(name, commit_id, path)
        target.write_bytes(data)
