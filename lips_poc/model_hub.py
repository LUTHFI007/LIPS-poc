import pathlib

from huggingface_hub import HfApi

_api = HfApi()


def search_models(keyword: str = "") -> list[dict]:
    results = _api.list_models(author="lips-poc", search=keyword or None, limit=100)
    rows = []
    for m in results:
        model_id = m.id
        last_modified = str(m.lastModified)[:10] if m.lastModified else ""
        url = f"https://huggingface.co/{model_id}"
        rows.append({
            "Model ID": model_id,
            "Last Modified": last_modified,
            "URL": url,
        })
    return rows


_VALID_MODEL_TYPES = {
    "tf_fc", "tf_leapnet", "torch_fc", "dc_approximation",
    "custom_tf", "custom_torch",
}


def validate_upload_inputs(model_type: str, repo_name: str, zip_bytes) -> list[str]:
    """Pure validation of upload inputs. Returns a list of error strings
    (empty = all clear). Does not write files, create repos, or upload."""
    import re
    import zipfile

    errors: list[str] = []

    # 1. model type
    if model_type not in _VALID_MODEL_TYPES:
        errors.append(f"Unknown model type '{model_type}'.")

    # 2. repo name non-empty
    if not repo_name:
        errors.append("Repository name cannot be empty.")
    else:
        # 3. repo name characters
        if not re.match(r"^[a-zA-Z0-9_\-.]+$", repo_name):
            errors.append(
                "Repository name may only contain letters, numbers, "
                "hyphens, underscores, and dots."
            )

        # 4. repo name must not already exist (best-effort)
        try:
            existing = [m.id for m in HfApi().list_models(author="lips-poc")]
            if f"lips-poc/{repo_name}" in existing:
                errors.append(f"A model named '{repo_name}' already exists in lips-poc.")
        except Exception:
            pass

    # 5. dc_approximation needs no ZIP — stop here.
    if model_type == "dc_approximation":
        return errors

    # 6. ZIP provided
    if not zip_bytes:
        errors.append("Please upload a ZIP file.")
        return errors

    # 7. valid ZIP
    import io

    if not zipfile.is_zipfile(io.BytesIO(zip_bytes)):
        errors.append("The uploaded file is not a valid ZIP archive.")
        return errors

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()

        # 8/10. weights file present
        if model_type in ("tf_fc", "tf_leapnet", "custom_tf"):
            if not any(n.endswith(".h5") or n.endswith(".weights.h5") for n in names):
                errors.append("ZIP must contain a weights file (.h5 or .weights.h5).")
        elif model_type in ("torch_fc", "custom_torch"):
            if not any(n.endswith(".pt") or n.endswith(".model.pt") for n in names):
                errors.append("ZIP must contain a weights file (.pt).")

        # 10. tf_fc: .h5 must be valid Keras HDF5 with a 'layers' group
        if model_type == "tf_fc":
            import h5py

            h5_names = [n for n in names if n.endswith(".h5")]
            if h5_names:
                try:
                    with h5py.File(io.BytesIO(zf.read(h5_names[0])), "r") as f:
                        if "layers" not in f:
                            errors.append(
                                "The weights file does not appear to be a valid "
                                "Keras HDF5 file (missing 'layers' group)."
                            )
                except Exception:
                    errors.append(
                        "The weights file does not appear to be a valid "
                        "Keras HDF5 file (missing 'layers' group)."
                    )

        # 11. at least one .ini
        ini_names = [pathlib.Path(n).name for n in names if n.endswith(".ini")]
        if not ini_names:
            errors.append("ZIP must contain a simulator config file (.ini).")

        # 12. .ini name must not collide with a benchmark config name
        import json

        registry_path = pathlib.Path(__file__).parent.parent / "dataset_registry.json"
        try:
            with registry_path.open() as f:
                registry = json.load(f)
            bench_ini_names = {
                pathlib.Path(v["config_path"]).name for v in registry.values()
            }
        except Exception:
            bench_ini_names = set()

        for name in ini_names:
            if name in bench_ini_names:
                errors.append(
                    f"The .ini file '{name}' is reserved for the benchmark config. "
                    "Rename your simulator config file."
                )

        # 13/14/15/16. custom: augmented_simulator.py present, parses, and
        # defines a class subclassing the required base.
        if model_type in ("custom_tf", "custom_torch"):
            required_base = (
                "TfFullyConnectedPowerGrid" if model_type == "custom_tf"
                else "TorchFullyConnected"
            )
            loader_names = [
                n for n in names
                if pathlib.Path(n).name == "augmented_simulator.py"
            ]
            if not loader_names:
                errors.append(
                    "Custom model type requires augmented_simulator.py in the ZIP. "
                    "Download the template from the instructions above."
                )
            else:
                import ast

                source = zf.read(loader_names[0]).decode("utf-8", errors="replace")
                try:
                    tree = ast.parse(source)
                except SyntaxError as e:
                    errors.append(f"augmented_simulator.py has a syntax error: {e}")
                else:
                    has_subclass = any(
                        isinstance(node, ast.ClassDef)
                        and any(
                            isinstance(b, ast.Name) and b.id == required_base
                            or isinstance(b, ast.Attribute) and b.attr == required_base
                            for b in node.bases
                        )
                        for node in tree.body
                    )
                    if not has_subclass:
                        errors.append(
                            "augmented_simulator.py must define a class that "
                            f"subclasses {required_base}."
                        )

    return errors


def validate_after_upload(repo_id: str, model_type: str) -> list[str]:
    """Verify the uploaded repo on HF actually has the expected files and tag."""
    from huggingface_hub import ModelCard

    errors: list[str] = []

    try:
        card = ModelCard.load(repo_id)
        tags = card.data.tags or []
    except Exception:
        tags = []
    if f"lips_model_type:{model_type}" not in tags:
        errors.append("HF model card is missing the lips_model_type tag.")

    try:
        repo_files = set(HfApi().list_repo_files(repo_id=repo_id, repo_type="model"))
    except Exception:
        repo_files = set()

    if model_type in ("tf_fc", "tf_leapnet", "custom_tf"):
        if not any(f.endswith(".h5") or f.endswith(".weights.h5") for f in repo_files):
            errors.append("No weights file (.h5) found in repo.")
    elif model_type in ("torch_fc", "custom_torch"):
        if not any(f.endswith(".pt") for f in repo_files):
            errors.append("No weights file (.pt) found in repo.")

    if model_type != "dc_approximation":
        if not any(f.endswith(".ini") for f in repo_files):
            errors.append("No simulator config (.ini) found in repo.")

    if model_type in ("custom_tf", "custom_torch"):
        if not any(pathlib.Path(f).name == "augmented_simulator.py" for f in repo_files):
            errors.append("augmented_simulator.py not found in repo.")

    return errors


def upload_model(
    repo_name: str,
    model_type: str,
    zip_bytes: "bytes | None",
    description: str = "",
) -> str:
    """Validate, create the HF repo, upload ZIP contents + model card, then
    verify. Returns the repo_id on success. Raises ValueError on any failure."""
    import io
    import zipfile

    errors = validate_upload_inputs(model_type, repo_name, zip_bytes)
    if errors:
        raise ValueError("\n".join(errors))

    repo_id = f"lips-poc/{repo_name}"
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=False)

    if model_type != "dc_approximation" and zip_bytes is not None:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                api.upload_file(
                    path_or_fileobj=io.BytesIO(zf.read(member)),
                    path_in_repo=member,
                    repo_id=repo_id,
                    repo_type="model",
                )

    from huggingface_hub import ModelCard, ModelCardData

    card_data = ModelCardData(
        tags=[f"lips_model_type:{model_type}", "lips", "powergrid"],
        library_name="lips",
    )
    card_content = f"---\n{card_data.to_yaml()}\n---\n\n{description or ''}"
    ModelCard(card_content).push_to_hub(repo_id)

    post_errors = validate_after_upload(repo_id, model_type)
    if post_errors:
        try:
            api.delete_repo(repo_id=repo_id, repo_type="model")
        except Exception:
            pass
        raise ValueError(
            f"Upload succeeded but post-upload check failed: {post_errors}. "
            "The repo has been deleted."
        )

    return repo_id
