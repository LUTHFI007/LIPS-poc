from huggingface_hub import HfApi

_api = HfApi()


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
