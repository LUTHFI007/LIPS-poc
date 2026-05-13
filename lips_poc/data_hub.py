from huggingface_hub import HfApi

_api = HfApi()


def search_datasets(keyword: str = "powergrid") -> list[dict]:
    results = _api.list_datasets(search=keyword, limit=50)
    rows = []
    for ds in results:
        dataset_id = ds.id
        author = dataset_id.split("/")[0] if "/" in dataset_id else ""
        last_modified = str(ds.lastModified)[:10] if ds.lastModified else ""
        url = f"https://huggingface.co/datasets/{dataset_id}"
        rows.append({
            "Dataset ID": dataset_id,
            "Author": author,
            "Last Modified": last_modified,
            "URL": url,
        })
    return rows
