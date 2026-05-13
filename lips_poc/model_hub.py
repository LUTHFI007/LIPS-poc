from huggingface_hub import HfApi

_api = HfApi()


def search_models(keyword: str = "powergrid") -> list[dict]:
    results = _api.list_models(search=keyword, limit=50)
    rows = []
    for m in results:
        model_id = m.id
        author = model_id.split("/")[0] if "/" in model_id else ""
        last_modified = str(m.lastModified)[:10] if m.lastModified else ""
        url = f"https://huggingface.co/{model_id}"
        rows.append({
            "Model ID": model_id,
            "Author": author,
            "Last Modified": last_modified,
            "URL": url,
        })
    return rows
