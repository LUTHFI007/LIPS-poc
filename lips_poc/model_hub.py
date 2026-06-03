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
