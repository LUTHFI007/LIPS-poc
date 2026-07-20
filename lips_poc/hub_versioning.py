"""Shared HuggingFace version-tag helpers for the Model Hub and the Data Hub.

Versions are HF git tags named v0, v1, v2, ... on a repo; `repo_type` selects
which kind of repo ("model" or "dataset") the tags live on. Tags are the
human-facing labels — each one points at an immutable commit SHA, which is what
downloads and evaluation records pin to.
"""
import re

from huggingface_hub import HfApi

_VERSION_RE = re.compile(r"^v(\d+)$")


def version_num(tag_name: str):
    """Return the integer N for a tag named 'vN', else None."""
    m = _VERSION_RE.match(tag_name)
    return int(m.group(1)) if m else None


def list_versions(repo_id: str, repo_type: str = "model") -> list[dict]:
    """Read a repo's versions from its HF git tags (v0, v1, ...), newest first.
    Each entry: {version, num, revision (commit SHA)}. Read-only; returns [] if
    the repo has no version tags or can't be read."""
    try:
        refs = HfApi().list_repo_refs(repo_id=repo_id, repo_type=repo_type)
    except Exception:
        return []

    versions = []
    for tag in getattr(refs, "tags", []) or []:
        num = version_num(tag.name)
        if num is None:
            continue
        versions.append({
            "version":  tag.name,
            "num":      num,
            "revision": tag.target_commit,
        })
    versions.sort(key=lambda v: v["num"], reverse=True)
    return versions


def next_version(repo_id: str, repo_type: str = "model") -> str:
    """The next version tag to assign for a repo: 'v0' if none exist yet,
    otherwise 'v{max+1}'."""
    versions = list_versions(repo_id, repo_type)
    return "v0" if not versions else f"v{versions[0]['num'] + 1}"
