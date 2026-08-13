"""Small dependency-free helpers shared by the RunPod entry point and tests."""

from __future__ import annotations


def media_urls(values: dict, name: str) -> list[str]:
    """Normalize Cog list inputs and direct-RunPod ``*_urls`` aliases."""
    urls = values.get(name)
    if urls is None:
        urls = values.get(f"{name}_urls", [])
    if isinstance(urls, str):
        urls = [urls]
    if not isinstance(urls, list) or not all(isinstance(url, str) for url in urls):
        raise ValueError(f"{name} must be a list of HTTPS URLs")
    return urls
