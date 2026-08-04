"""Small dependency-free helpers shared by the RunPod entry point and tests."""

from __future__ import annotations


def frame_url(values: dict, name: str) -> str | None:
    """Prefer the public Cog field while retaining the direct-RunPod alias."""
    return values.get(name) or values.get(f"{name}_url")
