"""Prompt normalization based on MiniMax's public FL2VA writing guide."""

from __future__ import annotations


CORE_MARKER = "integrated_multimodal_description:"


def format_h3_prompt(
    prompt: str,
    effective_seconds: float,
    *,
    first_frame: bool = False,
    last_frame: bool = False,
    structured: bool = True,
) -> str:
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("prompt is required")
    if not structured or CORE_MARKER in prompt:
        return prompt

    instruction = ""
    if first_frame and last_frame:
        instruction = (
            "How the reference pictures align with the target video — Picture 1 "
            "(from Shot 1) aligns with the 0.00-second mark of the target video; "
            f"Picture 2 (from Shot 1) aligns with the {effective_seconds:.2f}-second "
            "mark of the target video.\n\n"
        )
    elif first_frame:
        instruction = (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced.\n\n"
        )
    elif last_frame:
        instruction = (
            "How the reference pictures align with the target video — <Picture 1> "
            f"(from [Shot 1]) aligns with the {effective_seconds:.2f}-second mark "
            "of the target video.\n\n"
        )

    return (
        instruction
        + f"{CORE_MARKER} [Shot 1] {prompt}\n\n"
        + "overall_soundscape: Natural synchronized ambience and physical sounds "
        + "that match the described action.\n\n"
        + "non_diegetic_music: N/A"
    )
