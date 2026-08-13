"""Render the deterministic H3 launch suite and emit app.nz gallery sidecars."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def gallery_metadata(demo: dict) -> dict:
    values = demo["input"]
    mode = demo["mode"]
    return {
        "ext_id": demo["id"],
        "title": demo["id"].replace("-", " ").title(),
        "prompt": values["prompt"],
        "tags": [
            "minimax-h3",
            "ai-video",
            mode,
            values["aspect_ratio"],
            "native-audio",
            f"seed-{values['seed']}",
            f"steps-{values['steps']}",
            f"duration-{values['duration']}s",
        ],
        "source": "minimax-h3",
        "aspect": values["aspect_ratio"],
        "seed": values["seed"],
        "steps": values["steps"],
        "duration": values["duration"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    from h3_runtime import H3Runtime

    demos = json.loads((Path(__file__).resolve().parent / "demo_prompts.json").read_text())
    if args.limit is not None:
        demos = demos[: args.limit]
    if not args.reference_image.is_file():
        parser.error("--reference-image must point to an existing image")

    args.output.mkdir(parents=True, exist_ok=True)
    runtime = H3Runtime()
    for index, demo in enumerate(demos, 1):
        values = dict(demo["input"])
        values.pop("loop", None)
        generated = runtime.generate(reference_images=[args.reference_image], **values)
        destination = args.output / f"{demo['id']}{generated.suffix}"
        shutil.copy2(generated, destination)
        sidecar = destination.with_suffix(".gallery.json")
        sidecar.write_text(json.dumps(gallery_metadata(demo), indent=2) + "\n")
        print(f"[{index}/{len(demos)}] {demo['id']} -> {destination}", flush=True)


if __name__ == "__main__":
    main()
