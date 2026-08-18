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
    parser.add_argument("--keyframes-dir", type=Path, default=Path(__file__).resolve().parent / "gallery-keyframes")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    from h3_runtime import H3Runtime

    demos = json.loads((Path(__file__).resolve().parent / "demo_prompts.json").read_text())
    if args.limit is not None:
        demos = demos[: args.limit]
    if not args.keyframes_dir.is_dir():
        parser.error("--keyframes-dir must point to an existing directory")

    args.output.mkdir(parents=True, exist_ok=True)
    runtime = H3Runtime()
    for index, demo in enumerate(demos, 1):
        values = dict(demo["input"])
        loop = bool(values.pop("loop", False))
        required = set(demo.get("requires", []))
        first_frame = args.keyframes_dir / f"{demo['id']}-first.png" if "first_frame" in required else None
        last_frame = args.keyframes_dir / f"{demo['id']}-last.png" if "last_frame" in required else None
        if first_frame is not None and not first_frame.is_file():
            parser.error(f"missing keyframe: {first_frame}")
        if last_frame is not None and not last_frame.is_file():
            parser.error(f"missing keyframe: {last_frame}")
        generated = runtime.generate(
            **values,
            task="fl2va" if required else "t2va",
            first_frame=first_frame,
            last_frame=last_frame,
            loop=loop,
        )
        destination = args.output / f"{demo['id']}{generated.suffix}"
        shutil.copy2(generated, destination)
        sidecar = destination.with_suffix(".gallery.json")
        sidecar.write_text(json.dumps(gallery_metadata(demo), indent=2) + "\n")
        print(f"[{index}/{len(demos)}] {demo['id']} -> {destination}", flush=True)


if __name__ == "__main__":
    main()
