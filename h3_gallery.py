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


def _keyframe(assets: Path, demo_id: str, role: str) -> Path:
    matches = [path for suffix in ("png", "jpg", "jpeg", "webp") if (path := assets / f"{demo_id}-{role}.{suffix}").exists()]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {demo_id}-{role}.(png|jpg|jpeg|webp) in {assets}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=Path("gallery-keyframes"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    from h3_runtime import H3Runtime

    demos = json.loads((Path(__file__).resolve().parent / "demo_prompts.json").read_text())
    if args.limit is not None:
        demos = demos[: args.limit]
    jobs = []
    for demo in demos:
        required = set(demo.get("requires", []))
        first = _keyframe(args.assets, demo["id"], "first") if "first_frame" in required else None
        last = _keyframe(args.assets, demo["id"], "last") if "last_frame" in required else None
        jobs.append((demo, first, last))

    args.output.mkdir(parents=True, exist_ok=True)
    runtime = H3Runtime()
    for index, (demo, first, last) in enumerate(jobs, 1):
        values = dict(demo["input"])
        generated = runtime.generate(first_frame=first, last_frame=last, **values)
        destination = args.output / f"{demo['id']}{generated.suffix}"
        shutil.copy2(generated, destination)
        sidecar = destination.with_suffix(".gallery.json")
        sidecar.write_text(json.dumps(gallery_metadata(demo), indent=2) + "\n")
        print(f"[{index}/{len(jobs)}] {demo['id']} -> {destination}", flush=True)


if __name__ == "__main__":
    main()
