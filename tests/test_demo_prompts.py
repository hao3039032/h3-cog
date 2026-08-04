import json
from pathlib import Path

from h3_workflow import aligned_frames, dimensions, validate_inputs


def test_demo_suite_covers_gallery_modes_and_formats():
    demos = json.loads(Path("demo_prompts.json").read_text())
    assert len(demos) >= 7
    assert {demo["mode"] for demo in demos} >= {
        "text-to-video", "image-to-video", "first-last-frame", "first-last-loop",
    }
    assert {demo["input"]["aspect_ratio"] for demo in demos} >= {"16:9", "9:16", "1:1", "21:9"}
    assert len({demo["input"]["seed"] for demo in demos}) == len(demos)
    for demo in demos:
        values = demo["input"]
        first_frame = Path("anchor.png") if "first_frame" in demo.get("requires", []) else None
        last_frame = Path("end.png") if "last_frame" in demo.get("requires", []) else None
        validate_inputs(
            first_frame=first_frame,
            last_frame=last_frame,
            loop=values.get("loop", False),
            steps=values["steps"],
            seed=values["seed"],
        )
        assert aligned_frames(values["duration"]) % 17 == 5
        assert all(value % 32 == 0 for value in dimensions(values["aspect_ratio"], values["size"]))
