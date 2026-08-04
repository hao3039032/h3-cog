import json
from pathlib import Path

from h3_workflow import aligned_frames, dimensions, validate_inputs


def test_demo_suite_covers_text_image_and_loop():
    demos = json.loads(Path("demo_prompts.json").read_text())
    assert [demo["mode"] for demo in demos] == [
        "text-to-video",
        "image-to-video",
        "first-last-loop",
    ]
    assert len({demo["input"]["seed"] for demo in demos}) == len(demos)
    for demo in demos:
        values = demo["input"]
        first_frame = Path("anchor.png") if "first_frame" in demo.get("requires", []) else None
        validate_inputs(
            first_frame=first_frame,
            last_frame=None,
            loop=values.get("loop", False),
            steps=values["steps"],
            seed=values["seed"],
        )
        assert aligned_frames(values["duration"]) % 17 == 5
        assert all(value % 32 == 0 for value in dimensions(values["aspect_ratio"], values["size"]))
