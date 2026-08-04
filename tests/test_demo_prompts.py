import json
import struct
from pathlib import Path

from h3_workflow import aligned_frames, dimensions, validate_inputs


def png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[:24]
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


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
        required = set(demo.get("requires", []))
        root = Path("gallery-keyframes")
        first_frame = root / f"{demo['id']}-first.png" if "first_frame" in required else None
        last_frame = root / f"{demo['id']}-last.png" if "last_frame" in required else None
        for frame in (first_frame, last_frame):
            if frame is not None:
                assert frame.exists()
                width, height = png_dimensions(frame)
                expected_width, expected_height = dimensions(values["aspect_ratio"], values["size"])
                assert abs(width / height - expected_width / expected_height) < 0.02
        validate_inputs(
            first_frame=first_frame,
            last_frame=last_frame,
            loop=values.get("loop", False),
            steps=values["steps"],
            seed=values["seed"],
        )
        assert aligned_frames(values["duration"]) % 17 == 5
        assert all(value % 32 == 0 for value in dimensions(values["aspect_ratio"], values["size"]))
