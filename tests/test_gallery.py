from h3_gallery import gallery_metadata


def test_gallery_metadata_preserves_reproducible_h3_fields():
    demo = {
        "id": "test-shot",
        "mode": "first-last-frame",
        "input": {
            "prompt": "A fixed prompt",
            "aspect_ratio": "16:9",
            "duration": 5,
            "steps": 20,
            "seed": 42,
        },
    }
    metadata = gallery_metadata(demo)
    assert metadata["ext_id"] == "test-shot"
    assert metadata["source"] == "minimax-h3"
    assert metadata["seed"] == 42
    assert metadata["steps"] == 20
    assert {"minimax-h3", "first-last-frame", "16:9"} <= set(metadata["tags"])
    assert {"seed-42", "steps-20", "duration-5s"} <= set(metadata["tags"])
