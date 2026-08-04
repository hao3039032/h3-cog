from h3_serverless import frame_url


def test_frame_url_keeps_cog_and_runpod_inputs_in_parity():
    assert frame_url({"first_frame": "https://cdn.example/cog.png"}, "first_frame") == "https://cdn.example/cog.png"
    assert frame_url({"first_frame_url": "https://cdn.example/runpod.png"}, "first_frame") == "https://cdn.example/runpod.png"
    assert frame_url(
        {
            "last_frame": "https://cdn.example/cog-last.png",
            "last_frame_url": "https://cdn.example/legacy-last.png",
        },
        "last_frame",
    ) == "https://cdn.example/cog-last.png"
    assert frame_url({}, "first_frame") is None
