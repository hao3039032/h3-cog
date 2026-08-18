import pytest

from h3_serverless import media_urls


def test_media_urls_keep_cog_and_runpod_inputs_in_parity():
    assert media_urls({"reference_images": ["https://cdn.example/cog.png"]}, "reference_images") == [
        "https://cdn.example/cog.png"
    ]
    assert media_urls({"reference_videos_urls": "https://cdn.example/runpod.mp4"}, "reference_videos") == [
        "https://cdn.example/runpod.mp4"
    ]
    assert media_urls({"first_frame": "https://cdn.example/first.png"}, "first_frame") == [
        "https://cdn.example/first.png"
    ]
    assert media_urls({}, "reference_audios") == []
    with pytest.raises(ValueError):
        media_urls({"reference_images": 42}, "reference_images")
