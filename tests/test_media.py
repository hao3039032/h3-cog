import pytest

from h3_media import audio_profile, encode_profiles


def test_av1_prefers_5090_nvenc_then_svt_fallback():
    extension, profiles = encode_profiles("webm-av1", 26, {"av1_nvenc", "libsvtav1"})
    assert extension == "webm"
    assert profiles[0][1] == "av1_nvenc"
    assert profiles[1][1] == "libsvtav1"


def test_h264_has_cpu_fallback():
    extension, profiles = encode_profiles("mp4-h264", 26, set())
    assert extension == "mp4"
    assert profiles[-1][1] == "libx264"
    with pytest.raises(ValueError):
        encode_profiles("vp9", 26, set())


def test_mp4_copies_native_aac_but_webm_encodes_opus():
    assert audio_profile("mp4") == ["-c:a", "copy"]
    assert audio_profile("webm") == ["-c:a", "libopus", "-b:a", "160k"]
