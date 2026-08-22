from pathlib import Path

import pytest

from h3_tuning import CacheTuning
from h3_workflow import (
    build_workflow,
    infer_task,
    normalize_sol_profile,
    normalize_task,
    task_partition,
    validate_inputs,
)


def test_t2va_selects_fl2va_partition_and_image_conditioning():
    graph = build_workflow(
        prompt="p",
        task="t2va",
        width=1024,
        height=576,
        frames=124,
        steps=20,
        seed=7,
    )
    assert graph["1"]["inputs"]["unet_name"] == "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    assert graph["5"]["class_type"] == "MiniMaxH3ImageToVideo"
    assert "first_frame" not in graph["5"]["inputs"]
    assert "last_frame" not in graph["5"]["inputs"]


def test_fl2va_preprocesses_both_keyframes_to_target_canvas():
    graph = build_workflow(
        prompt="p",
        task="fl2va",
        width=864,
        height=480,
        frames=124,
        steps=20,
        seed=8,
        first_image_name="first.png",
        last_image_name="last.png",
    )
    assert graph["1"]["inputs"]["unet_name"] == "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    assert graph["5"]["class_type"] == "MiniMaxH3ImageToVideo"
    assert graph["15"] == {"class_type": "LoadImage", "inputs": {"image": "first.png"}}
    assert graph["16"]["class_type"] == "ImageScale"
    assert graph["16"]["inputs"] == {
        "image": ["15", 0],
        "upscale_method": "lanczos",
        "width": 864,
        "height": 480,
        "crop": "center",
    }
    assert graph["17"] == {"class_type": "LoadImage", "inputs": {"image": "last.png"}}
    assert graph["18"]["class_type"] == "ImageScale"
    assert graph["5"]["inputs"]["first_frame"] == ["16", 0]
    assert graph["5"]["inputs"]["last_frame"] == ["18", 0]


def test_fl2va_loop_reuses_scaled_first_frame_as_last_frame():
    graph = build_workflow(
        prompt="p",
        task="fl2va",
        width=768,
        height=768,
        frames=124,
        steps=20,
        seed=9,
        first_image_name="first.png",
        loop=True,
    )
    assert graph["5"]["inputs"]["first_frame"] == ["16", 0]
    assert graph["5"]["inputs"]["last_frame"] == ["16", 0]
    assert "17" not in graph


def test_ref2va_uses_reference_dit_and_zero_based_autogrow_keys():
    graph = build_workflow(
        prompt="Use <Picture 1>, <Video 1>, and <Audio 2>.",
        task="ref2va",
        width=864,
        height=480,
        frames=124,
        steps=20,
        seed=11,
        reference_image_names=["identity.png"],
        reference_video_names=["motion.mp4"],
        reference_audio_names=["voice.wav"],
    )
    assert graph["1"]["inputs"]["unet_name"] == "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    assert graph["2"]["inputs"]["clip_name"] == "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
    assert graph["3"]["inputs"]["vae_name"] == "minimax_h3_video_vae_fp32.safetensors"
    assert graph["5"]["class_type"] == "MiniMaxH3ReferenceToVideo"
    assert graph["5"]["inputs"]["audio_vae"] == ["4", 0]
    assert graph["8"]["inputs"]["sampler_name"] == "res_multistep"
    assert graph["9"]["inputs"] == {"model": ["1", 0], "scheduler": "simple", "steps": 20, "denoise": 1.0}
    assert graph["13"]["inputs"]["audio"] == ["12", 0]
    assert graph["14"]["inputs"]["codec"] == "h264"
    assert graph["14"]["inputs"]["codec.encoding"] == "re-encode"
    assert graph["14"]["inputs"]["codec.encoding.crf"] == 17.0
    assert graph["15"] == {"class_type": "LoadImage", "inputs": {"image": "identity.png"}}
    assert graph["16"] == {"class_type": "LoadVideo", "inputs": {"file": "motion.mp4"}}
    assert graph["17"] == {"class_type": "GetVideoComponents", "inputs": {"video": ["16", 0]}}
    assert graph["5"]["inputs"]["ref_images.ref_image_0"] == ["15", 0]
    assert graph["5"]["inputs"]["ref_videos.ref_video_0"] == ["17", 0]
    assert graph["5"]["inputs"]["ref_video_audios.ref_video_audio_0"] == ["17", 1]
    assert graph["18"] == {"class_type": "LoadAudio", "inputs": {"audio": "voice.wav"}}
    assert graph["5"]["inputs"]["ref_audios.ref_audio_0"] == ["18", 0]


def test_easycache_is_inserted_before_media_loader_nodes():
    cache = CacheTuning("balanced", 0.12, 0.15, 0.9, False, "sweep-1", "balanced")
    graph = build_workflow(
        prompt="p",
        task="ref2va",
        width=768,
        height=768,
        frames=124,
        steps=20,
        seed=9,
        reference_image_names=["identity.png"],
        cache=cache,
    )
    assert graph["15"]["class_type"] == "EasyCache"
    assert graph["15"]["inputs"]["model"] == ["1", 0]
    assert graph["6"]["inputs"]["model"] == ["15", 0]
    assert graph["9"]["inputs"]["model"] == ["15", 0]
    assert graph["16"] == {"class_type": "LoadImage", "inputs": {"image": "identity.png"}}


def test_sol_conservative_profile_is_inserted_with_sage_safe_fallback_settings():
    graph = build_workflow(
        prompt="p",
        task="t2va",
        width=864,
        height=480,
        frames=124,
        steps=20,
        seed=9,
        sol_profile="conservative",
    )
    assert graph["15"]["class_type"] == "SolAttnPatch"
    assert graph["15"]["inputs"] == {
        "model": ["1", 0],
        "tau": 1.0,
        "start_percent": 0.20,
        "end_percent": 0.90,
        "min_tokens": 4096,
        "int8_qk": True,
        "sink_conditioning": "exact_kv_and_rows",
        "morton": True,
        "morton_curve": "2d_frame",
        "int8_pv": False,
        "verbose": True,
        "use_tma": False,
        "dense_blocks": "0-2,-1",
    }
    assert graph["6"]["inputs"]["model"] == ["15", 0]
    assert graph["9"]["inputs"]["model"] == ["15", 0]


def test_sol_balanced_profile_composes_before_easycache():
    cache = CacheTuning("balanced", 0.12, 0.15, 0.9, False, "sweep-1", "balanced")
    graph = build_workflow(
        prompt="p",
        task="t2va",
        width=864,
        height=480,
        frames=124,
        steps=20,
        seed=9,
        sol_profile="balanced",
        cache=cache,
    )
    assert graph["15"]["class_type"] == "SolAttnPatch"
    assert graph["15"]["inputs"]["tau"] == 1.3
    assert graph["15"]["inputs"]["int8_pv"] is True
    assert graph["16"]["class_type"] == "EasyCache"
    assert graph["16"]["inputs"]["model"] == ["15", 0]
    assert graph["6"]["inputs"]["model"] == ["16", 0]


def test_sol_profile_defaults_off_and_rejects_unknown_values():
    assert normalize_sol_profile("") == "off"
    assert normalize_sol_profile(" Conservative ") == "conservative"
    with pytest.raises(ValueError, match="sol_profile must be one of"):
        normalize_sol_profile("fast")


def test_task_partitions_follow_sglang_contract():
    assert normalize_task(" REF2VA ") == "ref2va"
    assert task_partition("t2va") == "fl2va"
    assert task_partition("fl2va") == "fl2va"
    assert task_partition("ref2va") == "ref2va"
    with pytest.raises(ValueError, match="task must be one of"):
        normalize_task("i2v")


def test_upload_shape_infers_task_and_rejects_mixed_inputs():
    assert infer_task() == "t2va"
    assert infer_task(first_frame=Path("first.png")) == "fl2va"
    assert infer_task(last_frame=Path("last.png")) == "fl2va"
    assert infer_task(loop=True) == "fl2va"
    assert infer_task(reference_count=1) == "ref2va"
    with pytest.raises(ValueError, match="cannot be combined"):
        infer_task(first_frame=Path("first.png"), reference_count=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"task": "t2va", "first_frame": Path("first.png")},
        {"task": "t2va", "loop": True},
        {"task": "t2va", "reference_count": 1},
        {"task": "fl2va"},
        {"task": "fl2va", "reference_count": 1, "first_frame": Path("first.png")},
        {"task": "fl2va", "loop": True},
        {"task": "fl2va", "first_frame": Path("first.png"), "last_frame": Path("last.png"), "loop": True},
        {"task": "ref2va"},
        {"task": "ref2va", "first_frame": Path("first.png"), "reference_count": 1},
    ],
)
def test_invalid_task_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        validate_inputs(steps=20, seed=1, **kwargs)


def test_fl2va_accepts_last_frame_only():
    validate_inputs(task="fl2va", steps=20, seed=1, last_frame=Path("last.png"))
