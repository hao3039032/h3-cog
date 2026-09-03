from pathlib import Path

import pytest

from h3_tuning import CacheTuning
from h3_workflow import (
    ATTENTION_SAGE,
    FL2VA_PDD_ACC,
    FL2VA_TURBO_LORA,
    INFERENCE_PDD,
    INFERENCE_QUALITY,
    MODEL_QUANTIZATION_INT8,
    MODEL_QUANTIZATION_NVFP4,
    PDD_NFE,
    REF2VA_PDD_ACC,
    REF2VA_TURBO_LORA,
    build_workflow,
    infer_task,
    normalize_attention_backend,
    normalize_inference_mode,
    normalize_model_quantization,
    normalize_task,
    resolve_steps,
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
    assert graph["15"] == {
        "class_type": "MiniMaxH3FusedModulation",
        "inputs": {"model": ["1", 0], "enabled": True},
    }
    assert graph["6"]["inputs"]["model"] == ["15", 0]
    assert graph["9"]["inputs"]["model"] == ["15", 0]
    assert "first_frame" not in graph["5"]["inputs"]
    assert "last_frame" not in graph["5"]["inputs"]


def test_nvfp4_selects_task_matched_dit_and_text_encoder():
    fl2va = build_workflow(
        prompt="p",
        task="t2va",
        width=1344,
        height=768,
        frames=124,
        steps=20,
        seed=7,
        model_quantization=MODEL_QUANTIZATION_NVFP4,
    )
    assert fl2va["1"]["inputs"]["unet_name"] == "minimax_h3_fl2va_pruned_nvfp4.safetensors"
    assert fl2va["2"]["inputs"]["clip_name"] == "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"

    ref2va = build_workflow(
        prompt="Use <Picture 1>.",
        task="ref2va",
        width=768,
        height=1344,
        frames=124,
        steps=20,
        seed=8,
        reference_image_names=["identity.png"],
        model_quantization=MODEL_QUANTIZATION_NVFP4,
    )
    assert ref2va["1"]["inputs"]["unet_name"] == "minimax_h3_ref2va_pruned_nvfp4.safetensors"
    assert ref2va["2"]["inputs"]["clip_name"] == "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"


def test_model_quantization_defaults_to_int8_and_rejects_unknown_values():
    assert normalize_model_quantization(" INT8 ") == MODEL_QUANTIZATION_INT8
    assert normalize_model_quantization("NVFP4") == MODEL_QUANTIZATION_NVFP4
    with pytest.raises(ValueError, match="model_quantization"):
        normalize_model_quantization("fp8")


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
    assert graph["16"] == {"class_type": "LoadImage", "inputs": {"image": "first.png"}}
    assert graph["17"]["class_type"] == "ImageScale"
    assert graph["17"]["inputs"] == {
        "image": ["16", 0],
        "upscale_method": "lanczos",
        "width": 864,
        "height": 480,
        "crop": "center",
    }
    assert graph["18"] == {"class_type": "LoadImage", "inputs": {"image": "last.png"}}
    assert graph["19"]["class_type"] == "ImageScale"
    assert graph["5"]["inputs"]["first_frame"] == ["17", 0]
    assert graph["5"]["inputs"]["last_frame"] == ["19", 0]


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
    assert graph["5"]["inputs"]["first_frame"] == ["17", 0]
    assert graph["5"]["inputs"]["last_frame"] == ["17", 0]
    assert "18" not in graph


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
    assert graph["9"]["inputs"] == {"model": ["15", 0], "scheduler": "simple", "steps": 20, "denoise": 1.0}
    assert graph["13"]["inputs"]["audio"] == ["12", 0]
    assert graph["14"]["inputs"]["codec"] == "h264"
    assert graph["14"]["inputs"]["codec.encoding"] == "re-encode"
    assert graph["14"]["inputs"]["codec.encoding.crf"] == 17.0
    assert graph["16"] == {"class_type": "LoadImage", "inputs": {"image": "identity.png"}}
    assert graph["17"] == {"class_type": "LoadVideo", "inputs": {"file": "motion.mp4"}}
    assert graph["18"] == {"class_type": "GetVideoComponents", "inputs": {"video": ["17", 0]}}
    assert graph["5"]["inputs"]["ref_images.ref_image_0"] == ["16", 0]
    assert graph["5"]["inputs"]["ref_videos.ref_video_0"] == ["18", 0]
    assert graph["5"]["inputs"]["ref_video_audios.ref_video_audio_0"] == ["18", 1]
    assert graph["19"] == {"class_type": "LoadAudio", "inputs": {"audio": "voice.wav"}}
    assert graph["5"]["inputs"]["ref_audios.ref_audio_0"] == ["19", 0]


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
    assert graph["15"]["class_type"] == "MiniMaxH3FusedModulation"
    assert graph["16"]["class_type"] == "EasyCache"
    assert graph["16"]["inputs"]["model"] == ["15", 0]
    assert graph["6"]["inputs"]["model"] == ["16", 0]
    assert graph["9"]["inputs"]["model"] == ["16", 0]
    assert graph["17"] == {"class_type": "LoadImage", "inputs": {"image": "identity.png"}}


def test_fused_modulation_can_be_disabled_for_eager_ab_comparison():
    graph = build_workflow(
        prompt="p",
        task="t2va",
        width=768,
        height=768,
        frames=124,
        steps=20,
        seed=9,
        fused_modulation=False,
    )
    assert all(node["class_type"] != "MiniMaxH3FusedModulation" for node in graph.values())
    assert graph["6"]["inputs"]["model"] == ["1", 0]
    assert graph["9"]["inputs"]["model"] == ["1", 0]


def test_sol_int8_qk_is_an_opt_in_model_patch_with_sage_fallback():
    graph = build_workflow(
        prompt="p",
        task="t2va",
        width=768,
        height=768,
        frames=124,
        steps=20,
        seed=9,
        attention_backend="sol-int8-qk",
    )
    assert graph["15"] == {
        "class_type": "MiniMaxH3ScheduledSolAttentionPatch",
        "inputs": {
            "model": ["1", 0],
            "enabled": True,
            "tau_start": 1.0,
            "tau_end": 0.8,
            "curve": "linear",
            "min_tokens": 4096,
            "strict": False,
            "dense_percent": 0.0,
            "thresh_type": "diag",
            "int8_qk": True,
            "int8_pv": False,
            "sink_conditioning": "exact_kv",
            "dense_blocks": "",
        },
    }
    assert graph["16"] == {
        "class_type": "MiniMaxH3FusedModulation",
        "inputs": {"model": ["15", 0], "enabled": True},
    }
    assert graph["6"]["inputs"]["model"] == ["16", 0]
    assert graph["9"]["inputs"]["model"] == ["16", 0]


def test_attention_backend_defaults_to_sage_and_rejects_unknown_values():
    assert normalize_attention_backend(" SAGE-ATTENTION ") == ATTENTION_SAGE
    with pytest.raises(ValueError, match="attention_backend"):
        validate_inputs(
            task="t2va",
            steps=20,
            seed=1,
            attention_backend="sage-int8",
        )


def test_t2va_turbo_uses_official_fl2v_lora_euler_and_eight_steps():
    graph = build_workflow(
        prompt="p",
        task="t2va",
        width=960,
        height=544,
        frames=124,
        steps=24,
        seed=9,
        inference_mode="turbo",
    )
    assert graph["8"]["inputs"]["sampler_name"] == "euler"
    assert graph["9"]["inputs"] == {
        "model": ["17", 0],
        "scheduler": "simple",
        "steps": 8,
        "denoise": 1.0,
    }
    assert graph["15"] == {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["1", 0],
            "lora_name": FL2VA_TURBO_LORA,
            "strength_model": 1.0,
        },
    }
    assert graph["16"] == {
        "class_type": "MiniMaxH3SigmaShift",
        "inputs": {
            "model": ["15", 0],
            "shift_video": 12.0,
            "shift_audio": 3.0,
        },
    }
    assert graph["17"] == {
        "class_type": "MiniMaxH3FusedModulation",
        "inputs": {"model": ["16", 0], "enabled": True},
    }


def test_ref2va_turbo_uses_official_ref2v_lora_and_four_steps():
    graph = build_workflow(
        prompt="p",
        task="ref2va",
        width=960,
        height=544,
        frames=124,
        steps=24,
        seed=9,
        reference_image_names=["identity.png"],
        inference_mode="turbo",
    )
    assert graph["8"]["inputs"]["sampler_name"] == "euler"
    assert graph["9"]["inputs"]["steps"] == 4
    assert graph["15"]["inputs"]["lora_name"] == REF2VA_TURBO_LORA
    assert graph["16"]["inputs"]["shift_video"] == 12.0
    assert graph["16"]["inputs"]["shift_audio"] == 3.0


def test_inference_mode_defaults_to_quality_and_rejects_unknown_values():
    assert normalize_inference_mode(" QUALITY ") == INFERENCE_QUALITY
    assert resolve_steps("t2va", 24, "quality") == 24
    assert resolve_steps("t2va", 24, "turbo") == 8
    assert resolve_steps("fl2va", 24, "turbo") == 8
    assert resolve_steps("ref2va", 24, "turbo") == 4
    with pytest.raises(ValueError, match="inference_mode"):
        validate_inputs(
            task="t2va",
            steps=20,
            seed=1,
            inference_mode="fast",
        )


def test_t2va_pdd_uses_fl2va_pdd_file_euler_and_apply_sigmas():
    graph = build_workflow(
        prompt="p",
        task="t2va",
        width=960,
        height=544,
        frames=124,
        steps=24,
        seed=9,
        inference_mode="pdd",
    )
    assert graph["8"]["inputs"]["sampler_name"] == "euler"
    assert "9" not in graph
    assert graph["15"] == {
        "class_type": "MiniMaxH3SigmaShift",
        "inputs": {"model": ["1", 0], "shift_video": 12.0, "shift_audio": 3.0},
    }
    assert graph["16"] == {
        "class_type": "MiniMaxH3PDDAccApply",
        "inputs": {
            "model": ["15", 0],
            "pdd_file": FL2VA_PDD_ACC,
            "nfe": "8",
            "lora_strength": 1.0,
            "head_strength": 1.0,
            "on_off_grid": "error",
            "partition_check": "error",
        },
    }
    assert graph["17"] == {
        "class_type": "MiniMaxH3FusedModulation",
        "inputs": {"model": ["16", 0], "enabled": True},
    }
    assert graph["6"]["inputs"]["model"] == ["17", 0]
    assert graph["10"]["inputs"]["sigmas"] == ["16", 1]


def test_fl2va_pdd_matches_fl2va_partition_file():
    graph = build_workflow(
        prompt="p",
        task="fl2va",
        width=960,
        height=544,
        frames=124,
        steps=12,
        seed=9,
        first_image_name="first.png",
        loop=True,
        inference_mode="pdd",
    )
    assert graph["16"]["inputs"]["pdd_file"] == FL2VA_PDD_ACC
    assert graph["10"]["inputs"]["sigmas"] == ["16", 1]


def test_ref2va_pdd_uses_ref2va_pdd_file():
    graph = build_workflow(
        prompt="p",
        task="ref2va",
        width=960,
        height=544,
        frames=124,
        steps=24,
        seed=9,
        reference_image_names=["identity.png"],
        inference_mode="pdd",
    )
    assert graph["16"]["inputs"]["pdd_file"] == REF2VA_PDD_ACC
    assert graph["16"]["inputs"]["model"] == ["15", 0]
    assert graph["10"]["inputs"]["sigmas"] == ["16", 1]


def test_pdd_ignores_requested_steps_and_always_resolves_to_eight():
    assert normalize_inference_mode(" PDD ") == INFERENCE_PDD
    assert PDD_NFE == 8
    for task in ("t2va", "fl2va", "ref2va"):
        assert resolve_steps(task, 24, "pdd") == 8
        assert resolve_steps(task, 60, "pdd") == 8


def test_pdd_rejects_cache_tuning_before_submission():
    cache = CacheTuning("balanced", 0.12, 0.15, 0.9, False, "sweep-1", "balanced")
    with pytest.raises(ValueError, match="incompatible with EasyCache"):
        build_workflow(
            prompt="p",
            task="t2va",
            width=768,
            height=768,
            frames=124,
            steps=24,
            seed=9,
            cache=cache,
            inference_mode="pdd",
        )


def test_pdd_stacks_sol_attention_patch_after_apply_node():
    graph = build_workflow(
        prompt="p",
        task="t2va",
        width=768,
        height=768,
        frames=124,
        steps=24,
        seed=9,
        attention_backend="sol-int8-qk",
        inference_mode="pdd",
    )
    assert graph["15"]["class_type"] == "MiniMaxH3SigmaShift"
    assert graph["16"]["class_type"] == "MiniMaxH3PDDAccApply"
    assert graph["17"]["class_type"] == "MiniMaxH3ScheduledSolAttentionPatch"
    assert graph["17"]["inputs"]["model"] == ["16", 0]
    assert graph["18"]["class_type"] == "MiniMaxH3FusedModulation"
    assert graph["18"]["inputs"]["model"] == ["17", 0]
    assert graph["10"]["inputs"]["sigmas"] == ["16", 1]


def test_pdd_never_inserts_a_turbo_lora_loader():
    for task, kwargs in (
        ("t2va", {}),
        ("fl2va", {"first_image_name": "first.png"}),
        ("ref2va", {"reference_image_names": ["identity.png"]}),
    ):
        graph = build_workflow(
            prompt="p",
            task=task,
            width=768,
            height=768,
            frames=124,
            steps=24,
            seed=9,
            inference_mode="pdd",
            **kwargs,
        )
        assert all(
            node["class_type"] != "LoraLoaderModelOnly" for node in graph.values()
        ), task


def test_pdd_with_fused_modulation_disabled_keeps_apply_directly_on_guider():
    graph = build_workflow(
        prompt="p",
        task="t2va",
        width=768,
        height=768,
        frames=124,
        steps=24,
        seed=9,
        fused_modulation=False,
        inference_mode="pdd",
    )
    assert graph["15"]["class_type"] == "MiniMaxH3SigmaShift"
    assert graph["16"]["class_type"] == "MiniMaxH3PDDAccApply"
    assert "17" not in graph
    assert graph["6"]["inputs"]["model"] == ["16", 0]
    assert graph["10"]["inputs"]["sigmas"] == ["16", 1]


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


@pytest.mark.parametrize("enabled", [0, 1, "true", None])
def test_fused_modulation_requires_a_boolean(enabled):
    with pytest.raises(ValueError, match="fused_modulation"):
        validate_inputs(
            task="t2va",
            steps=20,
            seed=1,
            fused_modulation=enabled,
        )
