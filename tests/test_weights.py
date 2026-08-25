from pathlib import Path

import pytest

import weights


def test_license_acceptance_is_explicit(monkeypatch):
    monkeypatch.delenv("MINIMAX_H3_LICENSE_ACCEPTED", raising=False)
    with pytest.raises(RuntimeError, match="reviewing and accepting"):
        weights.ensure_weights()


def test_missing_weights_fail_with_explicit_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_H3_LICENSE_ACCEPTED", "1")
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="Required MiniMax H3 weights are missing") as error:
        weights.ensure_weights()
    for relative in weights._selected_files():
        assert relative in str(error.value)


def test_reference_weight_metadata_matches_verified_source():
    entry = weights.VERIFIED_WEIGHTS[weights.REF2VA_RELATIVE]
    assert entry["size"] == 20_970_379_616
    assert entry["sha256"] == "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779"
    assert entry["url"].endswith("/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors")


def test_fl2va_weight_metadata_matches_verified_source():
    entry = weights.VERIFIED_WEIGHTS[weights.FL2VA_RELATIVE]
    assert entry["size"] == 20_970_379_616
    assert entry["sha256"] == "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
    assert entry["url"].endswith("/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors")


def test_weight_sources_are_pinned_to_modelscope_without_nvfp4():
    assert set(weights.FILES) == {
        "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        "vae/minimax_h3_video_vae_fp16.safetensors",
        "vae/minimax_h3_audio_vae_fp32.safetensors",
        weights.FL2VA_TURBO_LORA_RELATIVE,
        weights.REF2VA_TURBO_LORA_RELATIVE,
    }
    entry = weights.VERIFIED_WEIGHTS[weights.TEXT_ENCODER_RELATIVE]
    assert entry["size"] == 27_141_342_152
    assert entry["sha256"] == "bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923"
    assert entry["url"] == "https://modelscope.cn/models/Comfy-Org/MiniMax-H3/resolve/master/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
    assert all(entry["url"].startswith("https://modelscope.cn/models/") for entry in weights.VERIFIED_WEIGHTS.values())
    assert "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" not in weights.VERIFIED_WEIGHTS


def test_fp32_video_vae_uses_repackaged_modelscope_source_by_default(monkeypatch):
    monkeypatch.delenv("H3_VIDEO_VAE_PRECISION", raising=False)
    assert weights.video_vae_precision() == "fp32"
    assert weights.video_vae_relative() == weights.VIDEO_VAE_FP32_RELATIVE
    assert weights.video_vae_filename() == "minimax_h3_video_vae_fp32.safetensors"
    assert set(weights._selected_files()) == {
        weights.TEXT_ENCODER_RELATIVE,
        weights.VIDEO_VAE_FP32_RELATIVE,
        "vae/minimax_h3_audio_vae_fp32.safetensors",
        weights.FL2VA_TURBO_LORA_RELATIVE,
        weights.REF2VA_TURBO_LORA_RELATIVE,
    }
    entry = weights.VERIFIED_WEIGHTS[weights.VIDEO_VAE_FP32_RELATIVE]
    assert entry["size"] == 10_415_548_688
    assert entry["sha256"] == "a28fa965eb65a3fe1279a8bf73f01dddaa36ecd039d08751f74bc8849e88767b"
    assert entry["url"] == "https://modelscope.cn/models/Austusm/minimax_h3_video_vae/resolve/master/minimax_h3_video_vae_fp32.safetensors"

    monkeypatch.setenv("H3_VIDEO_VAE_PRECISION", "fp16")
    assert weights.video_vae_relative() == weights.VIDEO_VAE_FP16_RELATIVE

    monkeypatch.setenv("H3_VIDEO_VAE_PRECISION", "bf16")
    with pytest.raises(ValueError, match="fp16 or fp32"):
        weights.video_vae_precision()


def test_turbo_lora_metadata_matches_official_lightx2v_files():
    fl2v = weights.VERIFIED_WEIGHTS[weights.FL2VA_TURBO_LORA_RELATIVE]
    ref2v = weights.VERIFIED_WEIGHTS[weights.REF2VA_TURBO_LORA_RELATIVE]
    assert fl2v["size"] == 1_956_193_000
    assert fl2v["sha256"] == "2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e"
    assert ref2v["size"] == 1_956_193_000
    assert ref2v["sha256"] == "5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c"
    assert fl2v["url"].endswith("/" + weights.FL2VA_TURBO_LORA_RELATIVE)
    assert ref2v["url"].endswith("/" + weights.REF2VA_TURBO_LORA_RELATIVE)


def test_existing_weights_are_linked_without_validation(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_H3_LICENSE_ACCEPTED", "1")
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path))
    monkeypatch.setattr(weights, "COMFY_ROOT", tmp_path / "comfy")

    for relative, folder in weights._selected_files().items():
        destination = tmp_path / "MiniMax-H3" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"operational-data")

    installed = weights.ensure_weights()
    assert set(installed) == set(weights._selected_files())
    for relative, folder in weights._selected_files().items():
        target = tmp_path / "comfy" / "models" / folder / Path(relative).name
        assert target.resolve() == installed[relative].resolve()


def test_missing_reference_weight_fails_with_explicit_path(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_H3_LICENSE_ACCEPTED", "1")
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError, match=weights.REF2VA_RELATIVE):
        weights.ensure_reference_weight()


def test_both_diffusion_weights_are_linked_for_task_routing(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_H3_LICENSE_ACCEPTED", "1")
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path))
    monkeypatch.setattr(weights, "COMFY_ROOT", tmp_path / "comfy")

    for relative in weights.DIFFUSION_RELATIVES:
        destination = tmp_path / "MiniMax-H3" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"operational-data")

    installed = weights.ensure_diffusion_weights()
    assert set(installed) == set(weights.DIFFUSION_RELATIVES)
    for relative in weights.DIFFUSION_RELATIVES:
        target = tmp_path / "comfy" / "models" / "diffusion_models" / Path(relative).name
        assert target.resolve() == installed[relative].resolve()
