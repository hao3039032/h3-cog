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


def test_weight_sources_are_pinned_with_explicit_hf_exceptions():
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
    hf_sourced = set(weights.PDD_ACC_RELATIVES) | set(weights.NVFP4_DIFFUSION_RELATIVES)
    for relative, entry in weights.VERIFIED_WEIGHTS.items():
        if relative in weights.PDD_ACC_RELATIVES:
            assert entry["url"].startswith("https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/resolve/main/")
        elif relative in weights.NVFP4_DIFFUSION_RELATIVES:
            assert entry["url"].startswith(weights.NVFP4_DIT_SOURCE_BASE + "/")
        else:
            assert entry["url"].startswith("https://modelscope.cn/models/")
    assert hf_sourced.isdisjoint(set(weights.FILES))
    assert weights.NVFP4_TEXT_ENCODER_RELATIVE not in weights.FILES


def test_nvfp4_metadata_matches_pinned_single_pass_release():
    text = weights.VERIFIED_WEIGHTS[weights.NVFP4_TEXT_ENCODER_RELATIVE]
    fl2va = weights.VERIFIED_WEIGHTS[weights.FL2VA_NVFP4_RELATIVE]
    ref2va = weights.VERIFIED_WEIGHTS[weights.REF2VA_NVFP4_RELATIVE]
    assert text == {
        "size": 15_687_142_551,
        "sha256": "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6",
        "url": f"{weights.SOURCE_BASE}/{weights.NVFP4_TEXT_ENCODER_RELATIVE}",
    }
    assert fl2va["size"] == 12_528_636_865
    assert ref2va["size"] == 12_528_636_866
    assert fl2va["sha256"] == "6ab7f0c48141e7919b32f925ca3def22e06a6aebeb9e0b6f5a0be0fe8409976f"
    assert ref2va["sha256"] == "3e1be702c95bc057c05a7d1867e8aeea33073dcf5743835f2f27f06a2f34c596"
    assert weights.NVFP4_DIT_REVISION in fl2va["url"]
    assert weights.NVFP4_DIT_REVISION in ref2va["url"]


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


def test_nvfp4_profile_is_linked_lazily_for_one_partition(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_H3_LICENSE_ACCEPTED", "1")
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path))
    monkeypatch.setattr(weights, "COMFY_ROOT", tmp_path / "comfy")
    relatives = weights.model_profile_relatives("nvfp4", "ref2va")
    for relative in relatives:
        destination = tmp_path / "MiniMax-H3" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"nvfp4-data")

    installed = weights.ensure_model_profile("ref2va", "nvfp4")
    assert set(installed) == set(relatives)
    assert weights.FL2VA_NVFP4_RELATIVE not in installed
    for relative, folder in zip(relatives, ("text_encoders", "diffusion_models")):
        target = tmp_path / "comfy" / "models" / folder / Path(relative).name
        assert target.resolve() == installed[relative].resolve()


def test_missing_nvfp4_profile_does_not_block_int8_startup(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_H3_LICENSE_ACCEPTED", "1")
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path))
    monkeypatch.setattr(weights, "COMFY_ROOT", tmp_path / "comfy")
    for relative, folder in weights._selected_files().items():
        destination = tmp_path / "MiniMax-H3" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"int8-data")

    weights.ensure_weights()
    with pytest.raises(FileNotFoundError, match="model_quantization=nvfp4") as error:
        weights.ensure_model_profile("fl2va", "nvfp4")
    assert weights.NVFP4_TEXT_ENCODER_RELATIVE in str(error.value)
    assert weights.FL2VA_NVFP4_RELATIVE in str(error.value)


def test_pdd_acc_metadata_matches_official_huggingface_release():
    fl2va = weights.VERIFIED_WEIGHTS[weights.FL2VA_PDD_ACC_RELATIVE]
    ref2va = weights.VERIFIED_WEIGHTS[weights.REF2VA_PDD_ACC_RELATIVE]
    for entry in (fl2va, ref2va):
        assert entry["size"] == 1_372_450_680
    assert fl2va["sha256"] == "0b29be7042d883970eb0c20774a9ba03d95669ed80a721bb4d21be8ea0d0a196"
    assert ref2va["sha256"] == "111c82e669f6e20e628228172edf39395f1a9fc3ad049793895e542c0f55b18c"
    assert fl2va["url"] == "https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/resolve/main/MiniMax-H3-FL2VA-Acc-8Step.safetensors"
    assert ref2va["url"] == "https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/resolve/main/MiniMax-H3-Ref2VA-Acc-8Step.safetensors"
    assert set(weights.PDD_ACC_RELATIVES).isdisjoint(set(weights.FILES))


def test_pdd_weights_are_lazily_linked_per_partition(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_H3_LICENSE_ACCEPTED", "1")
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path))
    monkeypatch.setattr(weights, "COMFY_ROOT", tmp_path / "comfy")

    fl2va = tmp_path / "MiniMax-H3" / weights.FL2VA_PDD_ACC_RELATIVE
    fl2va.parent.mkdir(parents=True, exist_ok=True)
    fl2va.write_bytes(b"pdd-data")

    installed = weights.ensure_pdd_weight("fl2va")
    assert installed == fl2va
    target = tmp_path / "comfy" / "models" / "pdd_acc" / "MiniMax-H3-FL2VA-Acc-8Step.safetensors"
    assert target.resolve() == fl2va.resolve()

    with pytest.raises(FileNotFoundError, match=weights.REF2VA_PDD_ACC_RELATIVE):
        weights.ensure_pdd_weight("ref2va")


def test_missing_pdd_weight_does_not_block_startup(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_H3_LICENSE_ACCEPTED", "1")
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path))
    monkeypatch.setattr(weights, "COMFY_ROOT", tmp_path / "comfy")

    for relative, folder in weights._selected_files().items():
        destination = tmp_path / "MiniMax-H3" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"operational-data")

    weights.ensure_weights()
    with pytest.raises(FileNotFoundError, match="PDD Acc weight is missing"):
        weights.ensure_pdd_weight("fl2va")


def test_pdd_partition_mapping_rejects_unknown_partitions():
    assert weights.pdd_weight_relative("fl2va") == weights.FL2VA_PDD_ACC_RELATIVE
    assert weights.pdd_weight_relative("ref2va") == weights.REF2VA_PDD_ACC_RELATIVE
    with pytest.raises(ValueError, match="unknown partition"):
        weights.pdd_weight_relative("t2va")


def test_pdd_weights_require_license_acceptance(monkeypatch):
    monkeypatch.delenv("MINIMAX_H3_LICENSE_ACCEPTED", raising=False)
    with pytest.raises(RuntimeError, match="reviewing and accepting"):
        weights.ensure_pdd_weight("fl2va")
