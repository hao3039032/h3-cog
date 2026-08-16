import hashlib

import inspect

import h3_runtime
from h3_runtime import GenerationResult, H3Runtime


def test_generation_metrics_cover_output_identity_and_lossless_default(tmp_path, monkeypatch):
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"raw")
    encoded = tmp_path / "result.webm"
    encoded.write_bytes(b"encoded-video")
    runtime = object.__new__(H3Runtime)
    monkeypatch.setattr(runtime, "_stage_media", lambda path, label, index: "identity.png")
    monkeypatch.setattr(runtime, "_history_output", lambda entry: raw)
    monkeypatch.setattr(h3_runtime.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(h3_runtime, "encode_video", lambda *args: encoded)

    def fake_json_request(path, payload=None, timeout=60):
        if path == "/prompt":
            return {"prompt_id": "job-1"}
        return {"job-1": {"status": {"completed": True}, "outputs": {}}}

    monkeypatch.setattr(h3_runtime, "_json_request", fake_json_request)
    result = runtime.generate(
        prompt="test", reference_images=[tmp_path / "identity.png"],
        duration=4, seed=42, return_metrics=True,
    )
    assert isinstance(result, GenerationResult)
    assert result.path == encoded
    assert result.metrics["seed"] == 42
    assert result.metrics["cache"] == {"profile": "off"}
    assert result.metrics["output_bytes"] == len(b"encoded-video")
    assert result.metrics["output_sha256"] == hashlib.sha256(b"encoded-video").hexdigest()
    assert result.metrics["generation_seconds"] >= 0
    assert result.metrics["encode_seconds"] >= 0
def test_ref2va_product_defaults_are_vertical_preview_mp4():
    defaults = inspect.signature(h3_runtime.H3Runtime.generate).parameters
    assert defaults["steps"].default == 24
    assert defaults["aspect_ratio"].default == "9:16"
    assert defaults["size"].default == "preview"
    assert defaults["structured_prompt"].default is False
    assert defaults["output_codec"].default == "mp4-h264"


def test_comfy_error_reports_exception_without_dumping_tensor_inputs():
    status = {"messages": [["execution_error", {
        "node_type": "MiniMaxH3ReferenceToVideo",
        "exception_type": "RuntimeError",
        "exception_message": "CUDA out of memory",
        "current_inputs": {"ref_image_1": "tensor(very large)"},
    }]]}
    message = h3_runtime._comfy_error(status)
    assert message == "RuntimeError in MiniMaxH3ReferenceToVideo: CUDA out of memory"
    assert "tensor" not in message


def test_comfy_defaults_to_dynamic_normal_vram_with_emergency_lowvram_switch(monkeypatch):
    monkeypatch.delenv("H3_LOWVRAM", raising=False)
    monkeypatch.delenv("H3_HIGHVRAM", raising=False)
    monkeypatch.delenv("H3_RESERVE_VRAM_GB", raising=False)
    monkeypatch.delenv("H3_VIDEO_VAE_PRECISION", raising=False)
    monkeypatch.setattr(h3_runtime, "_gpu_name", lambda: "NVIDIA L40S")
    monkeypatch.setattr(h3_runtime, "_gpu_memory_gib", lambda: 48.0)
    monkeypatch.setattr(h3_runtime, "_gpu_count", lambda: 1)
    command = h3_runtime._comfy_command()
    assert "--lowvram" not in command
    assert "--disable-dynamic-vram" not in command
    assert "--fp32-vae" in command

    monkeypatch.setenv("H3_VIDEO_VAE_PRECISION", "fp16")
    command = h3_runtime._comfy_command()
    assert "--fp32-vae" not in command
    assert command[command.index("--reserve-vram") + 1] == "1.0"

    monkeypatch.setenv("H3_LOWVRAM", "1")
    monkeypatch.setenv("H3_HIGHVRAM", "0")
    monkeypatch.setenv("H3_RESERVE_VRAM_GB", "3")
    command = h3_runtime._comfy_command()
    assert "--lowvram" in command
    assert "--highvram" not in command
    assert "--disable-dynamic-vram" not in command
    assert command[command.index("--reserve-vram") + 1] == "3"


def test_fp32_matmul_tf32_is_an_explicit_comfy_process_switch(monkeypatch):
    monkeypatch.delenv("H3_FP32_MATMUL_TF32", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")
    assert h3_runtime._fp32_matmul_precision() == "strict-fp32"
    assert h3_runtime._comfy_environment()["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] == "0"
    assert h3_runtime._comfy_environment()["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"

    monkeypatch.setenv("H3_FP32_MATMUL_TF32", "1")
    assert h3_runtime._fp32_matmul_precision() == "tf32"
    assert h3_runtime._comfy_environment()["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] == "1"


def test_large_single_gpu_uses_highvram_with_manual_overrides(monkeypatch):
    monkeypatch.delenv("H3_LOWVRAM", raising=False)
    monkeypatch.delenv("H3_HIGHVRAM", raising=False)
    monkeypatch.setattr(h3_runtime, "_gpu_memory_gib", lambda: 84.0)
    monkeypatch.setattr(h3_runtime, "_gpu_count", lambda: 1)
    command = h3_runtime._comfy_command()
    assert "--highvram" in command
    assert "--lowvram" not in command

    monkeypatch.setenv("H3_HIGHVRAM", "0")
    command = h3_runtime._comfy_command()
    assert "--highvram" not in command
    assert "--lowvram" not in command

    monkeypatch.setenv("H3_LOWVRAM", "1")
    command = h3_runtime._comfy_command()
    assert "--highvram" not in command
    assert "--lowvram" in command


def test_single_24gb_gpu_automatically_uses_safe_lowvram_mode(monkeypatch):
    monkeypatch.delenv("H3_LOWVRAM", raising=False)
    monkeypatch.delenv("H3_HIGHVRAM", raising=False)
    monkeypatch.setattr(h3_runtime, "_gpu_memory_gib", lambda: 24.0)
    monkeypatch.setattr(h3_runtime, "_gpu_count", lambda: 1)
    assert "--lowvram" in h3_runtime._comfy_command()

    monkeypatch.setenv("H3_HIGHVRAM", "0")
    assert "--lowvram" in h3_runtime._comfy_command()

    monkeypatch.delenv("H3_HIGHVRAM", raising=False)
    monkeypatch.setenv("H3_LOWVRAM", "0")
    assert "--lowvram" not in h3_runtime._comfy_command()


def test_sage_attention_covers_sm80_sm89_sm120_and_rejects_sm90(monkeypatch):
    class FakeCuda:
        available = True
        capability = (8, 9)

        @classmethod
        def is_available(cls):
            return cls.available

        @classmethod
        def get_device_capability(cls, _index):
            return cls.capability

    class FakeTorch:
        cuda = FakeCuda

    monkeypatch.setitem(__import__("sys").modules, "torch", FakeTorch)
    assert h3_runtime._sage_attention_supported() is True

    FakeCuda.capability = (8, 0)
    assert h3_runtime._sage_attention_supported() is True

    FakeCuda.capability = (12, 0)
    assert h3_runtime._sage_attention_supported() is True

    FakeCuda.capability = (9, 0)
    assert h3_runtime._sage_attention_supported() is False


def test_sage_attention_can_be_disabled_for_same_seed_ab_tests(monkeypatch):
    monkeypatch.setenv("H3_SAGE_ATTENTION", "0")
    assert h3_runtime._sage_attention_supported() is False


def test_parallel_mode_defaults_to_single_gpu_and_keeps_raylight_opt_in():
    assert h3_runtime.select_parallel_mode(gpu_count=2) == "single"
    assert h3_runtime.select_parallel_mode("auto", gpu_count=1) == "single"
    assert h3_runtime.select_parallel_mode("auto", gpu_count=2) == "raylight"
    assert h3_runtime.select_parallel_mode("single", gpu_count=2) == "single"
    assert h3_runtime.select_parallel_mode("fsdp", gpu_count=2) == "raylight"


def test_explicit_raylight_rejects_a_single_4090():
    import pytest

    with pytest.raises(RuntimeError, match="requires 2 visible GPUs"):
        h3_runtime.select_parallel_mode("raylight", gpu_count=1)
