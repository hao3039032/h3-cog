import hashlib
import inspect
import urllib.error
from pathlib import Path

import pytest

import h3_runtime
from h3_runtime import GenerationResult, H3Runtime


def test_generation_metrics_cover_output_identity_and_task_route(tmp_path, monkeypatch):
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"raw")
    encoded = tmp_path / "result.webm"
    encoded.write_bytes(b"encoded-video")
    runtime = object.__new__(H3Runtime)
    runtime.dit_switch_policy = "auto"
    runtime.current_task = None
    runtime.current_partition = None
    monkeypatch.setattr(runtime, "_stage_media", lambda path, label, index: "identity.png")
    monkeypatch.setattr(runtime, "_history_output", lambda entry: raw)
    monkeypatch.setattr(h3_runtime.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(h3_runtime, "encode_video", lambda *args: encoded)

    def fake_json_request(path, payload=None, timeout=60):
        if path == "/prompt":
            assert payload["prompt"]["1"]["inputs"]["unet_name"] == (
                "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
            )
            assert payload["prompt"]["5"]["inputs"]["prompt"] == "test"
            assert payload["prompt"]["15"] == {
                "class_type": "MiniMaxH3FusedModulation",
                "inputs": {"model": ["1", 0], "enabled": True},
            }
            return {"prompt_id": "job-1"}
        return {"job-1": {"status": {"completed": True}, "outputs": {}}}

    monkeypatch.setattr(h3_runtime, "_json_request", fake_json_request)
    result = runtime.generate(
        prompt="test",
        task="ref2va",
        reference_images=[tmp_path / "identity.png"],
        duration=4,
        seed=42,
        return_metrics=True,
    )
    assert isinstance(result, GenerationResult)
    assert result.path == encoded
    assert result.metrics["seed"] == 42
    assert result.metrics["task"] == "ref2va"
    assert result.metrics["partition"] == "ref2va"
    assert result.metrics["dit_switch_policy"] == "auto"
    assert result.metrics["cache"] == {"profile": "off"}
    assert result.metrics["attention_backend"] == "sage-attention"
    assert result.metrics["inference_mode"] == "quality"
    assert result.metrics["requested_steps"] == 24
    assert result.metrics["fused_modulation"] is True
    assert result.metrics["output_bytes"] == len(b"encoded-video")
    assert result.metrics["output_sha256"] == hashlib.sha256(b"encoded-video").hexdigest()
    assert result.metrics["generation_seconds"] >= 0
    assert result.metrics["encode_seconds"] >= 0


def test_product_defaults_are_vertical_preview_mp4():
    defaults = inspect.signature(H3Runtime.generate).parameters
    assert defaults["task"].default is inspect.Parameter.empty
    assert defaults["steps"].default == 24
    assert defaults["attention_backend"].default == "sage-attention"
    assert defaults["inference_mode"].default == "quality"
    assert defaults["fused_modulation"].default is True
    assert defaults["aspect_ratio"].default == "9:16"
    assert defaults["size"].default == "preview"
    assert defaults["structured_prompt"].default is None
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


def test_comfy_always_uses_dynamic_vram_below_80gb(monkeypatch):
    monkeypatch.delenv("H3_LOWVRAM", raising=False)
    monkeypatch.delenv("H3_HIGHVRAM", raising=False)
    monkeypatch.delenv("H3_RESERVE_VRAM_GB", raising=False)
    monkeypatch.delenv("H3_VIDEO_VAE_PRECISION", raising=False)
    monkeypatch.setattr(h3_runtime, "_gpu_memory_gib", lambda: 24.0)
    monkeypatch.setattr(h3_runtime, "_gpu_count", lambda: 1)
    command = h3_runtime._comfy_command()
    assert "--lowvram" not in command
    assert "--highvram" not in command
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
    assert "--lowvram" not in command
    assert "--highvram" not in command
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


def test_large_single_gpu_uses_highvram_with_manual_override(monkeypatch):
    monkeypatch.delenv("H3_LOWVRAM", raising=False)
    monkeypatch.delenv("H3_HIGHVRAM", raising=False)
    monkeypatch.setattr(h3_runtime, "_gpu_memory_gib", lambda: 84.0)
    monkeypatch.setattr(h3_runtime, "_gpu_count", lambda: 1)
    assert "--highvram" in h3_runtime._comfy_command()

    monkeypatch.setenv("H3_HIGHVRAM", "0")
    command = h3_runtime._comfy_command()
    assert "--highvram" not in command
    assert "--lowvram" not in command


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


def test_dit_switch_policy_defaults_to_auto_and_validates_values(monkeypatch):
    monkeypatch.delenv("H3_DIT_SWITCH_POLICY", raising=False)
    assert h3_runtime._dit_switch_policy() == "auto"
    monkeypatch.setenv("H3_DIT_SWITCH_POLICY", "evict")
    assert h3_runtime._dit_switch_policy() == "evict"
    monkeypatch.setenv("H3_DIT_SWITCH_POLICY", "free")
    with pytest.raises(ValueError, match="auto or evict"):
        h3_runtime._dit_switch_policy()


def test_auto_partition_switch_does_not_call_comfy_free(monkeypatch):
    runtime = object.__new__(H3Runtime)
    runtime.dit_switch_policy = "auto"
    runtime.current_task = "ref2va"
    runtime.current_partition = "ref2va"
    calls = []

    def fake_request(path, payload=None, timeout=60):
        calls.append((path, payload))
        return {}

    monkeypatch.setattr(h3_runtime, "_json_request", fake_request)
    runtime._prepare_generation("fl2va", "fl2va")
    assert calls == []
    assert runtime.current_task == "fl2va"
    assert runtime.current_partition == "fl2va"


def test_evict_policy_frees_only_on_partition_change(monkeypatch):
    runtime = object.__new__(H3Runtime)
    runtime.dit_switch_policy = "evict"
    runtime.current_task = "ref2va"
    runtime.current_partition = "ref2va"
    calls = []

    def fake_request(path, payload=None, timeout=60):
        calls.append((path, payload))
        return {}

    monkeypatch.setattr(h3_runtime, "_json_request", fake_request)
    runtime._prepare_generation("t2va", "fl2va")
    runtime._prepare_generation("fl2va", "fl2va")
    assert calls == [("/free", {"unload_models": True, "free_memory": True})]


def test_native_parallel_mode_rejects_retired_raylight_modes(monkeypatch):
    monkeypatch.delenv("H3_PARALLEL_MODE", raising=False)
    assert h3_runtime.select_parallel_mode("single") == "single"
    assert h3_runtime.select_parallel_mode("native") == "single"
    assert h3_runtime.select_parallel_mode("") == "single"
    for retired in ("raylight", "auto", "fsdp", "dual"):
        with pytest.raises(ValueError, match="Raylight has been removed"):
            h3_runtime.select_parallel_mode(retired)


def test_generate_requires_an_explicit_task():
    runtime = object.__new__(H3Runtime)
    with pytest.raises(ValueError, match="task must be one of"):
        runtime.generate(prompt="test", task="")


def test_generation_stages_and_cleans_fl_keyframes(tmp_path, monkeypatch):
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"raw")
    encoded = tmp_path / "result.mp4"
    encoded.write_bytes(b"encoded")
    first = tmp_path / "first.png"
    first.write_bytes(b"first")
    staged = []

    runtime = object.__new__(H3Runtime)
    runtime.dit_switch_policy = "auto"
    runtime.current_task = "t2va"
    runtime.current_partition = "fl2va"
    monkeypatch.setattr(h3_runtime, "COMFY_ROOT", tmp_path)
    monkeypatch.setattr(runtime, "_history_output", lambda entry: raw)
    monkeypatch.setattr(h3_runtime.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(h3_runtime, "encode_video", lambda *args: encoded)

    def fake_json_request(path, payload=None, timeout=60):
        if path == "/prompt":
            staged.append(payload["prompt"]["16"]["inputs"]["image"])
            assert payload["prompt"]["5"]["inputs"]["prompt"].startswith(
                "How the reference pictures align"
            )
            return {"prompt_id": "job-1"}
        return {"job-1": {"status": {"completed": True}, "outputs": {}}}

    monkeypatch.setattr(h3_runtime, "_json_request", fake_json_request)
    result = runtime.generate(
        prompt="test",
        task="fl2va",
        first_frame=first,
        loop=True,
        duration=4,
        steps=8,
        seed=1,
        return_metrics=True,
    )
    assert isinstance(result, GenerationResult)
    assert result.metrics["task"] == "fl2va"
    assert len(staged) == 1
    assert staged[0].startswith("h3-first-frame-")
    assert list((tmp_path / "input").iterdir()) == []


def test_pdd_generation_routes_apply_sigmas_and_records_metrics(tmp_path, monkeypatch):
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"raw")
    encoded = tmp_path / "result.webm"
    encoded.write_bytes(b"encoded-video")

    runtime = object.__new__(H3Runtime)
    runtime.dit_switch_policy = "auto"
    runtime.current_task = None
    runtime.current_partition = None
    monkeypatch.setattr(runtime, "_stage_media", lambda path, label, index: "identity.png")
    monkeypatch.setattr(runtime, "_history_output", lambda entry: raw)
    monkeypatch.setattr(h3_runtime.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(h3_runtime, "encode_video", lambda *args: encoded)
    linked_partitions = []
    monkeypatch.setattr(
        h3_runtime,
        "ensure_pdd_weight",
        lambda partition: linked_partitions.append(partition) or Path("/weights/pdd"),
        )
    monkeypatch.setattr(h3_runtime, "_pdd_nodes_verified", False)

    def fake_json_request(path, payload=None, timeout=60):
        if path.startswith("/object_info/"):
            return {path.rsplit("/", 1)[-1]: {}}
        if path == "/prompt":
            graph = payload["prompt"]
            assert graph["8"]["inputs"]["sampler_name"] == "euler"
            assert "9" not in graph
            assert graph["15"]["class_type"] == "MiniMaxH3SigmaShift"
            assert graph["16"]["class_type"] == "MiniMaxH3PDDAccApply"
            assert graph["16"]["inputs"]["pdd_file"] == "MiniMax-H3-Ref2VA-Acc-8Step.safetensors"
            assert graph["16"]["inputs"]["nfe"] == "8"
            assert graph["10"]["inputs"]["sigmas"] == ["16", 1]
            return {"prompt_id": "job-1"}
        return {"job-1": {"status": {"completed": True}, "outputs": {}}}

    monkeypatch.setattr(h3_runtime, "_json_request", fake_json_request)
    result = runtime.generate(
        prompt="test",
        task="ref2va",
        reference_images=[tmp_path / "identity.png"],
        duration=4,
        steps=24,
        seed=42,
        inference_mode="pdd",
        return_metrics=True,
    )
    assert result.metrics["inference_mode"] == "pdd"
    assert result.metrics["steps"] == 8
    assert result.metrics["requested_steps"] == 24
    assert result.metrics["task"] == "ref2va"
    assert result.metrics["schema_version"] == 2
    # The partition weight link happens inside the route lock, before /prompt.
    assert linked_partitions == ["ref2va"]


def test_pdd_generation_rejects_cache_before_submission(monkeypatch, tmp_path):
    from h3_tuning import CacheTuning

    runtime = object.__new__(H3Runtime)
    runtime.dit_switch_policy = "auto"
    runtime.current_task = None
    runtime.current_partition = None
    calls = []

    def fail_request(path, payload=None, timeout=60):
        calls.append(path)
        raise AssertionError("ComfyUI must not be contacted for a rejected PDD+cache request")

    monkeypatch.setattr(h3_runtime, "_json_request", fail_request)
    cache = CacheTuning("balanced", 0.12, 0.15, 0.9, False, "sweep-1", "balanced")
    with pytest.raises(ValueError, match="incompatible with EasyCache"):
        runtime.generate(
            prompt="test",
            task="t2va",
            duration=4,
            steps=24,
            seed=1,
            inference_mode="pdd",
            cache=cache,
        )
    assert calls == []


def test_pdd_node_availability_is_verified_via_object_info(monkeypatch):
    calls = []

    def fake_json_request(path, payload=None, timeout=60):
        calls.append(path)
        if "/object_info/" in path:
            name = path.rsplit("/", 1)[-1]
            return {name: {}}
        raise AssertionError(f"unexpected request: {path}")

    monkeypatch.setattr(h3_runtime, "_json_request", fake_json_request)
    monkeypatch.setattr(h3_runtime, "_pdd_nodes_verified", False)
    h3_runtime.verify_pdd_nodes()
    assert calls == [
        "/object_info/MiniMaxH3PDDAccApply",
        "/object_info/MiniMaxH3SigmaShift",
    ]
    # A successful probe is cached for the process lifetime.
    h3_runtime.verify_pdd_nodes()
    assert len(calls) == 2


def test_pdd_node_verification_reports_missing_custom_node(monkeypatch):
    def fake_json_request(path, payload=None, timeout=60):
        name = path.rsplit("/", 1)[-1]
        return {name: {}} if name == "MiniMaxH3SigmaShift" else {}

    monkeypatch.setattr(h3_runtime, "_json_request", fake_json_request)
    monkeypatch.setattr(h3_runtime, "_pdd_nodes_verified", False)
    with pytest.raises(RuntimeError, match="MiniMaxH3PDDAccApply"):
        h3_runtime.verify_pdd_nodes()


def test_pdd_node_verification_wraps_transport_and_json_failures(monkeypatch):
    import json as json_module

    monkeypatch.setattr(h3_runtime, "_pdd_nodes_verified", False)

    def raise_url_error(path, payload=None, timeout=60):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(h3_runtime, "_json_request", raise_url_error)
    with pytest.raises(RuntimeError, match="could not query ComfyUI object_info"):
        h3_runtime.verify_pdd_nodes()

    def raise_json_error(path, payload=None, timeout=60):
        raise json_module.JSONDecodeError("bad payload", "doc", 0)

    monkeypatch.setattr(h3_runtime, "_json_request", raise_json_error)
    with pytest.raises(RuntimeError, match="could not query ComfyUI object_info"):
        h3_runtime.verify_pdd_nodes()


def test_pdd_missing_weight_fails_with_explicit_path_before_sampling(monkeypatch, tmp_path):
    runtime = object.__new__(H3Runtime)
    runtime.dit_switch_policy = "auto"
    runtime.current_task = None
    runtime.current_partition = None
    monkeypatch.setenv("MINIMAX_H3_LICENSE_ACCEPTED", "1")
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path))
    monkeypatch.setattr(h3_runtime, "COMFY_ROOT", tmp_path / "comfy")
    monkeypatch.setattr(h3_runtime, "_pdd_nodes_verified", False)

    calls = []

    def fake_json_request(path, payload=None, timeout=60):
        calls.append(path)
        if path.startswith("/object_info/"):
            return {path.rsplit("/", 1)[-1]: {}}
        raise AssertionError("sampling must not start when the PDD weight is missing")

    monkeypatch.setattr(h3_runtime, "_json_request", fake_json_request)
    with pytest.raises(FileNotFoundError, match="pdd_acc"):
        runtime.generate(
            prompt="test",
            task="t2va",
            duration=4,
            steps=24,
            seed=1,
            inference_mode="pdd",
        )
    assert all(call.startswith("/object_info/") for call in calls)
