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
