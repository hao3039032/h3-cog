import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import weights


def test_license_acceptance_is_explicit(monkeypatch):
    monkeypatch.delenv("MINIMAX_H3_LICENSE_ACCEPTED", raising=False)
    with pytest.raises(RuntimeError, match="reviewing and accepting"):
        weights.ensure_weights()


def test_download_checks_size_sha_and_resumes(tmp_path):
    payload = b"verified-h3-weight" * 1024

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            start = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
            body = payload[start:]
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        target = tmp_path / "model.safetensors"
        partial = target.with_suffix(target.suffix + ".part")
        partial.write_bytes(payload[:37])
        weights._download(
            f"http://127.0.0.1:{server.server_port}/model",
            target,
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        )
        assert target.read_bytes() == payload
        assert not partial.exists()
    finally:
        server.shutdown()


def test_reference_weight_metadata_matches_verified_source():
    entry = weights.VERIFIED_WEIGHTS[weights.REF2VA_RELATIVE]
    assert entry["size"] == 20_970_379_616
    assert entry["sha256"] == "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779"
    assert entry["url"].endswith("/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors")


def test_weight_sources_are_pinned_to_modelscope_without_nvfp4():
    assert set(weights.FILES) == {
        "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        "vae/minimax_h3_video_vae_fp16.safetensors",
        "vae/minimax_h3_audio_vae_fp32.safetensors",
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


def test_weight_installer_never_requests_a_manifest(monkeypatch, tmp_path):
    def blocked(*args, **kwargs):
        raise AssertionError("weight installation must use pinned source URLs")

    monkeypatch.setenv("MINIMAX_H3_LICENSE_ACCEPTED", "1")
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path))
    monkeypatch.setattr(weights, "COMFY_ROOT", tmp_path / "comfy")
    downloaded = []
    monkeypatch.setattr(
        weights,
        "_download",
        lambda url, destination, size, sha256: downloaded.append((url, destination, size, sha256)),
    )
    monkeypatch.setattr(weights.urllib.request, "urlopen", blocked)
    installed = weights.ensure_weights()
    assert set(installed) == set(weights._selected_files())
    expected = {
        path: entry["url"]
        for path, entry in weights.VERIFIED_WEIGHTS.items()
        if path in weights._selected_files()
    }
    assert {
        destination.relative_to(tmp_path / "MiniMax-H3").as_posix(): url
        for url, destination, *_ in downloaded
    } == expected
