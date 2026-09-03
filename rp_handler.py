"""RunPod Serverless entry point using the exact same H3 runtime as Cog."""

from __future__ import annotations

import base64
import ipaddress
import os
import socket
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import runpod

from h3_runtime import H3Runtime
from h3_serverless import media_urls
from h3_tuning import authorize_tuning

_runtime = None
MAX_MEDIA_BYTES = 512 * 1024 * 1024


def _get_runtime() -> H3Runtime:
    global _runtime
    if _runtime is None:
        _runtime = H3Runtime()
    return _runtime


def _public_https(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("media URLs must use https")
    for result in socket.getaddrinfo(parsed.hostname, parsed.port or 443):
        address = ipaddress.ip_address(result[4][0])
        if not address.is_global:
            raise ValueError("media URL resolves to a private or reserved address")


def _download_media(url: str) -> Path:
    _public_https(url)
    request = urllib.request.Request(url, headers={"User-Agent": "appnz-h3-cog/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        length = int(response.headers.get("Content-Length", "0") or 0)
        if length > MAX_MEDIA_BYTES:
            raise ValueError("media exceeds 512 MiB")
        data = response.read(MAX_MEDIA_BYTES + 1)
    if len(data) > MAX_MEDIA_BYTES:
        raise ValueError("media exceeds 512 MiB")
    suffix = Path(urllib.parse.urlparse(url).path).suffix or ".png"
    fd, filename = tempfile.mkstemp(prefix="h3-input-", suffix=suffix)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    return Path(filename)


def _boolean(values: dict, name: str, default: bool) -> bool:
    value = values.get(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _optional_boolean(values: dict, name: str) -> bool | None:
    if values.get(name) is None:
        return None
    return _boolean(values, name, False)


def _download_one(values: dict, name: str) -> Path | None:
    urls = media_urls(values, name)
    if len(urls) > 1:
        raise ValueError(f"{name} accepts at most one URL")
    return _download_media(urls[0]) if urls else None


def handler(event):
    values = event.get("input") or {}
    downloaded: list[Path] = []
    try:
        cache = authorize_tuning(values.get("_tuning"), values.get("_tuning_signature"))
        first_frame = _download_one(values, "first_frame")
        last_frame = _download_one(values, "last_frame")
        reference_images = [_download_media(url) for url in media_urls(values, "reference_images")]
        reference_videos = [_download_media(url) for url in media_urls(values, "reference_videos")]
        reference_audios = [_download_media(url) for url in media_urls(values, "reference_audios")]
        downloaded.extend(
            [path for path in (first_frame, last_frame) if path is not None]
            + reference_images + reference_videos + reference_audios
        )
        output = _get_runtime().generate(
            prompt=values.get("prompt", ""),
            task=values.get("task"),
            first_frame=first_frame,
            last_frame=last_frame,
            loop=_boolean(values, "loop", False),
            reference_images=reference_images,
            reference_videos=reference_videos,
            reference_audios=reference_audios,
            aspect_ratio=values.get("aspect_ratio", "9:16"),
            size=values.get("size", "preview"),
            duration=float(values.get("duration", 5)),
            steps=int(values.get("steps", 24)),
            inference_mode=values.get("inference_mode", "quality"),
            model_quantization=values.get("model_quantization", "int8"),
            attention_backend=values.get("attention_backend", "sage-attention"),
            fused_modulation=_boolean(values, "fused_modulation", True),
            seed=values.get("seed"),
            structured_prompt=_optional_boolean(values, "structured_prompt"),
            include_audio=_boolean(values, "include_audio", True),
            output_codec=values.get("output_codec", "mp4-h264"),
            encode_quality=int(values.get("encode_quality", 26)),
            cache=cache,
            return_metrics=True,
        )
        data = base64.b64encode(output.path.read_bytes()).decode()
        content_type = "video/webm" if output.path.suffix == ".webm" else "video/mp4"
        return {
            "outputs": [{"filename": output.path.name, "data": data, "content_type": content_type}],
            "metrics": output.metrics,
        }
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        for path in downloaded:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
