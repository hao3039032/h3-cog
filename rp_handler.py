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
        raise ValueError("image URLs must use https")
    for result in socket.getaddrinfo(parsed.hostname, parsed.port or 443):
        address = ipaddress.ip_address(result[4][0])
        if not address.is_global:
            raise ValueError("image URL resolves to a private or reserved address")


def _download_media(url: str) -> Path:
    _public_https(url)
    request = urllib.request.Request(url, headers={"User-Agent": "appnz-h3-cog/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        length = int(response.headers.get("Content-Length", "0") or 0)
        if length > MAX_MEDIA_BYTES:
            raise ValueError("reference media exceeds 512 MiB")
        data = response.read(MAX_MEDIA_BYTES + 1)
    if len(data) > MAX_MEDIA_BYTES:
        raise ValueError("reference media exceeds 512 MiB")
    suffix = Path(urllib.parse.urlparse(url).path).suffix or ".png"
    fd, filename = tempfile.mkstemp(prefix="h3-input-", suffix=suffix)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    return Path(filename)


def handler(event):
    values = event.get("input") or {}
    downloaded: list[Path] = []
    try:
        cache = authorize_tuning(values.get("_tuning"), values.get("_tuning_signature"))
        reference_images = [_download_media(url) for url in media_urls(values, "reference_images")]
        reference_videos = [_download_media(url) for url in media_urls(values, "reference_videos")]
        reference_audios = [_download_media(url) for url in media_urls(values, "reference_audios")]
        downloaded.extend(reference_images + reference_videos + reference_audios)
        output = _get_runtime().generate(
            prompt=values.get("prompt", ""),
            reference_images=reference_images,
            reference_videos=reference_videos,
            reference_audios=reference_audios,
            aspect_ratio=values.get("aspect_ratio", "16:9"),
            size=values.get("size", "preview"),
            duration=float(values.get("duration", 5)),
            steps=int(values.get("steps", 20)),
            seed=values.get("seed"),
            structured_prompt=bool(values.get("structured_prompt", True)),
            include_audio=bool(values.get("include_audio", True)),
            output_codec=values.get("output_codec", "webm-av1"),
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
