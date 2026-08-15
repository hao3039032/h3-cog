"""Verified ModelScope MiniMax H3 weight installation for ComfyUI."""

from __future__ import annotations

import fcntl
import hashlib
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SOURCE_BASE = "https://modelscope.cn/models/Comfy-Org/MiniMax-H3/resolve/master"
REPACKAGED_VAE_SOURCE_BASE = "https://modelscope.cn/models/Austusm/minimax_h3_video_vae/resolve/master"
COMFY_ROOT = Path(os.getenv("COMFY_ROOT", "/opt/ComfyUI"))
TEXT_ENCODER_RELATIVE = "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
REF2VA_RELATIVE = "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
VIDEO_VAE_FP16_RELATIVE = "vae/minimax_h3_video_vae_fp16.safetensors"
VIDEO_VAE_FP32_RELATIVE = "vae/minimax_h3_video_vae_fp32.safetensors"
FILES = {
    TEXT_ENCODER_RELATIVE: "text_encoders",
    VIDEO_VAE_FP16_RELATIVE: "vae",
    "vae/minimax_h3_audio_vae_fp32.safetensors": "vae",
}
VERIFIED_WEIGHTS = {
    TEXT_ENCODER_RELATIVE: {
        "size": 27_141_342_152,
        "sha256": "bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923",
        "url": f"{SOURCE_BASE}/{TEXT_ENCODER_RELATIVE}",
    },
    VIDEO_VAE_FP16_RELATIVE: {
        "size": 5_207_808_496,
        "sha256": "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
        "url": f"{SOURCE_BASE}/vae/minimax_h3_video_vae_fp16.safetensors",
    },
    VIDEO_VAE_FP32_RELATIVE: {
        # Official FP32 tensors plus the ComfyUI latent normalization buffers.
        "size": 10_415_548_688,
        "sha256": "a28fa965eb65a3fe1279a8bf73f01dddaa36ecd039d08751f74bc8849e88767b",
        "url": f"{REPACKAGED_VAE_SOURCE_BASE}/minimax_h3_video_vae_fp32.safetensors",
    },
    "vae/minimax_h3_audio_vae_fp32.safetensors": {
        "size": 605_254_808,
        "sha256": "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
        "url": f"{SOURCE_BASE}/vae/minimax_h3_audio_vae_fp32.safetensors",
    },
    REF2VA_RELATIVE: {
        "size": 20_970_379_616,
        "sha256": "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779",
        "url": f"{SOURCE_BASE}/{REF2VA_RELATIVE}",
    },
}


def video_vae_precision() -> str:
    precision = os.getenv("H3_VIDEO_VAE_PRECISION", "fp32").strip().lower()
    if precision not in {"fp16", "fp32"}:
        raise ValueError("H3_VIDEO_VAE_PRECISION must be fp16 or fp32")
    return precision


def video_vae_relative() -> str:
    return VIDEO_VAE_FP32_RELATIVE if video_vae_precision() == "fp32" else VIDEO_VAE_FP16_RELATIVE


def video_vae_filename() -> str:
    return video_vae_relative().rsplit("/", 1)[-1]


def _selected_files() -> dict[str, str]:
    selected = dict(FILES)
    selected.pop(VIDEO_VAE_FP16_RELATIVE)
    selected[video_vae_relative()] = "vae"
    return selected


def weights_root() -> Path:
    configured = os.getenv("WEIGHTS_DIR")
    if configured:
        return Path(configured)
    if Path("/runpod-volume").is_dir():
        return Path("/runpod-volume/models")
    return Path("/weights")


def license_accepted() -> bool:
    return os.getenv("MINIMAX_H3_LICENSE_ACCEPTED", "").lower() in {"1", "true", "yes"}


def baked_weights_verified() -> bool:
    """Trust weights stored in immutable, content-addressed image layers."""
    return os.getenv("H3_BAKED_WEIGHTS_VERIFIED", "").lower() in {"1", "true", "yes"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_size: int, expected_sha: str, retries: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(retries):
        try:
            have = partial.stat().st_size if partial.exists() else 0
            if have > expected_size:
                partial.unlink()
                have = 0
            headers = {"User-Agent": "appnz-h3-cog/0.1"}
            if have:
                headers["Range"] = f"bytes={have}-"
            request = urllib.request.Request(url, headers=headers)
            print(
                f"Downloading {destination.name}: {have / 1e9:.2f}/{expected_size / 1e9:.2f} GB",
                flush=True,
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                append = have > 0 and response.status == 206
                with partial.open("ab" if append else "wb") as handle:
                    reported = have
                    for chunk in iter(lambda: response.read(8 << 20), b""):
                        handle.write(chunk)
                        current = handle.tell()
                        if current - reported >= 1_000_000_000:
                            reported = current
                            print(
                                f"Downloading {destination.name}: "
                                f"{current / 1e9:.2f}/{expected_size / 1e9:.2f} GB",
                                flush=True,
                            )
            if partial.stat().st_size != expected_size:
                raise IOError(f"size mismatch for {destination.name}")
            if expected_sha and _sha256(partial) != expected_sha:
                raise IOError(f"sha256 mismatch for {destination.name}")
            partial.replace(destination)
            return
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(1.5 * (attempt + 1))


def _install_link(source: Path, folder: str) -> None:
    target = COMFY_ROOT / "models" / folder / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)


def ensure_weights(workers: int = 4) -> dict[str, Path]:
    if not license_accepted():
        raise RuntimeError(
            "Set MINIMAX_H3_LICENSE_ACCEPTED=1 only after reviewing and accepting "
            "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"
        )
    root = weights_root() / "MiniMax-H3"
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".download.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = VERIFIED_WEIGHTS
        files = _selected_files()

        def fetch(relative: str) -> tuple[str, Path]:
            entry = manifest.get(relative)
            if not entry:
                raise RuntimeError(f"verified weight manifest is missing {relative}")
            destination = root / relative
            expected_size = int(entry["size"])
            expected_sha = str(entry.get("sha256", ""))
            valid = destination.exists() and destination.stat().st_size == expected_size
            if valid and expected_sha and not baked_weights_verified():
                valid = _sha256(destination) == expected_sha
            if not valid:
                _download(
                    entry["url"],
                    destination,
                    expected_size,
                    expected_sha,
                )
            _install_link(destination, files[relative])
            return relative, destination

        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(files)))) as pool:
            installed = dict(pool.map(fetch, files))
    return installed


def ensure_reference_weight() -> Path:
    """Install the separate REF2VA diffusion model on first reference request."""
    if not license_accepted():
        raise RuntimeError(
            "Set MINIMAX_H3_LICENSE_ACCEPTED=1 only after reviewing and accepting "
            "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"
        )
    root = weights_root() / "MiniMax-H3"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / REF2VA_RELATIVE
    lock_path = root / ".download.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        entry = VERIFIED_WEIGHTS[REF2VA_RELATIVE]
        valid = destination.exists() and destination.stat().st_size == int(entry["size"])
        if valid and not baked_weights_verified():
            valid = _sha256(destination) == entry["sha256"]
        if not valid:
            _download(entry["url"], destination, int(entry["size"]), entry["sha256"])
        _install_link(destination, "diffusion_models")
    return destination
