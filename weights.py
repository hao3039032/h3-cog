"""Model paths for externally provisioned MiniMax H3 weights."""

from __future__ import annotations

import os
from pathlib import Path

SOURCE_BASE = "https://modelscope.cn/models/Comfy-Org/MiniMax-H3/resolve/master"
REPACKAGED_VAE_SOURCE_BASE = "https://modelscope.cn/models/Austusm/minimax_h3_video_vae/resolve/master"
COMFY_ROOT = Path(os.getenv("COMFY_ROOT", "/opt/ComfyUI"))
TEXT_ENCODER_RELATIVE = "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
FL2VA_RELATIVE = "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
REF2VA_RELATIVE = "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
DIFFUSION_RELATIVES = (FL2VA_RELATIVE, REF2VA_RELATIVE)
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
    FL2VA_RELATIVE: {
        "size": 20_970_379_616,
        "sha256": "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a",
        "url": f"{SOURCE_BASE}/{FL2VA_RELATIVE}",
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


def _install_link(source: Path, folder: str) -> None:
    target = COMFY_ROOT / "models" / folder / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)


def ensure_weights() -> dict[str, Path]:
    if not license_accepted():
        raise RuntimeError(
            "Set MINIMAX_H3_LICENSE_ACCEPTED=1 only after reviewing and accepting "
            "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"
        )
    root = weights_root() / "MiniMax-H3"
    files = _selected_files()
    installed = {relative: root / relative for relative in files}
    missing = [path for path in installed.values() if not path.is_file()]
    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Required MiniMax H3 weights are missing; provision them before startup:\n"
            f"{details}"
        )
    for relative, destination in installed.items():
        _install_link(destination, files[relative])
    return installed


def _ensure_diffusion_weight(relative: str) -> Path:
    if not license_accepted():
        raise RuntimeError(
            "Set MINIMAX_H3_LICENSE_ACCEPTED=1 only after reviewing and accepting "
            "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"
        )
    root = weights_root() / "MiniMax-H3"
    destination = root / relative
    if not destination.is_file():
        raise FileNotFoundError(
            f"Required MiniMax H3 weight is missing; provision it before startup:\n- {destination}"
        )
    _install_link(destination, "diffusion_models")
    return destination


def ensure_reference_weight() -> Path:
    return _ensure_diffusion_weight(REF2VA_RELATIVE)


def ensure_fl2va_weight() -> Path:
    return _ensure_diffusion_weight(FL2VA_RELATIVE)


def ensure_diffusion_weights() -> dict[str, Path]:
    return {relative: _ensure_diffusion_weight(relative) for relative in DIFFUSION_RELATIVES}
