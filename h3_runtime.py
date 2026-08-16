"""Persistent ComfyUI process and one-request MiniMax H3 execution runtime."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

from h3_media import encode_video
from h3_prompt import format_h3_prompt
from h3_tuning import CacheTuning
from h3_workflow import aligned_frames, build_raylight_workflow, build_workflow, dimensions, validate_inputs
from weights import COMFY_ROOT, ensure_reference_weight, ensure_weights, video_vae_precision

COMFY_URL = "http://127.0.0.1:8188"


@dataclass(frozen=True)
class GenerationResult:
    path: Path
    metrics: dict


def _json_request(path: str, payload: dict | None = None, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        COMFY_URL + path,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _comfy_error(status: dict) -> str:
    """Keep the actionable Comfy exception, never multi-megabyte tensor inputs."""
    for message in reversed(status.get("messages") or []):
        if not isinstance(message, list) or len(message) != 2 or message[0] != "execution_error":
            continue
        details = message[1] if isinstance(message[1], dict) else {}
        kind = details.get("exception_type", "Error")
        text = details.get("exception_message", "Unknown ComfyUI execution error")
        node = details.get("node_type") or details.get("node_id") or "unknown node"
        return f"{kind} in {node}: {text}"
    return "ComfyUI generation failed without an execution_error message"


def _comfy_command() -> list[str]:
    """Keep large-memory GPUs resident and provide explicit fallback modes."""
    command = [
        sys.executable,
        str(COMFY_ROOT / "main.py"),
        "--listen", "127.0.0.1",
        "--port", "8188",
        "--disable-auto-launch",
        "--disable-metadata",
        "--reserve-vram", os.getenv("H3_RESERVE_VRAM_GB", "1.0"),
    ]
    if _use_highvram():
        command.append("--highvram")
    if _use_lowvram():
        command.append("--lowvram")
    if video_vae_precision() == "fp32":
        command.append("--fp32-vae")
    return command


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes"}


def _fp32_matmul_precision() -> str:
    return "tf32" if _env_enabled("H3_FP32_MATMUL_TF32") else "strict-fp32"


def _comfy_environment() -> dict[str, str]:
    env = os.environ.copy()
    # This must be present before ComfyUI imports torch.
    env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1" if _env_enabled("H3_FP32_MATMUL_TF32") else "0"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def _gpu_name() -> str:
    """Return the CUDA device name without making hardware detection fatal."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "unknown"


def _gpu_count() -> int:
    """Return the number of visible CUDA devices without making startup fatal."""
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _gpu_memory_gib() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 2**30
    except Exception:
        pass
    return None


def _use_lowvram() -> bool:
    """Make a 24GB single-card worker safe while retaining an override."""
    configured = os.getenv("H3_LOWVRAM")
    if configured is not None:
        return configured.lower() in {"1", "true", "yes"}
    memory_gib = _gpu_memory_gib()
    return _gpu_count() == 1 and memory_gib is not None and memory_gib < 28


def _use_highvram() -> bool:
    """Keep the complete weight set resident on one 80GB-class GPU."""
    configured = os.getenv("H3_HIGHVRAM")
    if configured is not None:
        return configured.lower() in {"1", "true", "yes"}
    if os.getenv("H3_LOWVRAM") is not None:
        return False
    memory_gib = _gpu_memory_gib()
    return _gpu_count() == 1 and memory_gib is not None and memory_gib >= 80


def select_parallel_mode(requested: str | None = None, gpu_count: int | None = None) -> str:
    """Select native single-GPU execution or Raylight's two-GPU path."""
    requested = (requested or os.getenv("H3_PARALLEL_MODE", "single")).strip().lower()
    aliases = {"native": "single", "dual": "raylight", "fsdp": "raylight"}
    requested = aliases.get(requested, requested)
    if requested not in {"auto", "single", "raylight"}:
        raise ValueError("H3_PARALLEL_MODE must be auto, single, or raylight")
    gpu_count = _gpu_count() if gpu_count is None else int(gpu_count)
    if requested == "auto":
        return "raylight" if gpu_count >= 2 else "single"
    if requested == "raylight" and gpu_count < 2:
        raise RuntimeError(f"Raylight mode requires 2 visible GPUs; found {gpu_count}")
    return requested


def _raylight_nodes_available() -> bool:
    required = ("RayInitializer", "RayUNETLoader", "RayBasicGuider", "RayBasicScheduler", "XFuserSamplerCustomAdvanced")
    try:
        return all(node in _json_request("/object_info/" + urllib.parse.quote(node), timeout=10) for node in required)
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False


def _sage_attention_supported() -> bool:
    """Only enable SageAttention on architectures built into the operator image."""
    if os.getenv("H3_SAGE_ATTENTION", "1").strip().lower() in {"0", "false", "no"}:
        return False
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_capability(0) in {(8, 0), (8, 9), (12, 0)}
    except Exception:
        pass
    return False


def _vram_mode(command: list[str]) -> str:
    if "--lowvram" in command:
        return "low"
    if "--highvram" in command:
        return "high-resident"
    return "normal-dynamic"


class H3Runtime:
    def __init__(self) -> None:
        ensure_weights()
        ensure_reference_weight()
        requested_mode = os.getenv("H3_PARALLEL_MODE", "single").strip().lower()
        self.parallel_mode = select_parallel_mode(requested_mode)
        self.process = self._start_comfy()
        if self.parallel_mode == "raylight" and not _raylight_nodes_available():
            if requested_mode == "auto":
                print("Raylight nodes unavailable; falling back to native single-GPU workflow", flush=True)
                self.parallel_mode = "single"
            else:
                self.process.terminate()
                raise RuntimeError("H3_PARALLEL_MODE=raylight but the required Raylight ComfyUI nodes did not load")
        print(f"H3 execution backend: mode={self.parallel_mode} visible_gpus={_gpu_count()}", flush=True)

    def _start_comfy(self) -> subprocess.Popen:
        command = _comfy_command()
        attention = "pytorch-sdpa"
        if _sage_attention_supported():
            try:
                __import__("sageattention")
                command.append("--use-sage-attention")
                attention = "sageattention"
            except ImportError:
                pass
        process = subprocess.Popen(command, cwd=COMFY_ROOT, env=_comfy_environment())
        deadline = time.time() + 180
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"ComfyUI exited during startup with code {process.returncode}")
            try:
                _json_request("/system_stats", timeout=3)
                print(
                    f"MiniMax H3 ready: attention={attention} gpu={_gpu_name()} "
                    f"vram_mode={_vram_mode(command)} "
                    f"video_vae={video_vae_precision()} "
                    f"fp32_matmul={_fp32_matmul_precision()} "
                    f"reserve_vram_gb={command[command.index('--reserve-vram') + 1]} comfy_pid={process.pid}",
                    flush=True,
                )
                return process
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                time.sleep(1)
        process.terminate()
        raise RuntimeError("ComfyUI did not become ready within 180 seconds")

    @staticmethod
    def _stage_image(path: Path | None, label: str) -> str | None:
        if path is None:
            return None
        source = Path(path)
        suffix = source.suffix.lower() if source.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} else ".png"
        name = f"h3-{label}-{uuid.uuid4().hex}{suffix}"
        destination = COMFY_ROOT / "input" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return name

    @staticmethod
    def _stage_media(path: Path, label: str, index: int) -> str:
        source = Path(path)
        suffix = source.suffix.lower()
        name = f"h3-{label}-{index}-{uuid.uuid4().hex}{suffix}"
        destination = COMFY_ROOT / "input" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return name

    @staticmethod
    def _history_output(entry: dict) -> Path:
        outputs = entry.get("outputs") or {}
        for node in outputs.values():
            for key in ("videos", "video", "gifs", "images"):
                for item in node.get(key, []) if isinstance(node, dict) else []:
                    if isinstance(item, dict) and item.get("filename"):
                        folder = item.get("type", "output")
                        base = COMFY_ROOT / ("output" if folder == "output" else folder)
                        path = base / item.get("subfolder", "") / item["filename"]
                        if path.exists():
                            return path
        raise RuntimeError("ComfyUI completed without a saved video")

    def generate(
        self,
        *,
        prompt: str,
        reference_images: list[Path] | None = None,
        reference_videos: list[Path] | None = None,
        reference_audios: list[Path] | None = None,
        aspect_ratio: str = "9:16",
        size: str = "preview",
        duration: float = 5.0,
        steps: int = 24,
        seed: int | None = None,
        structured_prompt: bool = False,
        include_audio: bool = True,
        output_codec: str = "mp4-h264",
        encode_quality: int = 26,
        cache: CacheTuning | None = None,
        return_metrics: bool = False,
    ) -> Path | GenerationResult:
        total_started = time.monotonic()
        command = _comfy_command()
        print(
            "H3 execution config: "
            f"gpu={_gpu_name()} "
            f"vram_mode={_vram_mode(command)} "
            f"dynamic_vram={'on' if '--highvram' not in command else 'off'} "
            f"video_vae={video_vae_precision()} "
            f"fp32_matmul={_fp32_matmul_precision()} "
            f"reserve_vram_gb={command[command.index('--reserve-vram') + 1]}",
            flush=True,
        )
        reference_images = reference_images or []
        reference_videos = reference_videos or []
        reference_audios = reference_audios or []
        reference_mode = bool(reference_images or reference_videos or reference_audios)
        if not reference_mode:
            raise ValueError("REF2VA requires at least one reference image, video, or audio clip")
        if len(reference_images) > 9 or len(reference_videos) > 3 or len(reference_audios) > 3:
            raise ValueError("reference mode supports at most 9 images, 3 videos, and 3 audio clips")
        validate_inputs(steps=steps, seed=seed)
        frames = aligned_frames(duration)
        width, height = dimensions(aspect_ratio, size)
        actual_seconds = frames / 24
        if seed is None:
            seed = random.SystemRandom().randint(0, 2**63 - 1)
        formatted = format_h3_prompt(
            prompt,
            actual_seconds,
            first_frame=False,
            last_frame=False,
            structured=structured_prompt,
        )
        reference_image_names = [self._stage_media(path, "ref-image", i) for i, path in enumerate(reference_images, 1)]
        reference_video_names = [self._stage_media(path, "ref-video", i) for i, path in enumerate(reference_videos, 1)]
        reference_audio_names = [self._stage_media(path, "ref-audio", i) for i, path in enumerate(reference_audios, 1)]
        workflow_builder = build_raylight_workflow if getattr(self, "parallel_mode", "single") == "raylight" else build_workflow
        graph = workflow_builder(
            prompt=formatted,
            width=width,
            height=height,
            frames=frames,
            steps=steps,
            seed=seed,
            reference_image_names=reference_image_names,
            reference_video_names=reference_video_names,
            reference_audio_names=reference_audio_names,
            cache=cache,
        )
        try:
            sample_started = time.monotonic()
            queued = _json_request("/prompt", {"prompt": graph})
            prompt_id = queued.get("prompt_id")
            if not prompt_id:
                raise RuntimeError(f"ComfyUI rejected workflow: {queued}")
            deadline = time.time() + 45 * 60
            while time.time() < deadline:
                time.sleep(1)
                history = _json_request("/history/" + urllib.parse.quote(prompt_id), timeout=30)
                entry = history.get(prompt_id)
                if not entry:
                    continue
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise RuntimeError("ComfyUI generation failed: " + _comfy_error(status))
                if status.get("completed"):
                    raw = self._history_output(entry)
                    sample_seconds = time.monotonic() - sample_started
                    encode_started = time.monotonic()
                    try:
                        output = encode_video(raw, output_codec, encode_quality, include_audio)
                    finally:
                        raw.unlink(missing_ok=True)
                    encode_seconds = time.monotonic() - encode_started
                    total_seconds = time.monotonic() - total_started
                    print(
                        f"generated {width}x{height} frames={frames} seconds={actual_seconds:.2f} "
                        f"steps={steps} seed={seed} mode=ref2va backend={getattr(self, 'parallel_mode', 'single')} "
                        f"total_seconds={total_seconds:.3f} "
                        f"cache={cache.profile if cache is not None else 'off'}",
                        flush=True,
                    )
                    if not return_metrics:
                        return output
                    metrics = {
                        "schema_version": 1,
                        "total_seconds": round(total_seconds, 3),
                        "generation_seconds": round(sample_seconds, 3),
                        "encode_seconds": round(encode_seconds, 3),
                        "output_bytes": output.stat().st_size,
                        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                        "width": width,
                        "height": height,
                        "frames": frames,
                        "duration_seconds": round(actual_seconds, 3),
                        "steps": steps,
                        "seed": seed,
                        "cache": cache.public_dict() if cache is not None else {"profile": "off"},
                    }
                    return GenerationResult(output, metrics)
            raise RuntimeError("generation exceeded 45 minute safety timeout")
        finally:
            for name in (*reference_image_names, *reference_video_names, *reference_audio_names):
                if name:
                    (COMFY_ROOT / "input" / name).unlink(missing_ok=True)
