"""Persistent ComfyUI process and one-request MiniMax H3 execution runtime."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import threading
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
from h3_workflow import (
    ATTENTION_SAGE,
    INFERENCE_PDD,
    INFERENCE_QUALITY,
    MODEL_QUANTIZATION_INT8,
    MODEL_QUANTIZATION_NVFP4,
    TASK_FL2VA,
    TASK_REF2VA,
    aligned_frames,
    build_workflow,
    dimensions,
    normalize_attention_backend,
    normalize_inference_mode,
    normalize_model_quantization,
    normalize_task,
    resolve_steps,
    task_partition,
    validate_inputs,
)
from weights import (
    COMFY_ROOT,
    ensure_diffusion_weights,
    ensure_model_profile,
    ensure_pdd_weight,
    ensure_weights,
    video_vae_precision,
)

COMFY_URL = "http://127.0.0.1:8188"
_ROUTE_LOCK = threading.Lock()
PDD_REQUIRED_NODES = ("MiniMaxH3PDDAccApply", "MiniMaxH3SigmaShift")
# ComfyUI loads custom nodes once at startup, so one successful probe is valid
# for the whole process lifetime.
_pdd_nodes_verified = False


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


def verify_pdd_nodes() -> None:
    """Fail before queueing a PDD workflow if its custom nodes are unavailable."""
    global _pdd_nodes_verified
    if _pdd_nodes_verified:
        return
    missing = []
    for name in PDD_REQUIRED_NODES:
        # Per-node endpoint keeps each response tiny instead of pulling the
        # multi-megabyte full-node object_info dump.
        try:
            object_info = _json_request(f"/object_info/{name}", timeout=30)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"could not query ComfyUI object_info to verify PDD node {name}: {error}"
            ) from error
        if name not in object_info:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "PDD inference requires ComfyUI v0.33.0+ and the "
            "ComfyUI-MiniMax-H3-PDD-Acc custom node; missing from this runtime: "
            + ", ".join(missing)
        )
    _pdd_nodes_verified = True


def _comfy_command() -> list[str]:
    """Keep large-memory GPUs resident and otherwise trust DynamicVRAM."""
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


def _use_highvram() -> bool:
    """Give an 80GiB-class card the option to keep both INT8 DiTs warm."""
    configured = os.getenv("H3_HIGHVRAM")
    if configured is not None:
        return configured.lower() in {"1", "true", "yes"}
    memory_gib = _gpu_memory_gib()
    return _gpu_count() == 1 and memory_gib is not None and memory_gib >= 80


def select_parallel_mode(requested: str | None = None) -> str:
    """Validate the native-only execution mode while failing loudly on old aliases."""
    value = (requested if requested is not None else os.getenv("H3_PARALLEL_MODE", "single")).strip().lower()
    if value in {"raylight", "auto", "fsdp", "dual"}:
        raise ValueError("Raylight has been removed; use H3_PARALLEL_MODE=single or unset it")
    if value in {"", "native", "single"}:
        return "single"
    raise ValueError("H3_PARALLEL_MODE must be single for the native ComfyUI runtime")


def _dit_switch_policy() -> str:
    policy = os.getenv("H3_DIT_SWITCH_POLICY", "auto").strip().lower()
    if policy not in {"auto", "evict"}:
        raise ValueError("H3_DIT_SWITCH_POLICY must be auto or evict")
    return policy


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
    return "high-resident" if "--highvram" in command else "normal-dynamic"


class H3Runtime:
    def __init__(self) -> None:
        parallel_mode = select_parallel_mode()
        dit_switch_policy = _dit_switch_policy()
        ensure_weights()
        ensure_diffusion_weights()
        if os.getenv("H3_LOWVRAM") is not None:
            print("H3_LOWVRAM is retired; ComfyUI DynamicVRAM controls memory", flush=True)
        self.parallel_mode = parallel_mode
        self.dit_switch_policy = dit_switch_policy
        self.current_task: str | None = None
        self.current_partition: str | None = None
        self.current_model_quantization: str | None = None
        self.process = self._start_comfy()
        print(
            f"H3 execution backend: mode=native/single visible_gpus={_gpu_count()} "
            f"dit_switch_policy={self.dit_switch_policy}",
            flush=True,
        )

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

    def _prepare_generation(
        self,
        task: str,
        partition: str,
        model_quantization: str = MODEL_QUANTIZATION_INT8,
    ) -> None:
        previous_task = self.current_task
        previous_partition = self.current_partition
        previous_quantization = getattr(self, "current_model_quantization", None)
        route_changed = (
            task != previous_task
            or partition != previous_partition
            or model_quantization != previous_quantization
        )
        if route_changed:
            print(
                f"H3 task route: {previous_task or 'cold'}->{task} "
                f"partition={previous_partition or 'cold'}->{partition} "
                f"quantization={previous_quantization or 'cold'}->{model_quantization}",
                flush=True,
            )
        if (
            previous_partition is not None
            and (
                partition != previous_partition
                or model_quantization != previous_quantization
            )
            and self.dit_switch_policy == "evict"
        ):
            _json_request("/free", {"unload_models": True, "free_memory": True})
            print(
                "ComfyUI model cache freed for route switch to "
                f"partition={partition} quantization={model_quantization}",
                flush=True,
            )
        self.current_task = task
        self.current_partition = partition
        self.current_model_quantization = model_quantization

    def generate(
        self,
        *,
        prompt: str,
        task: str,
        reference_images: list[Path] | None = None,
        reference_videos: list[Path] | None = None,
        reference_audios: list[Path] | None = None,
        first_frame: Path | None = None,
        last_frame: Path | None = None,
        loop: bool = False,
        aspect_ratio: str = "9:16",
        size: str = "preview",
        duration: float = 5.0,
        steps: int = 24,
        seed: int | None = None,
        structured_prompt: bool | None = None,
        include_audio: bool = True,
        output_codec: str = "mp4-h264",
        encode_quality: int = 26,
        cache: CacheTuning | None = None,
        fused_modulation: bool = True,
        attention_backend: str = ATTENTION_SAGE,
        inference_mode: str = INFERENCE_QUALITY,
        model_quantization: str = MODEL_QUANTIZATION_INT8,
        return_metrics: bool = False,
    ) -> Path | GenerationResult:
        total_started = time.monotonic()
        task = normalize_task(task)
        attention_backend = normalize_attention_backend(attention_backend)
        inference_mode = normalize_inference_mode(inference_mode)
        model_quantization = normalize_model_quantization(model_quantization)
        effective_steps = resolve_steps(task, steps, inference_mode)
        partition = task_partition(task)
        command = _comfy_command()
        print(
            "H3 execution config: "
            f"gpu={_gpu_name()} "
            f"vram_mode={_vram_mode(command)} "
            f"dynamic_vram={'on' if '--highvram' not in command else 'off'} "
            f"attention_backend={attention_backend} "
            f"inference_mode={inference_mode} "
            f"model_quantization={model_quantization} "
            f"video_vae={video_vae_precision()} "
            f"fp32_matmul={_fp32_matmul_precision()} "
            f"reserve_vram_gb={command[command.index('--reserve-vram') + 1]}",
            flush=True,
        )
        reference_images = reference_images or []
        reference_videos = reference_videos or []
        reference_audios = reference_audios or []
        reference_count = len(reference_images) + len(reference_videos) + len(reference_audios)
        validate_inputs(
            task=task,
            steps=steps,
            seed=seed,
            first_frame=first_frame,
            last_frame=last_frame,
            loop=loop,
            reference_count=reference_count,
            fused_modulation=fused_modulation,
            attention_backend=attention_backend,
            inference_mode=inference_mode,
            model_quantization=model_quantization,
        )
        if len(reference_images) > 9 or len(reference_videos) > 3 or len(reference_audios) > 3:
            raise ValueError("ref2va supports at most 9 images, 3 videos, and 3 audio clips")
        if inference_mode == INFERENCE_PDD:
            if cache is not None:
                raise ValueError(
                    "inference_mode=pdd is incompatible with EasyCache; drop the cache "
                    "tuning or select quality/turbo inference"
                )
            verify_pdd_nodes()
        frames = aligned_frames(duration)
        width, height = dimensions(aspect_ratio, size)
        actual_seconds = frames / 24
        if seed is None:
            seed = random.SystemRandom().randint(0, 2**63 - 1)
        prompt_has_first = task == TASK_FL2VA and first_frame is not None
        prompt_has_last = task == TASK_FL2VA and (last_frame is not None or loop)
        structured = structured_prompt if structured_prompt is not None else task != TASK_REF2VA
        formatted = format_h3_prompt(
            prompt,
            actual_seconds,
            first_frame=prompt_has_first,
            last_frame=prompt_has_last,
            structured=structured,
        )
        first_image_name: str | None = None
        last_image_name: str | None = None
        reference_image_names: list[str] = []
        reference_video_names: list[str] = []
        reference_audio_names: list[str] = []
        try:
            first_image_name = self._stage_image(first_frame, "first-frame")
            last_image_name = self._stage_image(last_frame, "last-frame")
            reference_image_names = [
                self._stage_media(path, "ref-image", index)
                for index, path in enumerate(reference_images)
            ]
            reference_video_names = [
                self._stage_media(path, "ref-video", index)
                for index, path in enumerate(reference_videos)
            ]
            reference_audio_names = [
                self._stage_media(path, "ref-audio", index)
                for index, path in enumerate(reference_audios)
            ]
            graph = build_workflow(
                prompt=formatted,
                task=task,
                width=width,
                height=height,
                frames=frames,
                steps=steps,
                seed=seed,
                first_image_name=first_image_name,
                last_image_name=last_image_name,
                loop=loop,
                reference_image_names=reference_image_names,
                reference_video_names=reference_video_names,
                reference_audio_names=reference_audio_names,
                cache=cache,
                fused_modulation=fused_modulation,
                attention_backend=attention_backend,
                inference_mode=inference_mode,
                model_quantization=model_quantization,
            )
            with _ROUTE_LOCK:
                self._prepare_generation(task, partition, model_quantization)
                if model_quantization == MODEL_QUANTIZATION_NVFP4:
                    # Symlink swaps for the shared text encoder / DiT are not
                    # atomic; serialize them against concurrent generations.
                    ensure_model_profile(partition, model_quantization)
                if inference_mode == INFERENCE_PDD:
                    # Symlink creation is not atomic; serialize it against
                    # concurrent generations under the same route lock.
                    ensure_pdd_weight(partition)
                    print(
                        f"H3 PDD route: partition={partition} nfe=8 "
                        f"pdd_file={'Ref2VA' if partition == 'ref2va' else 'FL2VA'} Acc-8Step",
                        flush=True,
                    )
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
                        f"steps={effective_steps} seed={seed} mode={task} partition={partition} backend=native/single "
                        f"dit_switch={self.dit_switch_policy} total_seconds={total_seconds:.3f} "
                        f"cache={cache.profile if cache is not None else 'off'} "
                        f"attention_backend={attention_backend} "
                        f"inference_mode={inference_mode} "
                        f"model_quantization={model_quantization} "
                        f"fused_modulation={'on' if fused_modulation else 'off'}",
                        flush=True,
                    )
                    if not return_metrics:
                        return output
                    metrics = {
                        # v3: model_quantization records INT8 vs NVFP4 routing.
                        "schema_version": 3,
                        "task": task,
                        "partition": partition,
                        "dit_switch_policy": self.dit_switch_policy,
                        "total_seconds": round(total_seconds, 3),
                        "generation_seconds": round(sample_seconds, 3),
                        "encode_seconds": round(encode_seconds, 3),
                        "output_bytes": output.stat().st_size,
                        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                        "width": width,
                        "height": height,
                        "frames": frames,
                        "duration_seconds": round(actual_seconds, 3),
                        "steps": effective_steps,
                        "requested_steps": int(steps),
                        "seed": seed,
                        "cache": cache.public_dict() if cache is not None else {"profile": "off"},
                        "attention_backend": attention_backend,
                        "inference_mode": inference_mode,
                        "model_quantization": model_quantization,
                        "fused_modulation": fused_modulation,
                    }
                    return GenerationResult(output, metrics)
            raise RuntimeError("generation exceeded 45 minute safety timeout")
        finally:
            for name in (
                first_image_name,
                last_image_name,
                *reference_image_names,
                *reference_video_names,
                *reference_audio_names,
            ):
                if name:
                    (COMFY_ROOT / "input" / name).unlink(missing_ok=True)
