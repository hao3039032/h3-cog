"""Validation and API-format ComfyUI workflow construction for MiniMax H3."""

from __future__ import annotations

import math
import uuid
from pathlib import Path

from h3_tuning import CacheTuning
from weights import video_vae_filename

FPS = 24
TASK_T2VA = "t2va"
TASK_FL2VA = "fl2va"
TASK_REF2VA = "ref2va"
TASKS = (TASK_T2VA, TASK_FL2VA, TASK_REF2VA)
TASK_PARTITIONS = {
    TASK_T2VA: TASK_FL2VA,
    TASK_FL2VA: TASK_FL2VA,
    TASK_REF2VA: TASK_REF2VA,
}
FL2VA_MODEL = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
REF2VA_MODEL = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

ASPECTS = {
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "1:1": (768, 768),
    "4:3": (1024, 768),
    "3:4": (768, 1024),
    "21:9": (1344, 576),
}
SCALES = {"preview": 0.40, "balanced": 0.58, "native": 1.0}


def aligned_frames(seconds: float) -> int:
    if not 4 <= float(seconds) <= 15:
        raise ValueError("duration must be between 4 and 15 seconds")
    frames = max(5, round(float(seconds) * FPS))
    while frames % 17 != 5:
        frames += 1
    return frames


def dimensions(aspect_ratio: str, size: str) -> tuple[int, int]:
    if aspect_ratio not in ASPECTS:
        raise ValueError(f"aspect_ratio must be one of {sorted(ASPECTS)}")
    if size not in SCALES:
        raise ValueError(f"size must be one of {sorted(SCALES)}")
    width, height = ASPECTS[aspect_ratio]
    scale = math.sqrt(SCALES[size])
    width = max(32, round(width * scale / 32) * 32)
    height = max(32, round(height * scale / 32) * 32)
    return width, height


def normalize_task(task: str) -> str:
    normalized = str(task or "").strip().lower()
    if normalized not in TASKS:
        raise ValueError(f"task must be one of {list(TASKS)}")
    return normalized


def task_partition(task: str) -> str:
    return TASK_PARTITIONS[normalize_task(task)]


def infer_task(
    *,
    first_frame: Path | None = None,
    last_frame: Path | None = None,
    loop: bool = False,
    reference_count: int = 0,
) -> str:
    has_keyframe = first_frame is not None or last_frame is not None or bool(loop)
    has_reference = int(reference_count) > 0
    if has_keyframe and has_reference:
        raise ValueError("FL2VA keyframes/loop cannot be combined with REF2VA references")
    if has_reference:
        return TASK_REF2VA
    if has_keyframe:
        return TASK_FL2VA
    return TASK_T2VA


def validate_inputs(
    *,
    task: str,
    steps: int,
    seed: int | None,
    first_frame: Path | None = None,
    last_frame: Path | None = None,
    loop: bool = False,
    reference_count: int = 0,
    fused_modulation: bool = True,
) -> None:
    normalized_task = normalize_task(task)
    inferred_task = infer_task(
        first_frame=first_frame,
        last_frame=last_frame,
        loop=loop,
        reference_count=reference_count,
    )
    if normalized_task == TASK_T2VA and inferred_task != TASK_T2VA:
        raise ValueError("t2va does not allow first_frame, last_frame, loop, or references")
    if normalized_task == TASK_FL2VA:
        if inferred_task == TASK_REF2VA:
            raise ValueError("fl2va does not allow reference media")
        if inferred_task == TASK_T2VA:
            raise ValueError("fl2va requires first_frame, last_frame, or loop")
        if loop and first_frame is None:
            raise ValueError("loop requires first_frame because it is reused as the last frame")
        if loop and last_frame is not None:
            raise ValueError("loop cannot be combined with last_frame")
    if normalized_task == TASK_REF2VA:
        if inferred_task == TASK_FL2VA:
            raise ValueError("ref2va does not allow first_frame, last_frame, or loop")
        if inferred_task == TASK_T2VA:
            raise ValueError("ref2va requires at least one reference image, video, or audio clip")

    if not 8 <= int(steps) <= 60:
        raise ValueError("steps must be between 8 and 60")
    if seed is not None and not 0 <= int(seed) <= 2**63 - 1:
        raise ValueError("seed must be between 0 and 2^63-1")
    if not isinstance(fused_modulation, bool):
        raise ValueError("fused_modulation must be a boolean")


def _scaled_image(graph: dict[str, dict], node_id: str, name: str, width: int, height: int) -> tuple[str, dict]:
    load_id = str(int(node_id) + 1)
    graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
    graph[load_id] = {
        "class_type": "ImageScale",
        "inputs": {
            "image": [node_id, 0],
            "upscale_method": "lanczos",
            "width": int(width),
            "height": int(height),
            "crop": "center",
        },
    }
    return load_id, graph[load_id]


def build_workflow(
    *,
    prompt: str,
    task: str,
    width: int,
    height: int,
    frames: int,
    steps: int,
    seed: int,
    first_image_name: str | None = None,
    last_image_name: str | None = None,
    loop: bool = False,
    reference_image_names: list[str] | None = None,
    reference_video_names: list[str] | None = None,
    reference_audio_names: list[str] | None = None,
    cache: CacheTuning | None = None,
    fused_modulation: bool = True,
) -> dict[str, dict]:
    task = normalize_task(task)
    reference_image_names = reference_image_names or []
    reference_video_names = reference_video_names or []
    reference_audio_names = reference_audio_names or []
    validate_inputs(
        task=task,
        steps=steps,
        seed=seed,
        first_frame=Path(first_image_name) if first_image_name else None,
        last_frame=Path(last_image_name) if last_image_name else None,
        loop=loop,
        reference_count=len(reference_image_names) + len(reference_video_names) + len(reference_audio_names),
        fused_modulation=fused_modulation,
    )

    if task == TASK_REF2VA:
        model_name = REF2VA_MODEL
        conditioning = {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "inputs": {
                "clip": ["2", 0],
                "vae": ["3", 0],
                "audio_vae": ["4", 0],
                "prompt": prompt,
                "width": int(width),
                "height": int(height),
                "length": int(frames),
                "ref_image_size": "match",
            },
        }
    else:
        model_name = FL2VA_MODEL
        conditioning = {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["2", 0],
                "vae": ["3", 0],
                "prompt": prompt,
                "width": int(width),
                "height": int(height),
                "length": int(frames),
            },
        }

    graph: dict[str, dict] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": model_name, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": video_vae_filename()}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "5": conditioning,
        "6": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["5", 0]}},
        "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "9": {"class_type": "BasicScheduler", "inputs": {"model": ["1", 0], "scheduler": "simple", "steps": int(steps), "denoise": 1.0}},
        "10": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {"noise": ["7", 0], "guider": ["6", 0], "sampler": ["8", 0], "sigmas": ["9", 0], "latent_image": ["5", 1]},
        },
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}},
        "13": {"class_type": "CreateVideo", "inputs": {"images": ["11", 0], "audio": ["12", 0], "fps": 24.0, "bit_depth": 8}},
        "14": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["13", 0],
                "filename_prefix": f"h3/raw-{uuid.uuid4().hex}",
                "format": "mp4",
                # Comfy's V3 DynamicCombo API is flattened in prompt JSON and
                # reconstructed before SaveVideo.execute(). A nested object is
                # silently discarded and only fails after the expensive sample.
                "codec": "h264",
                "codec.encoding": "re-encode",
                "codec.encoding.crf": 17.0,
            },
        },
    }
    next_id = 15
    model_link: list[str | int] = ["1", 0]

    if fused_modulation:
        graph[str(next_id)] = {
            "class_type": "MiniMaxH3FusedModulation",
            "inputs": {
                "model": model_link,
                "enabled": True,
            },
        }
        model_link = [str(next_id), 0]
        next_id += 1

    if cache is not None:
        graph[str(next_id)] = {
            "class_type": "EasyCache",
            "inputs": {
                "model": model_link,
                "reuse_threshold": cache.reuse_threshold,
                "start_percent": cache.start_percent,
                "end_percent": cache.end_percent,
                "verbose": cache.verbose,
            },
        }
        model_link = [str(next_id), 0]
        next_id += 1

    graph["6"]["inputs"]["model"] = model_link
    graph["9"]["inputs"]["model"] = model_link

    if task != TASK_REF2VA:
        first_output_id: str | None = None
        if first_image_name:
            node_id = str(next_id)
            first_output_id, _ = _scaled_image(graph, node_id, first_image_name, width, height)
            next_id += 2
        if last_image_name:
            node_id = str(next_id)
            last_output_id, _ = _scaled_image(graph, node_id, last_image_name, width, height)
            graph["5"]["inputs"]["last_frame"] = [last_output_id, 0]
            next_id += 2
        if first_output_id is not None:
            graph["5"]["inputs"]["first_frame"] = [first_output_id, 0]
            if loop:
                graph["5"]["inputs"]["last_frame"] = [first_output_id, 0]

    for index, name in enumerate(reference_image_names):
        node_id = str(next_id)
        graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
        graph["5"]["inputs"][f"ref_images.ref_image_{index}"] = [node_id, 0]
        next_id += 1
    for index, name in enumerate(reference_video_names):
        load_id = str(next_id)
        components_id = str(next_id + 1)
        graph[load_id] = {"class_type": "LoadVideo", "inputs": {"file": name}}
        graph[components_id] = {"class_type": "GetVideoComponents", "inputs": {"video": [load_id, 0]}}
        graph["5"]["inputs"][f"ref_videos.ref_video_{index}"] = [components_id, 0]
        graph["5"]["inputs"][f"ref_video_audios.ref_video_audio_{index}"] = [components_id, 1]
        next_id += 2
    for index, name in enumerate(reference_audio_names):
        node_id = str(next_id)
        graph[node_id] = {"class_type": "LoadAudio", "inputs": {"audio": name}}
        graph["5"]["inputs"][f"ref_audios.ref_audio_{index}"] = [node_id, 0]
        next_id += 1
    return graph
