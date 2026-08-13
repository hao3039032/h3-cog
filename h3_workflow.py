"""Validation and API-format ComfyUI workflow construction for MiniMax H3."""

from __future__ import annotations

import math
import uuid

from h3_tuning import CacheTuning

FPS = 24
REFERENCE_MODEL = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
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


def validate_inputs(
    *,
    steps: int,
    seed: int | None,
) -> None:
    if not 8 <= int(steps) <= 30:
        raise ValueError("steps must be between 8 and 30")
    if seed is not None and not 0 <= int(seed) <= 2**63 - 1:
        raise ValueError("seed must be between 0 and 2^63-1")


def build_workflow(
    *,
    prompt: str,
    width: int,
    height: int,
    frames: int,
    steps: int,
    seed: int,
    reference_image_names: list[str] | None = None,
    reference_video_names: list[str] | None = None,
    reference_audio_names: list[str] | None = None,
    cache: CacheTuning | None = None,
) -> dict[str, dict]:
    reference_image_names = reference_image_names or []
    reference_video_names = reference_video_names or []
    reference_audio_names = reference_audio_names or []
    graph: dict[str, dict] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": REFERENCE_MODEL, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "5": {
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
        },
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
                "codec": {"codec": "h264", "encoding": {"encoding": "re-encode", "crf": 17.0}},
            },
        },
    }
    next_id = 15
    if cache is not None:
        graph[str(next_id)] = {
            "class_type": "EasyCache",
            "inputs": {
                "model": ["1", 0],
                "reuse_threshold": cache.reuse_threshold,
                "start_percent": cache.start_percent,
                "end_percent": cache.end_percent,
                "verbose": cache.verbose,
            },
        }
        graph["6"]["inputs"]["model"] = [str(next_id), 0]
        graph["9"]["inputs"]["model"] = [str(next_id), 0]
        next_id += 1
    for index, name in enumerate(reference_image_names, 1):
        node_id = str(next_id)
        graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
        graph["5"]["inputs"][f"ref_image_{index}"] = [node_id, 0]
        next_id += 1
    for index, name in enumerate(reference_video_names, 1):
        load_id = str(next_id)
        components_id = str(next_id + 1)
        graph[load_id] = {"class_type": "LoadVideo", "inputs": {"file": name}}
        graph[components_id] = {"class_type": "GetVideoComponents", "inputs": {"video": [load_id, 0]}}
        graph["5"]["inputs"][f"ref_video_{index}"] = [components_id, 0]
        graph["5"]["inputs"][f"ref_video_audio_{index}"] = [components_id, 1]
        next_id += 2
    for index, name in enumerate(reference_audio_names, 1):
        node_id = str(next_id)
        graph[node_id] = {"class_type": "LoadAudio", "inputs": {"audio": name}}
        graph["5"]["inputs"][f"ref_audio_{index}"] = [node_id, 0]
        next_id += 1
    return graph
