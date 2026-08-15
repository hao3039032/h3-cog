"""Small safetensors utilities that avoid pulling torch into weight setup."""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

LATENTS_MEAN = (
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683049442874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
)
LATENTS_STD = (
    1.2223774194717407, 1.2767263650894165, 1.68317747116088865, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.96531379222869875, 1.05698859691619875,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049925, 0.7197399735450745, 0.69362932443618775,
    2.961095094680786, 2.7694199085552395, 3.0496184825897215, 2.1084054180145265,
    3.276226282119758, 3.1627357006073, 2.28168129920959475, 2.6127843856811525,
)


def _read_header(source: Path) -> tuple[dict, int]:
    with source.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"invalid safetensors prefix in {source}")
        header_length = struct.unpack("<Q", prefix)[0]
        raw_header = handle.read(header_length)
    if len(raw_header) != header_length:
        raise ValueError(f"truncated safetensors header in {source}")
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid safetensors JSON header in {source}") from error
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors header object in {source}")
    return header, 8 + header_length


def _tensor_entries(header: dict) -> dict[str, dict]:
    return {
        name: entry
        for name, entry in header.items()
        if not name.startswith("__") and isinstance(entry, dict)
    }


def _float32_bytes(values: tuple[float, ...]) -> bytes:
    return struct.pack("<" + "f" * len(values), *values)


def add_minimax_h3_video_vae_buffers(source: Path, destination: Path) -> Path:
    """Append the two persistent ComfyUI buffers to an official source VAE.

    Tensor payloads are copied byte-for-byte. This matters for the FP32 build:
    loading and re-saving through torch would offer no benefit and would make
    the packaging tool require a matching GPU/torch runtime.
    """
    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    header, data_start = _read_header(source)
    tensors = _tensor_entries(header)
    if "latents_mean" in tensors or "latents_std" in tensors:
        raise ValueError("source video VAE already contains Comfy latent buffers")
    if len(tensors) < 500:
        raise ValueError(f"unexpected MiniMax H3 video VAE tensor count: {len(tensors)}")
    if any(entry.get("dtype") != "F32" for entry in tensors.values()):
        raise ValueError("FP32 video VAE source contains non-F32 tensors")

    ranges = sorted(
        (int(entry["data_offsets"][0]), int(entry["data_offsets"][1]))
        for entry in tensors.values()
        if "data_offsets" in entry
    )
    if not ranges or ranges[0][0] != 0:
        raise ValueError("FP32 video VAE source has no contiguous tensor data")
    for (_, previous_end), (next_start, _) in zip(ranges, ranges[1:]):
        if next_start != previous_end:
            raise ValueError("FP32 video VAE source has non-contiguous tensor data")
    source_data_size = ranges[-1][1]
    if source.stat().st_size != data_start + source_data_size:
        raise ValueError("FP32 video VAE source has trailing or truncated data")

    mean_bytes = _float32_bytes(LATENTS_MEAN)
    std_bytes = _float32_bytes(LATENTS_STD)
    header["latents_mean"] = {
        "dtype": "F32",
        "shape": [len(LATENTS_MEAN)],
        "data_offsets": [source_data_size, source_data_size + len(mean_bytes)],
    }
    header["latents_std"] = {
        "dtype": "F32",
        "shape": [len(LATENTS_STD)],
        "data_offsets": [
            source_data_size + len(mean_bytes),
            source_data_size + len(mean_bytes) + len(std_bytes),
        ],
    }
    encoded_header = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    padding = b" " * (-len(encoded_header) % 8)
    output_header_length = len(encoded_header) + len(padding)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            reader.seek(data_start)
            writer.write(struct.pack("<Q", output_header_length))
            writer.write(encoded_header)
            writer.write(padding)
            while chunk := reader.read(32 << 20):
                writer.write(chunk)
            writer.write(mean_bytes)
            writer.write(std_bytes)
            writer.flush()
            os.fsync(writer.fileno())
        expected_size = 8 + output_header_length + source_data_size + len(mean_bytes) + len(std_bytes)
        if temporary.stat().st_size != expected_size:
            raise ValueError("packaged FP32 video VAE has an unexpected size")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
