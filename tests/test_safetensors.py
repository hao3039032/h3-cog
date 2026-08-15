import json
import struct

import h3_safetensors


def _write_source(path, tensor_count=500):
    header = {"__metadata__": {"format": "pt"}}
    payload = bytearray()
    for index in range(tensor_count):
        start = len(payload)
        payload.extend(struct.pack("<f", float(index)))
        header[f"tensor.{index}"] = {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [start, len(payload)],
        }
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    padding = b" " * (-len(raw_header) % 8)
    path.write_bytes(struct.pack("<Q", len(raw_header) + len(padding)) + raw_header + padding + payload)
    return bytes(payload)


def test_packaging_appends_comfy_buffers_without_rewriting_payload(tmp_path):
    source = tmp_path / "source.safetensors"
    destination = tmp_path / "minimax_h3_video_vae_fp32.safetensors"
    payload = _write_source(source)
    h3_safetensors.add_minimax_h3_video_vae_buffers(source, destination)

    raw = destination.read_bytes()
    header_length = struct.unpack("<Q", raw[:8])[0]
    assert header_length % 8 == 0
    header = json.loads(raw[8:8 + header_length])
    assert len(header) == 503
    assert header["latents_mean"]["dtype"] == "F32"
    assert header["latents_mean"]["shape"] == [24]
    assert header["latents_mean"]["data_offsets"] == [2000, 2096]
    assert header["latents_std"]["data_offsets"] == [2096, 2192]
    assert raw[8 + header_length:8 + header_length + 2000] == payload
    expected_buffers = h3_safetensors.LATENTS_MEAN + h3_safetensors.LATENTS_STD
    expected_f32 = struct.unpack("<48f", struct.pack("<48f", *expected_buffers))
    assert struct.unpack("<48f", raw[-192:]) == expected_f32
