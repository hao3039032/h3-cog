#!/usr/bin/env python3
"""Package the official MiniMax H3 FP32 video VAE for ComfyUI upload."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from h3_safetensors import add_minimax_h3_video_vae_buffers


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    output = add_minimax_h3_video_vae_buffers(args.source, args.destination)
    print(f"path={output}")
    print(f"size={output.stat().st_size}")
    print(f"sha256={sha256(output)}")


if __name__ == "__main__":
    main()
