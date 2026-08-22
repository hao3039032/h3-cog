"""GPU-first ffmpeg output encoding with deterministic CPU fallbacks."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def available_encoders() -> set[str]:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        fields[1]
        for line in proc.stdout.splitlines()
        if len(fields := line.split()) >= 2 and fields[0].startswith("V")
    }


def encode_profiles(codec: str, quality: int, encoders: set[str]) -> tuple[str, list[list[str]]]:
    quality = min(max(int(quality), 16), 45)
    if codec == "webm-av1":
        profiles: list[list[str]] = []
        if "av1_nvenc" in encoders:
            profiles.append(["-c:v", "av1_nvenc", "-preset", "p4", "-tune", "hq", "-rc", "vbr", "-cq", str(quality), "-b:v", "0", "-pix_fmt", "yuv420p"])
        if "libsvtav1" in encoders:
            profiles.append(["-c:v", "libsvtav1", "-preset", "8", "-crf", str(quality), "-pix_fmt", "yuv420p"])
        if not profiles:
            raise RuntimeError("ffmpeg has neither av1_nvenc nor libsvtav1")
        return "webm", profiles
    if codec != "mp4-h264":
        raise ValueError("output_codec must be mp4-h264 or webm-av1")
    profiles = []
    if "h264_nvenc" in encoders:
        profiles.append(["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq", "-rc", "vbr", "-cq", str(min(quality, 32)), "-b:v", "0", "-pix_fmt", "yuv420p"])
    profiles.append(["-c:v", "libx264", "-preset", "veryfast", "-crf", str(min(quality, 32)), "-pix_fmt", "yuv420p"])
    return "mp4", profiles


def audio_profile(extension: str) -> list[str]:
    """Copy ComfyUI's native AAC into MP4; WebM requires Opus."""
    if extension == "mp4":
        return ["-c:a", "copy"]
    return ["-c:a", "libopus", "-b:a", "160k"]


def encode_video(source: Path, codec: str, quality: int, include_audio: bool) -> Path:
    extension, profiles = encode_profiles(codec, quality, available_encoders())
    output = Path(tempfile.mkdtemp(prefix="h3-output-")) / f"video.{extension}"
    last_error = ""
    for profile in profiles:
        command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source), "-map", "0:v:0"]
        if include_audio:
            command += ["-map", "0:a?"]
        command += profile
        if include_audio:
            command += audio_profile(extension)
        else:
            command += ["-an"]
        if extension == "mp4":
            command += ["-movflags", "+faststart"]
        command.append(str(output))
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        if proc.returncode == 0 and output.exists() and output.stat().st_size:
            used = profile[1]
            print(f"ffmpeg encoded {extension} with {used}", flush=True)
            return output
        last_error = (proc.stderr or proc.stdout or "")[-1200:]
        output.unlink(missing_ok=True)
    shutil.rmtree(output.parent, ignore_errors=True)
    raise RuntimeError(f"ffmpeg encode failed: {last_error}")
