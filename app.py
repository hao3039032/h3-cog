"""Gradio entry point for a ModelScope Studio or a local H3 workstation."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import gradio as gr

from h3_runtime import H3Runtime

_runtime: H3Runtime | None = None
_runtime_lock = threading.Lock()
_generation_lock = threading.Lock()


def _get_runtime() -> H3Runtime:
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = H3Runtime()
    return _runtime


def _paths(files: Any) -> list[Path]:
    if not files:
        return []
    if not isinstance(files, list):
        files = [files]
    paths: list[Path] = []
    for item in files:
        value = item if isinstance(item, (str, Path)) else getattr(item, "name", None)
        if value:
            paths.append(Path(value))
    return paths


def generate_video(
    prompt: str,
    reference_images: Any,
    reference_videos: Any,
    reference_audios: Any,
    aspect_ratio: str,
    size: str,
    duration: float,
    steps: int,
    seed: float | None,
    include_audio: bool,
) -> tuple[str, str]:
    if not prompt or not prompt.strip():
        raise gr.Error("请输入提示词。")
    images = _paths(reference_images)
    videos = _paths(reference_videos)
    audios = _paths(reference_audios)
    if not (images or videos or audios):
        raise gr.Error("REF2VA 至少需要一张参考图、一个参考视频或一段参考音频。")
    resolved_seed = int(seed) if seed is not None else None
    try:
        # The model/runtime is intentionally single-concurrency even when its
        # sampling graph uses two GPUs. Parallel requests would duplicate the
        # 40+GB resident working set and make OOMs non-deterministic.
        with _generation_lock:
            runtime = _get_runtime()
            output = runtime.generate(
                prompt=prompt.strip(),
                reference_images=images,
                reference_videos=videos,
                reference_audios=audios,
                aspect_ratio=aspect_ratio,
                size=size,
                duration=float(duration),
                steps=int(steps),
                seed=resolved_seed,
                structured_prompt=False,
                include_audio=include_audio,
                output_codec="mp4-h264",
            )
        return str(output), f"完成 · backend={runtime.parallel_mode} · seed={resolved_seed if resolved_seed is not None else 'random'}"
    except gr.Error:
        raise
    except Exception as error:
        raise gr.Error(str(error)) from error


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="MiniMax H3 REF2VA") as demo:
        gr.Markdown(
            "# MiniMax H3 · REF2VA\n"
            "默认单卡运行：48GB 显卡使用 DynamicVRAM，24GB 显卡自动降级为低显存模式。"
        )
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(label="提示词", lines=12, placeholder="使用 <Picture 1>、<Video 1>、<Audio 1> 指代参考素材")
                with gr.Row():
                    reference_images = gr.File(label="参考图片（最多 9 张）", file_count="multiple", file_types=["image"], type="filepath")
                    reference_videos = gr.File(label="参考视频（最多 3 个）", file_count="multiple", file_types=["video"], type="filepath")
                    reference_audios = gr.File(label="参考音频（最多 3 段）", file_count="multiple", file_types=["audio"], type="filepath")
                with gr.Row():
                    aspect_ratio = gr.Dropdown(["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"], value="9:16", label="画幅")
                    size = gr.Dropdown(["preview", "balanced", "native"], value="preview", label="尺寸（preview 为 480p）")
                    duration = gr.Slider(4, 15, value=5, step=0.5, label="时长（秒）")
                    steps = gr.Slider(8, 60, value=20, step=1, label="采样步数")
                with gr.Row():
                    seed = gr.Number(value=None, precision=0, minimum=0, maximum=2**63 - 1, label="Seed（留空随机）")
                    include_audio = gr.Checkbox(value=True, label="保留生成音频")
                submit = gr.Button("生成视频", variant="primary")
            with gr.Column(scale=2):
                output = gr.Video(label="输出 MP4")
                status = gr.Textbox(label="状态", interactive=False)
        submit.click(
            generate_video,
            inputs=[prompt, reference_images, reference_videos, reference_audios, aspect_ratio, size, duration, steps, seed, include_audio],
            outputs=[output, status],
        )
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
    )
