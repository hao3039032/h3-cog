"""Gradio entry point for a ModelScope Studio or a local H3 workstation."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import gradio as gr
from fastapi.middleware import Middleware

from h3_gradio import (
    PublicOriginMiddleware,
    configured_public_port,
    configured_public_proto,
)
from h3_seed import resolve_seed
from h3_runtime import H3Runtime
from h3_workflow import infer_task

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


def _single_path(value: Any) -> Path | None:
    paths = _paths(value)
    return paths[0] if paths else None


def _infer_task(
    first_frame: Any,
    last_frame: Any,
    loop: bool,
    reference_images: Any,
    reference_videos: Any,
    reference_audios: Any,
) -> str:
    return infer_task(
        first_frame=_single_path(first_frame),
        last_frame=_single_path(last_frame),
        loop=bool(loop),
        reference_count=len(
            _paths(reference_images) + _paths(reference_videos) + _paths(reference_audios)
        ),
    )


def generate_video(
    prompt: str,
    first_frame: Any,
    last_frame: Any,
    loop: bool,
    reference_images: Any,
    reference_videos: Any,
    reference_audios: Any,
    aspect_ratio: str,
    size: str,
    duration: float,
    steps: int,
    fused_modulation: bool,
    seed: str | None,
    include_audio: bool,
) -> tuple[str, str]:
    if not prompt or not prompt.strip():
        raise gr.Error("请输入提示词。")
    try:
        task = _infer_task(
            first_frame,
            last_frame,
            loop,
            reference_images,
            reference_videos,
            reference_audios,
        )
        resolved_seed = resolve_seed(seed)
        # Keep one ComfyUI prompt queue and one coherent model-cache state.
        with _generation_lock:
            runtime = _get_runtime()
            output = runtime.generate(
                prompt=prompt.strip(),
                task=task,
                first_frame=_single_path(first_frame),
                last_frame=_single_path(last_frame),
                loop=bool(loop),
                reference_images=_paths(reference_images),
                reference_videos=_paths(reference_videos),
                reference_audios=_paths(reference_audios),
                aspect_ratio=aspect_ratio,
                size=size,
                duration=float(duration),
                steps=int(steps),
                fused_modulation=bool(fused_modulation),
                seed=resolved_seed,
                structured_prompt=task != "ref2va",
                include_audio=include_audio,
                output_codec="mp4-h264",
            )
        return (
            str(output),
            f"完成 · task={task} · backend={runtime.parallel_mode} · "
            f"fused_modulation={'on' if fused_modulation else 'off'} · "
            f"seed={resolved_seed}",
        )
    except gr.Error:
        raise
    except Exception as error:
        raise gr.Error(str(error)) from error


def _task_display(
    first_frame: Any,
    last_frame: Any,
    loop: bool,
    reference_images: Any,
    reference_videos: Any,
    reference_audios: Any,
) -> str:
    try:
        task = _infer_task(
            first_frame,
            last_frame,
            loop,
            reference_images,
            reference_videos,
            reference_audios,
        )
        return f"当前任务：{task}"
    except ValueError as error:
        return f"输入冲突：{error}"


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="MiniMax H3") as demo:
        gr.Markdown(
            "# MiniMax H3 · t2va / fl2va / ref2va\n"
            "默认单进程 native ComfyUI 路径；上传内容会自动推导任务类型。"
        )
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="提示词",
                    lines=12,
                    placeholder="使用 <Picture 1>、<Video 1>、<Audio 1> 指代 REF2VA 参考素材",
                )
                with gr.Row():
                    first_frame = gr.Image(label="首帧", type="filepath")
                    last_frame = gr.Image(label="尾帧", type="filepath")
                    loop = gr.Checkbox(value=False, label="首帧循环")
                with gr.Row():
                    reference_images = gr.File(label="参考图片（最多 9 张）", file_count="multiple", file_types=["image"], type="filepath")
                    reference_videos = gr.File(label="参考视频（最多 3 个）", file_count="multiple", file_types=["video"], type="filepath")
                    reference_audios = gr.File(label="参考音频（最多 3 段）", file_count="multiple", file_types=["audio"], type="filepath")
                with gr.Row():
                    task_display = gr.Textbox(value="当前任务：t2va", label="任务路由", interactive=False)
                with gr.Row():
                    aspect_ratio = gr.Dropdown(["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"], value="9:16", label="画幅")
                    size = gr.Dropdown(["preview", "balanced", "native"], value="preview", label="尺寸（preview 为 480p）")
                    duration = gr.Slider(4, 15, value=5, step=0.5, label="时长（秒）")
                    steps = gr.Slider(8, 60, value=24, step=1, label="采样步数")
                with gr.Row():
                    seed = gr.Textbox(value="", label="Seed（留空随机）")
                    include_audio = gr.Checkbox(value=True, label="保留生成音频")
                    fused_modulation = gr.Checkbox(
                        value=True,
                        label="Fused Modulation（逐位精确）",
                    )
                submit = gr.Button("生成视频", variant="primary")
            with gr.Column(scale=2):
                output = gr.Video(label="输出 MP4")
                status = gr.Textbox(label="状态", interactive=False)

        routing_inputs = [
            first_frame,
            last_frame,
            loop,
            reference_images,
            reference_videos,
            reference_audios,
        ]
        for component in routing_inputs:
            component.change(
                _task_display,
                inputs=routing_inputs,
                outputs=task_display,
            )
        submit.click(
            generate_video,
            inputs=[
                prompt,
                first_frame,
                last_frame,
                loop,
                reference_images,
                reference_videos,
                reference_audios,
                aspect_ratio,
                size,
                duration,
                steps,
                fused_modulation,
                seed,
                include_audio,
            ],
            outputs=[output, status],
        )
    return demo


demo = build_demo()

if __name__ == "__main__":
    public_port = configured_public_port(os.getenv("GRADIO_PUBLIC_PORT"))
    public_proto = configured_public_proto(os.getenv("GRADIO_PUBLIC_PROTO"))
    app_kwargs = {}
    middleware_options = {}
    if public_port is not None:
        middleware_options["public_port"] = public_port
    if public_proto is not None:
        middleware_options["public_proto"] = public_proto
    if middleware_options:
        app_kwargs["middleware"] = [
            Middleware(PublicOriginMiddleware, **middleware_options)
        ]
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
        app_kwargs=app_kwargs,
    )
