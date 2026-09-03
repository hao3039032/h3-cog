from cog import BaseRunner, Input, Path

from h3_runtime import H3Runtime


class Runner(BaseRunner):
    def setup(self) -> None:
        self.runtime = H3Runtime()

    def run(
        self,
        prompt: str = Input(description="Scene, motion, camera, dialogue, sound, and music prompt"),
        task: str = Input(description="Explicit MiniMax H3 task", choices=["t2va", "fl2va", "ref2va"]),
        first_frame: Path | None = Input(description="FL2VA first keyframe; required for loop", default=None),
        last_frame: Path | None = Input(description="FL2VA last keyframe", default=None),
        loop: bool = Input(description="Reuse first_frame as the final keyframe", default=False),
        reference_images: list[Path] = Input(description="Optional REF2VA reference images (up to 9); use <Picture 1>, etc. in the prompt", default=[]),
        reference_videos: list[Path] = Input(description="Optional REF2VA reference videos with their soundtracks (up to 3); use <Video 1>/<Audio 1>", default=[]),
        reference_audios: list[Path] = Input(description="Optional standalone REF2VA reference audio clips (up to 3); use <Audio 1>, etc.", default=[]),
        aspect_ratio: str = Input(default="9:16", choices=["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"]),
        size: str = Input(description="Preview is 480p and recommended; native uses H3's full 768px short edge", default="preview", choices=["preview", "balanced", "native"]),
        duration: float = Input(description="Requested seconds; snaps to H3's 17k+5 frame grid at 24fps", default=5.0, ge=4.0, le=15.0),
        steps: int = Input(description="24 is the deployment quality default; 12-16 is useful for previews", default=24, ge=8, le=60),
        inference_mode: str = Input(description="Quality uses the requested steps; Turbo selects the official FL2V 8-step or Ref2V 4-step LoRA automatically; PDD runs the official 8-step PDD Acc LoRA+heads", default="quality", choices=["quality", "turbo", "pdd"]),
        model_quantization: str = Input(description="INT8 ConvRot is the default; NVFP4 is experimental and natively accelerated on Blackwell GPUs such as RTX 5090", default="int8", choices=["int8", "nvfp4"]),
        attention_backend: str = Input(description="H3 attention backend; Sol keeps Sage as the fallback path", default="sage-attention", choices=["sage-attention", "sol-int8-qk"]),
        fused_modulation: bool = Input(description="Enable bit-exact H3 AdaLN and gated-residual kernel fusion", default=True),
        seed: int | None = Input(description="Blank selects a cryptographically random seed", default=None, ge=0, le=9223372036854775807),
        structured_prompt: bool | None = Input(description="Defaults to true for t2va/fl2va and false for native REF2VA prompts", default=None),
        include_audio: bool = Input(description="Keep H3's native synchronized stereo audio", default=True),
        output_codec: str = Input(description="GPU NVENC is used when available, then a CPU fallback", default="mp4-h264", choices=["webm-av1", "mp4-h264"]),
        encode_quality: int = Input(description="Lower is higher quality/larger; 26 is a strong web default", default=26, ge=16, le=45),
    ) -> Path:
        return Path(self.runtime.generate(
            prompt=prompt,
            task=task,
            first_frame=first_frame,
            last_frame=last_frame,
            loop=loop,
            reference_images=reference_images,
            reference_videos=reference_videos,
            reference_audios=reference_audios,
            aspect_ratio=aspect_ratio,
            size=size,
            duration=duration,
            steps=steps,
            inference_mode=inference_mode,
            model_quantization=model_quantization,
            attention_backend=attention_backend,
            fused_modulation=fused_modulation,
            seed=seed,
            structured_prompt=structured_prompt,
            include_audio=include_audio,
            output_codec=output_codec,
            encode_quality=encode_quality,
        ))
