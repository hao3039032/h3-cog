from cog import BaseRunner, Input, Path

from h3_runtime import H3Runtime


class Runner(BaseRunner):
    def setup(self) -> None:
        self.runtime = H3Runtime()

    def run(
        self,
        prompt: str = Input(description="Scene, motion, camera, dialogue, sound, and music prompt"),
        first_frame: Path | None = Input(description="Optional first frame for image-to-video", default=None),
        last_frame: Path | None = Input(description="Optional final frame for first/last-frame interpolation", default=None),
        reference_images: list[Path] = Input(description="Optional R2V reference images (up to 9); use <Picture 1>, etc. in the prompt", default=[]),
        reference_videos: list[Path] = Input(description="Optional R2V reference videos with their soundtracks (up to 3); use <Video 1>/<Audio 1>", default=[]),
        reference_audios: list[Path] = Input(description="Optional standalone R2V reference audio clips (up to 3); use <Audio 1>, etc.", default=[]),
        aspect_ratio: str = Input(default="16:9", choices=["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"]),
        size: str = Input(description="Preview is fastest; native uses H3's full 768px short edge", default="balanced", choices=["preview", "balanced", "native"]),
        duration: float = Input(description="Requested seconds; snaps to H3's 17k+5 frame grid at 24fps", default=5.0, ge=4.0, le=15.0),
        steps: int = Input(description="20 is the official quality setting; 12-16 is useful for previews", default=20, ge=8, le=30),
        seed: int | None = Input(description="Blank selects a cryptographically random seed", default=None, ge=0, le=9223372036854775807),
        structured_prompt: bool = Input(description="Wrap plain prompts in MiniMax's documented audiovisual prompt structure", default=True),
        loop: bool = Input(description="Close the last frame onto first_frame for a seamless content loop", default=False),
        include_audio: bool = Input(description="Keep H3's native synchronized stereo audio", default=True),
        output_codec: str = Input(description="GPU NVENC is used when available, then a CPU fallback", default="webm-av1", choices=["webm-av1", "mp4-h264"]),
        encode_quality: int = Input(description="Lower is higher quality/larger; 26 is a strong web default", default=26, ge=16, le=45),
    ) -> Path:
        return Path(self.runtime.generate(
            prompt=prompt,
            first_frame=first_frame,
            last_frame=last_frame,
            reference_images=reference_images,
            reference_videos=reference_videos,
            reference_audios=reference_audios,
            aspect_ratio=aspect_ratio,
            size=size,
            duration=duration,
            steps=steps,
            seed=seed,
            structured_prompt=structured_prompt,
            loop=loop,
            include_audio=include_audio,
            output_codec=output_codec,
            encode_quality=encode_quality,
        ))
