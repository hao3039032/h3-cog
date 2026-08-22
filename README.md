# MiniMax H3 Cog for app.nz

MiniMax H3 text, first/last-frame, and reference-to-video generation in a
portable [Cog](https://github.com/replicate/cog). The public task contract is
`t2va`, `fl2va`, or `ref2va`; `t2va` and `fl2va` share the FL2VA INT8 DiT while
REF2VA uses its own INT8 DiT. The runtime expects all model files to be
provisioned externally; missing paths fail startup instead of triggering network
downloads or integrity scans.

[![Deploy H3 on app.nz](https://app.nz/deploy-button.svg)](https://app.nz/deploy?template=minimax-h3)

## Important model-license boundary

This repository's adapter code is MIT licensed. **MiniMax H3 weights are not
MIT licensed.** They use the [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE),
which currently defines an applicable territory that excludes the United
States, European Union, United Kingdom, Republic of Korea, and other uses in its
acceptable-use policy. Review the current upstream terms with your counsel.

The deployer must explicitly acknowledge the weight license before running:

```sh
MINIMAX_H3_LICENSE_ACCEPTED=1
```

Setting the variable is your acknowledgement, not app.nz's. Do not deploy the
model to an excluded territory or expose a hosted service where the license
does not permit it.

## What is included

- Explicit task routing plus REF2VA support for up to 9 reference images, 3
  reference videos (including their audio), and 3 standalone audio clips.
- FL2VA first-frame, last-frame, and loop conditioning with deterministic target
  canvas pre-cropping.
- Native H3 aspect ratios and a 768px short-edge quality tier.
- Official `res_multistep` sampler and 24-step deployment default; 12–16 steps and
  reduced pixel area are exposed for previews.
- optional SageAttention when supplied by the image operator, with PyTorch
  attention fallback and ComfyUI DynamicVRAM on every GPU size.
- opt-in Kijai Sol-Attn profiles for training-free sparse-attention A/B tests;
  SageAttention remains the dense and error fallback.
- GPU `av1_nvenc` WebM output with SVT-AV1 fallback; GPU H.264 with x264
  fallback. Native audio is remuxed as Opus or AAC.
- Standard Cog HTTP plus a RunPod Serverless handler using the same runtime.

H3's current open release is dense full-attention. Sol-Attn is an experimental,
lossy runtime sparsification path rather than a new H3 checkpoint. The default
path remains dense and does not apply a lossy cache. Operator
sweeps can use ComfyUI's built-in EasyCache behind a short-lived signed
envelope, described below. We do not apply CG-Taylor or a "latent teleport"
cache: neither has been validated as an H3-compatible node against its joint
audio/video latent stream and first/last-frame fidelity. SageAttention is
compiled from a pinned source build for SM80, SM89, and SM120; SM120 requires
CUDA 12.8 or newer. It is usable on RTX 6000 D / RTX PRO 6000-class Blackwell
cards, but its H3 visual output still needs a same-seed comparison before
treating it as a quality-neutral default for production traffic.

## Inputs

| Input | Default | Notes |
| --- | --- | --- |
| `prompt` | required | Scene, motion, camera, dialogue, SFX, and music |
| `task` | required | `t2va`, `fl2va`, or `ref2va` |
| `first_frame` | empty | FL2VA opening keyframe; required when `loop=true` |
| `last_frame` | empty | FL2VA closing keyframe |
| `loop` | `false` | Reuses `first_frame` as the closing keyframe |
| `reference_images` | empty | Up to 9 R2V images; prompt tags are `<Picture 1>`, etc. |
| `reference_videos` | empty | Up to 3 R2V videos; exposes `<Video n>` and its `<Audio n>` |
| `reference_audios` | empty | Up to 3 standalone R2V audio clips |
| `aspect_ratio` | `9:16` | `16:9`, `9:16`, `1:1`, `4:3`, `3:4`, `21:9` |
| `size` | `preview` | `preview`, `balanced`, `native` |
| `duration` | `5` | 4–15 seconds; snaps upward to H3's `17k+5` frame grid |
| `steps` | `24` | 20 official; 12–16 preview; allowed 8–60 |
| `sol_profile` | `off` | Experimental `off`, `conservative`, or `balanced`; positive profiles change attention numerics |
| `structured_prompt` | automatic | True for t2va/fl2va; false for native REF2VA prompts |
| `include_audio` | `true` | Keep or strip H3's generated stereo audio |
| `output_codec` | `mp4-h264` | `webm-av1` or `mp4-h264` |
| `encode_quality` | `26` | Lower is higher quality and larger |

The default native 9:16 canvas is 768×1344 and `preview` is 480×864, while
preserving 32-pixel alignment. A requested 5 seconds is 124
frames, or 5.17 seconds, because H3 only accepts the `17k+5` grid.

On a 48GB L40S, ComfyUI's DynamicVRAM path measured about 7% faster than
estimate-based loading on the same 480x864, 362-frame workload.

The runtime is native ComfyUI, single-process, and uses the first CUDA device.
GPUs reporting at least 80GiB use ComfyUI's HighVRAM mode so the shared encoder,
VAEs, and both INT8 DiTs have the opportunity to remain cached; smaller cards
use normal DynamicVRAM and let ComfyUI evict components as needed.
`H3_HIGHVRAM=0` selects DynamicVRAM for an A/B test. Retired values of
`H3_PARALLEL_MODE=raylight/auto/fsdp/dual` fail startup with an explicit
"Raylight has been removed" error; unset it or use `single`.

Task switching defaults to `H3_DIT_SWITCH_POLICY=auto`: no `/free` request is
sent when moving between the FL2VA and REF2VA partitions, so a 96GB card can
keep both DiTs warm. Set `H3_DIT_SWITCH_POLICY=evict` only when diagnosing
memory pressure; it calls ComfyUI `/free` on a real partition switch.
See [MODELSCOPE.md](MODELSCOPE.md) for the Gradio Studio entry point and
deployment requirements.

The Cog image builds SageAttention for SM80, SM89, and SM120. H100 uses
PyTorch SDPA because the SM90 FP8 SageAttention path produced corrupted H3
video during validation. Set `H3_SAGE_ATTENTION=0` for a same-seed PyTorch SDPA
comparison on supported cards. `H3_LOWVRAM` is retired and ignored;
`H3_RESERVE_VRAM_GB` defaults to `1.0`.

FP32 matmuls stay strict by default. Set `H3_FP32_MATMUL_TF32=1` to allow
cuBLAS TF32 for the FP32 VideoVAE; this changes numerics and is intended for
same-seed speed and quality A/B tests.

### Experimental Sol-Attn profiles

The image pins `kijai/ComfyUI-SolAttn_triton` at commit
`dfc2e31a41afd72bd53dd2137fc8b2931d5ec192`. `sol_profile=off` does not insert
the node. Both enabled profiles apply Sol-Attn only from 20% through 90% of the
sampling trajectory, keep blocks `0-2` and the final block dense, use H3's 2D
frame Morton order, and keep conditioning plus audio query rows exact. The
`conservative` profile uses `tau=1.0` and BF16 P/V; `balanced` uses `tau=1.3`
and enables INT8 P/V. Both use INT8 Q/K and retain the existing SageAttention
path for dense windows, ineligible calls, and kernel failures.

Sol-Attn compiles and autotunes Triton kernels for each new token length. Do
not time the first request at a new resolution/frame count; compare the second
warm request against an interleaved same-seed `off` run. This is approximate
sparse attention, so promotion requires review of speech, synchronized sound,
fast motion, FL2VA endpoint fidelity, and Ref2VA identity/reference adherence.
The branch makes no production speed or quality claim before those GPU runs.

## Local Cog usage

For a portable Gradio container instead of Cog, see [DOCKER.md](DOCKER.md). The
default deployment listens on container port `7860`, mounts verified weights at
`/weights`, and supports the platform-neutral `GRADIO_PUBLIC_PORT` /
`GRADIO_PUBLIC_PROTO` proxy overrides. AutoDL can override `PORT` and `H3_PORT`
to `6006`; that requirement is platform-specific rather than an image default.

```sh
export MINIMAX_H3_LICENSE_ACCEPTED=1
cog run \
  -i task=ref2va \
  -i reference_images=@character.png \
  -i reference_videos=@camera-move.mp4 \
  -i reference_audios=@voice.wav \
  -i prompt="Keep <Picture 1>'s identity, follow <Video 1>'s camera move, and use <Audio 2>'s voice."
```

For FL2VA, pass `task=fl2va` plus a first and/or last frame (or `loop=true`
with a first frame). For T2VA, pass `task=t2va` with no media. `preview`
produces a 480-pixel short edge and is the recommended default for
latency-sensitive use.

The first build installs CUDA 12.8 / PyTorch 2.11 and pins ComfyUI v0.31.0
(`43cb4fffc89bba20ab7bd61467a36d0339338dab`), whose joint audio/video sampler
contract matches the official H3 nodes. The published production image contains
its model weights.

## Verified weight sources

The task-routing production set contains about 80.10GB of weights:

```text
diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors
text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
vae/minimax_h3_video_vae_fp32.safetensors
vae/minimax_h3_audio_vae_fp32.safetensors
```

The FP32 video VAE is the deterministic Comfy-native repack published at
`Austusm/minimax_h3_video_vae`. It preserves every official F32 tensor byte
and appends the two `[24]` latent normalization buffers expected by ComfyUI;
the runtime enables `--fp32-vae`. Set `H3_VIDEO_VAE_PRECISION=fp16` to fall
back to the smaller conversion for capacity-constrained workers. All selected
weights are pinned to ModelScope. Provision them under
`${WEIGHTS_DIR}/MiniMax-H3` before startup; the runtime links existing files
into ComfyUI and consumes them as-is.

Each entry includes its exact source URL, size, and SHA-256 for deployment-side
provisioning and verification. The runtime makes no weight-network requests.

## RunPod Serverless

Build and push the same Cog image, then override the container command:

```text
python -u /src/rp_handler.py
```

The handler requires `task`, accepts FL keyframe URLs plus Cog-compatible
reference URL lists and `*_urls` aliases. It rejects private/reserved network
targets, caps each file at 512MiB, and returns a bounded base64 media output
compatible with app.nz's Cog serverless shim. Set minimum workers to zero. A
persistent network volume must already contain the 80.10GB model set.

### Authenticated acceleration sweeps

EasyCache is experimental and may trade temporal, audio, or keyframe fidelity
for speed. It is deliberately absent from Cog inputs and disabled by default.
Only the RunPod operator can activate it by signing a bounded, expiring `_tuning`
object with `H3_TUNING_SECRET`. The runtime never logs or returns that secret or
the request signature.

Create a fixed-seed baseline plus conservative, balanced, and aggressive
candidates:

```sh
export H3_TUNING_SECRET="$(openssl rand -hex 32)"
python h3_sweep.py request.json --sweep-id gallery-a01 > sweep.json
```

Set the same secret only on the private worker. Each candidate expires within
one hour, and the server rejects unknown fields, invalid signatures, cache
thresholds above `0.30`, and invalid cache windows. Custom settings can be
signed with `sign_tuning` from `h3_tuning.py`; the three named profiles are the
recommended first pass. Never expose the secret in the app.nz client or public
Cog schema.

Each RunPod response includes schema-versioned metrics for total generation and
encode time, dimensions, frame count, exact seed, output size and SHA-256, and
the non-secret cache configuration. Compare every cache candidate with its
same-seed `off` baseline. Promote a profile only after visual review of motion,
audio synchronization, first/last-frame alignment, and loop seam; no H3
EasyCache speedup or quality claim is made before those GPU A/B results exist.

## Single-GPU memory and cost

Each pruned INT8 DiT is 20.97GB; the INT8 Qwen encoder is 27.14GB, visual VAE
10.42GB, and audio VAE 0.61GB. Both DiT partitions bring the selected set to
about 80.10GB. They cannot all remain resident on a 24GB or 48GB card, so
ComfyUI stages components between prompt encoding, denoising, and VAE decode.
Use at least 96GB system RAM on DynamicVRAM workers and a persistent weight
volume. GPUs reporting at least 80GiB use HighVRAM after the first load and may
keep both partitions warm on 96GB-class cards.

Use preview/balanced for most requests, keep the process warm for bursts, and
scale the worker to zero when idle. app.nz's H3 template publishes the final
customer rate and uses an 80% platform markup over the selected RunPod
serverless compute price; billing remains per execution second.

## Tests

```sh
uv run --extra dev pytest
python -m py_compile h3_*.py predict.py rp_handler.py weights.py
cog build -t h3-cog:local
```

Unit tests cover the H3 frame grid, native dimensions, official keyframe prompt
prefixes, task routing, model-cache policy, GPU/CPU encoder fallback, explicit
license gate, and externally provisioned weight paths. A real GPU smoke test
additionally requires the accepted model license, 80.10GB of weights, a 48GB
CUDA GPU, and enough host RAM.

`demo_prompts.json` is the deterministic seven-clip launch suite covering text,
image, first/last-frame, loop, vertical, square, and ultrawide output. Keep seed,
prompt, dimensions, steps, and source keyframes attached to every gallery
record. The suite remains prompt-only until the deployer accepts the weight
license and supplies a compliant worker region; no synthetic or untested clip
is presented as H3 output.

The five required source keyframes are included in `gallery-keyframes/` as
`<demo-id>-first.png` and `<demo-id>-last.png`. After the license/region gate is
cleared, render and prepare app.nz sidecars with:

```sh
python h3_gallery.py --output ./h3-gallery-renders
cd ../app-site
bun scripts/ingest-videos.mjs --in ../h3-cog/h3-gallery-renders \
  --out public/assets/videos --base-url /assets/videos --source minimax-h3
```

The second command runs from the app-site repository. It encodes delivery WebM
and poster files and upserts stable `minimax-h3:<demo-id>` gallery records. Use
`--limit 1` for the first licensed smoke render.

## Upstream references

- [MiniMax H3 model card and license](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [ComfyUI H3 workflow documentation](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)
- [ComfyUI-packaged checkpoints](https://huggingface.co/Comfy-Org/MiniMax-H3)
- [Kijai ComfyUI Sol-Attn](https://github.com/kijai/ComfyUI-SolAttn_triton)
- [MiniMax base-mode prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
