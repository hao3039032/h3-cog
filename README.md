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
- Opt-in official LightX2V Turbo inference: FL2VA/T2VA uses the 8-step v1.0
  LoRA, while REF2VA automatically uses its matching 4-step v0.1 LoRA.
- Opt-in official PDD 8-step Acc inference (`inference_mode=pdd`): T2VA/FL2VA
  uses the FL2VA Acc LoRA+head bank and REF2VA uses its Ref2VA counterpart,
  with Euler on the node's trained block-boundary sigmas.
- Opt-in experimental NVFP4 model quantization (`model_quantization=nvfp4`):
  swaps the partition-matched DiT and the text encoder for NVFP4 checkpoints,
  targeting native FP4 acceleration on Blackwell GPUs such as RTX 5090.
- default SageAttention with an opt-in per-request Sol-Attn residual INT8 QK
  path, plus PyTorch attention fallback and ComfyUI DynamicVRAM on every GPU size.
- default-on, bit-exact H3 fused modulation for lower AdaLN and gated-residual
  overhead, with an explicit per-request opt-out for A/B validation.
- GPU `av1_nvenc` WebM output with SVT-AV1 fallback; GPU H.264 with x264
  fallback. Native AAC is stream-copied into MP4; WebM encodes it once as Opus.
- Standard Cog HTTP plus a RunPod Serverless handler using the same runtime.

H3's current open release is dense full-attention. MiniMax says sparse
attention will follow. The public path does not apply a lossy cache. Operator
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
| `inference_mode` | `quality` | `quality` uses `steps`; `turbo` uses FL2V 8-step or Ref2V 4-step; `pdd` is a fixed 8-step PDD Acc run |
| `model_quantization` | `int8` | `int8` selects the default INT8 ConvRot DiT and text encoder; `nvfp4` is experimental and only meaningful on Blackwell-class GPUs |
| `attention_backend` | `sage-attention` | `sage-attention` or experimental `sol-int8-qk` |
| `fused_modulation` | `true` | Enables bit-exact H3 AdaLN and gated-residual Triton fusion |
| `structured_prompt` | automatic | True for t2va/fl2va; false for native REF2VA prompts |
| `include_audio` | `true` | Keep or strip H3's generated stereo audio |
| `output_codec` | `mp4-h264` | `webm-av1` or `mp4-h264` |
| `encode_quality` | `26` | Lower is higher quality and larger |

The default native 9:16 canvas is 768×1344 and `preview` is 480×864, while
preserving 32-pixel alignment. A requested 5 seconds is 124
frames, or 5.17 seconds, because H3 only accepts the `17k+5` grid.

`inference_mode=turbo` applies the task-matched official LightX2V LoRA at
strength `1.0`, switches to Euler, and applies MiniMax H3 video/audio sigma
shifts `12/3`. T2VA and FL2VA use
`minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` for exactly 8
steps. REF2VA uses `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors`
for exactly 4 steps. The request's `steps` value is only used in `quality`
mode. Quality remains the default on every interface.

`inference_mode=pdd` runs the official MiniMax-H3 PDD Acc LoRAs
(`alibaba-pai/MiniMax-H3-Acc-LoRAs`) at a fixed 8 NFE: T2VA and FL2VA use
`MiniMax-H3-FL2VA-Acc-8Step.safetensors` and REF2VA uses
`MiniMax-H3-Ref2VA-Acc-8Step.safetensors`. The workflow is
`UNETLoader → MiniMaxH3SigmaShift(12,3) → MiniMaxH3PDDAccApply(nfe=8,
lora/head strength 1.0, on_off_grid=error, partition_check=error)` with Euler,
and the Apply node's trained block-boundary `SIGMAS` output feeds
`SamplerCustomAdvanced` directly — no `BasicScheduler` and no turbo LoRA loader.
`steps` is accepted for API compatibility but ignored; the effective step count
is always 8 and is logged together with the caller's `requested_steps`. PDD is
mutually exclusive with EasyCache: a PDD request carrying a cache tuning is
rejected with an explicit error before anything is submitted to ComfyUI. Fused
modulation and the optional Sol INT8-QK patch still stack after the Apply node.
First release intentionally does not expose 4/6 NFE, custom partitions, or a
PDD strength knob.

The two PDD Acc files (1.37GB each, 2.56GiB total) and the two NVFP4 DiTs are
the only weights sourced from Hugging Face rather than ModelScope; only the
PDD files come from the official MiniMax release. Both sets are optional:
missing PDD or NVFP4 files never block Quality/Turbo/INT8 startup, but a PDD or
NVFP4 request fails with the exact missing paths. Before the first PDD request
the runtime also
verifies via ComfyUI `object_info` that `MiniMaxH3PDDAccApply` and
`MiniMaxH3SigmaShift` are available, so a node-less image fails fast instead of
during sampling. PDD weights keep the MiniMax H3 Community License and the
`MINIMAX_H3_LICENSE_ACCEPTED=1` gate.

`model_quantization=nvfp4` is an experimental, request-level model swap. It
replaces the partition DiT with `MiniMax_H3_{FL2VA,Ref2VA}_pruned_nvfp4.safetensors`
(12.53GB each) and the text encoder with the official
`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (15.69GB). The two NVFP4 DiTs
are third-party single-pass conversions pinned to
`Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot` at a fixed revision; unlike every other weight in
this repository they are not published by MiniMax or Comfy-Org, so treat their
outputs as unvalidated until your own same-seed INT8 comparison passes. NVFP4
compute is natively accelerated only on Blackwell (SM120+) GPUs such as RTX
5090 — on older cards ComfyUI falls back to dequantizing emulation, which is
slower than INT8. The three NVFP4 files are optional: they never block INT8
startup, and an NVFP4 request fails fast with the exact missing paths. Both
quantization profiles can coexist in one `WEIGHTS_DIR`; swapping them rewrites
the shared ComfyUI symlinks and switches the DiT cache route. NVFP4 stacks
with `quality`, `turbo`, and `pdd` inference modes.

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

`attention_backend=sol-int8-qk` inserts the H3 scheduled Sol-Attn patch for
eligible main-DiT attention calls. It enables residual INT8 QK, keeps P·V in
BF16, preserves conditioning KV blocks exactly, and schedules tau from 1.0 to
0.8. The already configured SageAttention path remains the fallback for short,
gated, unsupported, or failed non-strict calls. All interfaces default to
`sage-attention`, which leaves the existing workflow unchanged.

FP32 matmuls stay strict by default. Set `H3_FP32_MATMUL_TF32=1` to allow
cuBLAS TF32 for the FP32 VideoVAE; this changes numerics and is intended for
same-seed speed and quality A/B tests.

`fused_modulation=true` inserts `MiniMaxH3FusedModulation` before an optional
EasyCache patch. The image pins `Saganaki22/ComfyUI-sol-attn` v0.6.2 at commit
`930a4d6e432ff8b8ed5e30ff2f72519b92d69bdf`. This optimization does not change
attention topology, skip sampling work, or alter weights: it fuses H3's
segmented AdaLN scale/shift and gated residual updates while explicitly
reproducing eager BF16 rounding. Upstream's real ComfyUI `DiTBlock` regression
uses `torch.equal` for bit-exact comparison. All product entry points enable the
fusion by default; pass `fused_modulation=false` to run the original eager path.

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

The first build installs CUDA 12.8 / PyTorch 2.11 and pins ComfyUI v0.33.0
(`2f35f4a08176d993cded35dac3332be4f7287f41`), whose joint audio/video sampler
contract matches the official H3 nodes and whose carried-audio mechanics are
required by the PDD Acc node pack. The published production image contains
its model weights.

## Verified weight sources

The task-routing production set contains about 84.02GB of weights, plus two
optional PDD Acc files adding 2.56GiB:

```text
diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors
text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
vae/minimax_h3_video_vae_fp32.safetensors
vae/minimax_h3_audio_vae_fp32.safetensors
pdd_acc/MiniMax-H3-FL2VA-Acc-8Step.safetensors      (optional, PDD mode)
pdd_acc/MiniMax-H3-Ref2VA-Acc-8Step.safetensors     (optional, PDD mode)
diffusion_models/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors   (optional, NVFP4)
diffusion_models/MiniMax_H3_Ref2VA_pruned_nvfp4.safetensors  (optional, NVFP4)
text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors   (optional, NVFP4)
```

The FP32 video VAE is the deterministic Comfy-native repack published at
`Austusm/minimax_h3_video_vae`. It preserves every official F32 tensor byte
and appends the two `[24]` latent normalization buffers expected by ComfyUI;
the runtime enables `--fp32-vae`. Set `H3_VIDEO_VAE_PRECISION=fp16` to fall
back to the smaller conversion for capacity-constrained workers. All selected
weights are pinned to ModelScope except the two PDD Acc files, which are pinned
to the official `alibaba-pai/MiniMax-H3-Acc-LoRAs` Hugging Face release, and
the two NVFP4 DiTs, which are pinned to the third-party
`Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot` conversion at a fixed commit.
Provision them under
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
persistent network volume must already contain the 84.02GB model set.

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

Each RunPod response includes schema-versioned metrics (schema v3, which added
`model_quantization`) for total generation and
encode time, dimensions, frame count, exact seed, output size and SHA-256, and
the non-secret cache configuration. Compare every cache candidate with its
same-seed `off` baseline. Promote a profile only after visual review of motion,
audio synchronization, first/last-frame alignment, and loop seam; no H3
EasyCache speedup or quality claim is made before those GPU A/B results exist.

## Single-GPU memory and cost

Each pruned INT8 DiT is 20.97GB; the INT8 Qwen encoder is 27.14GB, visual VAE
10.42GB, audio VAE 0.61GB, and the two Turbo LoRAs total 3.91GB. Both DiT
partitions bring the selected set to about 84.02GB; enabling PDD adds
2.56GiB for the two Acc files (1.37GB each), and enabling NVFP4 adds about
40.74GB (two 12.53GB DiTs plus the 15.69GB AWQ encoder). They cannot all remain
resident on a 24GB or 48GB card, so
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
additionally requires the accepted model license, 84.02GB of weights, a 48GB
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
- [Official LightX2V MiniMax H3 Turbo LoRAs and settings](https://github.com/ModelTC/Minimax-H3-Turbo)
- [Official PDD Acc LoRAs](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs) and the [ComfyUI PDD node](https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc)
- [MiniMax base-mode prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
