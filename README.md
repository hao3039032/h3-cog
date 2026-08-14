# MiniMax H3 Cog for app.nz

MiniMax H3 reference-to-video in a portable [Cog](https://github.com/replicate/cog).
This deployment intentionally uses only REF2VA: one or more reference images,
videos, or audio clips are required. The production image bakes the transformer,
text encoder, and both VAEs so a scale-to-zero cold start does not download
weights, fetch a manifest, or hash tens of gigabytes.

[![Deploy H3 on app.nz](https://app.nz/deploy-button.svg)](https://app.nz/deploy?template=minimax-h3)

## Important model-license boundary

This repository's adapter code is MIT licensed. **MiniMax H3 weights are not
MIT licensed.** They use the [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE),
which currently defines an applicable territory that excludes the United
States, European Union, United Kingdom, Republic of Korea, and other uses in its
acceptable-use policy. Review the current upstream terms with your counsel.

The runtime refuses to download weights until the deployer explicitly sets:

```sh
MINIMAX_H3_LICENSE_ACCEPTED=1
```

Setting the variable is your acknowledgement, not app.nz's. Do not deploy the
model to an excluded territory or expose a hosted service where the license
does not permit it.

## What is included

- REF2VA: up to 9 reference images, 3 reference videos (including their audio),
  and 3 standalone audio clips for identity, style, motion, camera, or voice.
- Native H3 aspect ratios and a 768px short-edge quality tier.
- Official `res_multistep` sampler and 24-step deployment default; 12–16 steps and
  reduced pixel area are exposed for previews.
- optional SageAttention when supplied by the image operator, with PyTorch
  attention fallback; one 48GB card uses normal DynamicVRAM, while a 24GB card
  is limited to the low-VRAM correctness fallback.
- GPU `av1_nvenc` WebM output with SVT-AV1 fallback; GPU H.264 with x264
  fallback. Native audio is remuxed as Opus or AAC.
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
| `reference_images` | empty | Up to 9 R2V images; prompt tags are `<Picture 1>`, etc. |
| `reference_videos` | empty | Up to 3 R2V videos; exposes `<Video n>` and its `<Audio n>` |
| `reference_audios` | empty | Up to 3 standalone R2V audio clips |
| `aspect_ratio` | `9:16` | `16:9`, `9:16`, `1:1`, `4:3`, `3:4`, `21:9` |
| `size` | `preview` | `preview`, `balanced`, `native` |
| `duration` | `5` | 4–15 seconds; snaps upward to H3's `17k+5` frame grid |
| `steps` | `24` | 20 official; 12–16 preview; allowed 8–60 |
| `structured_prompt` | `false` | Optional FL-style audiovisual wrapper; native REF2VA prompts should leave this off |
| `include_audio` | `true` | Keep or strip H3's generated stereo audio |
| `output_codec` | `mp4-h264` | `webm-av1` or `mp4-h264` |
| `encode_quality` | `26` | Lower is higher quality and larger |

The default native 9:16 canvas is 768×1344 and `preview` is 480×864, while
preserving 32-pixel alignment. A requested 5 seconds is 124
frames, or 5.17 seconds, because H3 only accepts the `17k+5` grid.

On a 48GB L40S, ComfyUI's DynamicVRAM path measured about 7% faster than
estimate-based loading on the same 480x864, 362-frame workload.

The runtime defaults to `H3_PARALLEL_MODE=single`. GPUs reporting at least
80GiB use ComfyUI's HighVRAM mode so the INT8 weight set can stay resident;
28-to-79GiB cards use normal DynamicVRAM; a single 24GB GPU automatically adds
ComfyUI's `--lowvram` flag for correctness. `H3_HIGHVRAM=0` selects DynamicVRAM
for an A/B test, while `H3_LOWVRAM=1` remains the emergency override. Raylight
FSDP2 + Ulysses2 remains available as an explicit two-GPU compatibility path.
See [MODELSCOPE.md](MODELSCOPE.md) for the Gradio Studio entry point and
deployment requirements.

The Cog image builds SageAttention for SM80, SM89, and SM120. H100 uses
PyTorch SDPA because the SM90 FP8 SageAttention path produced corrupted H3
video during validation. Set `H3_SAGE_ATTENTION=0` for a same-seed PyTorch SDPA
comparison on supported cards. Set `H3_LOWVRAM=1` only as an emergency fallback;
`H3_RESERVE_VRAM_GB` defaults to `1.0`.

## Local Cog usage

```sh
export MINIMAX_H3_LICENSE_ACCEPTED=1
cog run \
  -i reference_images=@character.png \
  -i reference_videos=@camera-move.mp4 \
  -i reference_audios=@voice.wav \
  -i prompt="Keep <Picture 1>'s identity, follow <Video 1>'s camera move, and use <Audio 2>'s voice."
```

At least one reference input is required. `preview` produces a 480-pixel short
edge and is the recommended default for latency-sensitive use.

The first build installs CUDA 12.8 / PyTorch 2.11 and pins ComfyUI v0.31.0
(`43cb4fffc89bba20ab7bd61467a36d0339338dab`), whose joint audio/video sampler
contract matches Raylight's H3 distributed forward. The published production
image contains its model weights.

## Verified weight sources

The REF2VA-only production set contains about 53.92GB of weights:

```text
diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors
text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
vae/minimax_h3_video_vae_fp16.safetensors
vae/minimax_h3_audio_vae_fp32.safetensors
```

All four weights are pinned to ModelScope. Development images resume downloads,
verify exact size and SHA-256, and link the cache into ComfyUI. A
production image sets `H3_BAKED_WEIGHTS_VERIFIED=1` after verifying its
immutable layers, so startup checks sizes without network access or a full
SHA-256 pass.

Each entry includes its exact source URL, size, and SHA-256. No remote manifest
request is made before installation or startup.

## RunPod Serverless

Build and push the same Cog image, then override the container command:

```text
python -u /src/rp_handler.py
```

The handler accepts Cog-compatible reference URL lists and `reference_*_urls`
aliases. It rejects private/reserved network targets, caps each file at 512MiB, and
returns a bounded base64 media output compatible with app.nz's Cog serverless
shim. Set minimum workers to zero. A persistent network volume avoids
re-downloading the 53.92GB model on cold workers.

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

The pruned INT8 transformer is 20.97GB, INT8 Qwen encoder 27.14GB, visual VAE
5.21GB, and audio VAE 0.61GB, totaling about 53.92GB. They cannot all remain
resident on a 24GB or 48GB card, so ComfyUI stages components between prompt
encoding, denoising, and VAE decode. The 48GB path keeps normal DynamicVRAM
and is the practical single-card target; 24GB remains a correctness fallback.
Use at least 64GB system RAM (96GB is more comfortable) and a persistent weight
volume for the DynamicVRAM path. GPUs reporting at least 80GiB have enough room
for the weights plus activations and use HighVRAM after the first load.

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
prefixes, workflow graph, GPU/CPU encoder fallback, explicit license gate, and
resumable SHA-verified downloads. A real GPU smoke test additionally requires
the accepted model license, 54GB of weights, a 48GB CUDA GPU, and enough host
RAM.

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
- [MiniMax base-mode prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
