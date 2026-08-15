# ModelScope / single-GPU deployment

The runtime has a native single-GPU path and an explicit opt-in Raylight path:

| Visible GPUs | Backend | Memory policy |
| --- | --- | --- |
| 1× 84GB-class card | native ComfyUI | HighVRAM keeps loaded weights resident |
| 1× 48GB card | native ComfyUI | normal DynamicVRAM with staged component loading |
| 1× RTX 4090 24GB | native ComfyUI | automatically enables low-VRAM mode for correctness |
| 2+ cards | Raylight opt-in | FSDP2 shards transformer weights; Ulysses2 shards the token sequence |

`H3_PARALLEL_MODE=single` is the default. Set `raylight` to require exactly the
distributed path and fail early if fewer than two GPUs or the Raylight nodes
are available. Set `auto` only when two-GPU Raylight selection is wanted.

The single-card path is functional, but a 48GB card cannot keep the 20.97GB
transformer, 27.14GB INT8 text encoder, and both VAEs resident simultaneously.
It therefore needs at least 64GB of system RAM (96GB is recommended) and will
stage components during generation. Both modes default to 480p (`preview`),
MP4/H.264, and one generation at a time.

Set `H3_LOWVRAM=0` only to experiment with normal DynamicVRAM on a 24GB card;
the safe automatic default is low-VRAM mode. GPUs with at least 28GB keep the
existing normal DynamicVRAM policy unless `H3_LOWVRAM=1` is explicitly set.
GPUs reporting at least 80GiB use ComfyUI's HighVRAM mode so the INT8 weights can
remain on the card after their first load. Set `H3_HIGHVRAM=0` to compare the
DynamicVRAM path; `H3_LOWVRAM=1` still takes precedence as the emergency mode.
The Studio image rebuilds SageAttention for SM80, SM89, and SM120. SM120 needs
CUDA toolkit 12.8 or newer. On an SM120 RTX 6000 D / RTX PRO 6000-class card,
compare SageAttention against PyTorch SDPA with the same seed before promoting
accelerated output.

## Gradio Studio entry point

The ModelScope Studio entry file is `app.py`. It lazily starts ComfyUI on the
first request so the web page can become available before the 59.13GB weight set
has been checked or downloaded. Configure these environment variables:

```text
MINIMAX_H3_LICENSE_ACCEPTED=1
H3_PARALLEL_MODE=single
H3_VIDEO_VAE_PRECISION=fp32
WEIGHTS_DIR=/persistent-volume/models
PORT=7860
```

The deployment image must contain CUDA 12.8, PyTorch, ComfyUI v0.31.0 at the
commit in `cog.yaml`, and the pinned Raylight custom node. A plain Gradio SDK
image is not
enough. `Dockerfile.modelscope` derives from the already baked REF2VA image so
the four weights are not assembled again:

```sh
docker build -f Dockerfile.modelscope -t ghcr.io/OWNER/minimax-h3:modelscope-single .
docker push ghcr.io/OWNER/minimax-h3:modelscope-single
```

Publish it to a registry ModelScope can pull, expose port 7860, and give the
container all selected GPUs plus adequate shared memory. The separate
`cog.yaml` remains available when a Cog/Replicate-compatible image is needed.

For scale-to-zero, bake the four verified REF2VA files into the image and set
`H3_BAKED_WEIGHTS_VERIFIED=1`; otherwise preserve `WEIGHTS_DIR` on a persistent
volume. Without either option, every cold worker must download roughly 59.13GB.
Set `H3_VIDEO_VAE_PRECISION=fp16` only for a storage-constrained comparison
worker. The runtime downloads the deterministic Comfy-native FP32 artifact from
`Austusm/minimax_h3_video_vae` (`10,415,548,688` bytes, SHA-256
`a28fa965eb65a3fe1279a8bf73f01dddaa36ecd039d08751f74bc8849e88767b`). The
repack preserves every official F32 tensor byte and appends the two `[24]`
latent normalization buffers expected by ComfyUI.

## AutoDL startup

AutoDL containers use the platform's built-in supervisor rather than systemd.
Copy the startup files before restarting the instance:

```sh
scp scripts/autodl/start_h3.sh root@HOST:/root/start_h3.sh
scp scripts/autodl/autodl.sh root@HOST:/etc/autodl.sh
ssh root@HOST 'chmod 755 /root/start_h3.sh /etc/autodl.sh'
```

The platform invokes `/etc/autodl.sh` through its `customer-cmd` supervisor
program. The wrapper runs the Gradio service on port 6006 and forwards stops to
the ComfyUI/Ray process group so lazy-started workers cannot outlive a service
restart.

Some reverse proxies publish the service on a non-standard HTTPS port but omit
that port from `X-Forwarded-Host`. Set `GRADIO_PUBLIC_PORT` to that external
port, for example `8443` on AutoDL. If the proxy also omits
`X-Forwarded-Proto` or reports the internal protocol, set
`GRADIO_PUBLIC_PROTO=https`; Gradio will then construct file URLs with the
current forwarded host instead of a startup-specific domain. Leave both unset
for normal proxies. A full
`GRADIO_ROOT_PATH=https://host:port` override still takes precedence for
deployments with a fixed public origin.

For a bare-metal SM120 host with CUDA toolkit 12.8 or newer, build only the
local architecture:

```sh
TORCH_CUDA_ARCH_LIST='12.0' python -m pip install --no-build-isolation \
  --force-reinstall --no-deps \
  git+https://github.com/thu-ml/SageAttention.git@eb615cf6cf4d221338033340ee2de1c37fbdba4a
```

The startup log should say `attention=sageattention`; if the extension is
absent, the runtime safely falls back to `pytorch-sdpa`.
