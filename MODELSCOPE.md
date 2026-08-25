# ModelScope / native ComfyUI deployment

The runtime uses one native ComfyUI process and the first CUDA device:

| Visible GPUs | Backend | Memory policy |
| --- | --- | --- |
| 1× 84GB-class card | native ComfyUI | HighVRAM keeps loaded weights resident |
| 1× 48GB card | native ComfyUI | normal DynamicVRAM with staged component loading |
| 1× RTX 4090 24GB | native ComfyUI | normal DynamicVRAM with staged component loading |

`H3_PARALLEL_MODE` is no longer needed. Retired values
`raylight/auto/fsdp/dual` fail startup with an explicit removal error; unset the
variable or use `single` for compatibility.

The single-card path is functional, but a 48GB card cannot keep either 20.97GB
DiT, the 27.14GB INT8 text encoder, and both VAEs resident simultaneously. It
therefore needs at least 96GB of system RAM and will stage components during
generation. All tasks default to 480p (`preview`), MP4/H.264, and one
generation at a time.

`H3_LOWVRAM` is retired and ignored. GPUs reporting at least 80GiB use
ComfyUI's HighVRAM mode so both INT8 DiTs can potentially remain warm; set
`H3_HIGHVRAM=0` to compare the DynamicVRAM path. Task switches use
`H3_DIT_SWITCH_POLICY=auto` by default and do not call ComfyUI `/free`. Use
`evict` only for memory-pressure diagnostics.
The Studio image rebuilds SageAttention for SM80, SM89, and SM120. SM120 needs
CUDA toolkit 12.8 or newer. On an SM120 RTX 6000 D / RTX PRO 6000-class card,
compare SageAttention against PyTorch SDPA with the same seed before promoting
accelerated output.

## Gradio Studio entry point

The ModelScope Studio entry file is `app.py`. It lazily starts ComfyUI on the
first request. The 84.02GB weight set must already be provisioned at
`WEIGHTS_DIR`; missing files fail that initialization with their full paths.
Configure these environment variables:

```text
MINIMAX_H3_LICENSE_ACCEPTED=1
H3_VIDEO_VAE_PRECISION=fp32
H3_DIT_SWITCH_POLICY=auto
WEIGHTS_DIR=/persistent-volume/models
PORT=7860
```

The deployment image must contain CUDA 12.8, PyTorch, ComfyUI v0.31.0 at the
commit in `cog.yaml`. A plain Gradio SDK image is not enough.
`Dockerfile.modelscope` derives from the already baked H3 image. If that base
still contains only the REF2VA DiT, provision the FL2VA file on the same
persistent `WEIGHTS_DIR` before startup:

```sh
docker build -f Dockerfile.modelscope -t ghcr.io/OWNER/minimax-h3:modelscope-single .
docker push ghcr.io/OWNER/minimax-h3:modelscope-single
```

Publish it to a registry ModelScope can pull, expose port 7860, and give the
container adequate shared memory. The separate
`cog.yaml` remains available when a Cog/Replicate-compatible image is needed.

For scale-to-zero, preserve `WEIGHTS_DIR` on a persistent volume. Otherwise
every cold worker must be provisioned with roughly 84.02GB before it can start.
Set `H3_VIDEO_VAE_PRECISION=fp16` only for a storage-constrained comparison
worker. Provision the deterministic Comfy-native FP32 artifact from
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
the ComfyUI process group so a lazy-started worker cannot outlive a service
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
