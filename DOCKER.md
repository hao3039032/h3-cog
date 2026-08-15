# Docker Deployment

This image reproduces the validated single-GPU CUDA 13 stack from source:
Python 3.12, PyTorch 2.12.1+cu130, ComfyUI v0.31.0
(`43cb4fff...`), and SageAttention `eb615cf...`. It intentionally does not
install Raylight: the first portable target is the native single-GPU path used
by the working deployment.

The CUDA user-space runtime is bundled in the image. A host therefore needs an
NVIDIA driver that supports CUDA 13 (typically version 580 or newer), but it
does not need a host CUDA toolkit. If `nvidia-smi` reports `CUDA Version:
12.8` or lower, use a host driver image that is new enough or build a separate
CUDA 12.8 variant of this container.

The container listens on `7860` by default. Port `6006` is not built into the
image; set both values on AutoDL, where the platform expects `6006`:

```sh
PORT=6006 H3_PORT=6006 docker compose up -d --build
```

On a normal platform, publish any host port to the same container port:

```sh
docker run --gpus all --shm-size 8gb --restart unless-stopped \
  -p 7860:7860 \
  -e MINIMAX_H3_LICENSE_ACCEPTED=1 \
  -v h3-weights:/weights \
  h3-gradio:local
```

For a host port other than 7860, change only the left side of `-p`, for example
`-p 8080:7860`. Use `PORT` only when the platform requires the process itself
to listen on a different socket.

The four INT8/REF2VA weights are not baked into this portable image. Mount at
least 56GB of persistent storage at `/weights`; the app performs resumable
ModelScope downloads, SHA-256 verification, and linking into ComfyUI. Set
`H3_BAKED_WEIGHTS_VERIFIED=1` only for a separate image whose verified weights
are baked into immutable layers.

SageAttention is compiled for SM89 and SM120 by default, covering Ada GPUs
such as RTX 4090 and Blackwell consumer/workstation GPUs. Build with a
different `SAGE_CUDA_ARCH_LIST` for other supported targets. Runtime still
selects PyTorch SDPA on unsupported configurations, including the known SM90
H3 corruption path.

`GRADIO_PUBLIC_PORT` and `GRADIO_PUBLIC_PROTO` remain optional corrections for
proxies that omit the public port or protocol. Hostnames are inferred from each
request, so changing the public URL after a restart needs no image or config
change.
