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

The four INT8/REF2VA weights are not baked into this portable image. The
default FP32 visual VAE brings the selected set to about 59.13GB; mount at
least 62GB of persistent storage at `/weights` and provision the files listed
in the README before startup. The app links those files into ComfyUI and
consumes them as-is; missing files fail startup with their full paths.

Set `H3_VIDEO_VAE_PRECISION=fp16` to use the 5.21GB visual VAE instead. This
is useful on storage-constrained workers, but it intentionally gives up the
official FP32 decode path.

Set `H3_FP32_MATMUL_TF32=1` to enable cuBLAS TF32 while keeping the FP32 VAE
weights and activations. This can accelerate the VAE on tensor-core GPUs, but
it is numerically different from strict FP32 and should be A/B tested.

SageAttention is compiled for SM89 and SM120 by default, covering Ada GPUs
such as RTX 4090 and Blackwell consumer/workstation GPUs. Build with a
different `SAGE_CUDA_ARCH_LIST` for other supported targets. Runtime still
selects PyTorch SDPA on unsupported configurations, including the known SM90
H3 corruption path.

`GRADIO_PUBLIC_PORT` and `GRADIO_PUBLIC_PROTO` remain optional corrections for
proxies that omit the public port or protocol. Hostnames are inferred from each
request, so changing the public URL after a restart needs no image or config
change.
