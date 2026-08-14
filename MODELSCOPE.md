# ModelScope / RTX 4090 deployment

This branch has one runtime and two automatically selected execution paths:

| Visible GPUs | Backend | Memory policy |
| --- | --- | --- |
| 1× RTX 4090 24GB | native ComfyUI | automatically enables the low-VRAM DynamicVRAM policy and swaps components as needed |
| 2× RTX 4090 24GB | Raylight | FSDP2 shards transformer weights; Ulysses2 shards the video/audio token sequence |

`H3_PARALLEL_MODE=auto` is the default. Set it to `single` to force the native
path on a multi-GPU machine, or `raylight` to require exactly the distributed
path and fail early if fewer than two GPUs or the Raylight nodes are available.

The single-card path is functional, but it cannot keep the 20.97GB transformer,
15.69GB text encoder, and both VAEs resident simultaneously. It therefore needs
at least 64GB of system RAM and will be considerably slower than the two-card
path. Both modes default to 480p (`preview`), MP4/H.264, and one generation at a
time.

Set `H3_LOWVRAM=0` only to experiment with normal DynamicVRAM on a 24GB card;
the safe automatic default is low-VRAM mode. GPUs with at least 28GB keep the
existing normal DynamicVRAM policy unless `H3_LOWVRAM=1` is explicitly set.

## Gradio Studio entry point

The ModelScope Studio entry file is `app.py`. It lazily starts ComfyUI on the
first request so the web page can become available before the 42.5GB weight set
has been checked or downloaded. Configure these environment variables:

```text
MINIMAX_H3_LICENSE_ACCEPTED=1
H3_PARALLEL_MODE=auto
WEIGHTS_DIR=/persistent-volume/models
PORT=7860
```

The deployment image must contain CUDA 12.8, PyTorch, ComfyUI at the commit in
`cog.yaml`, and the pinned Raylight custom node. A plain Gradio SDK image is not
enough. `Dockerfile.modelscope` derives from the already baked REF2VA image so
the four weights are not assembled again:

```sh
docker build -f Dockerfile.modelscope -t ghcr.io/OWNER/minimax-h3:modelscope-4090 .
docker push ghcr.io/OWNER/minimax-h3:modelscope-4090
```

Publish it to a registry ModelScope can pull, expose port 7860, and give the
container all selected GPUs plus adequate shared memory. The separate
`cog.yaml` remains available when a Cog/Replicate-compatible image is needed.

For scale-to-zero, bake the four verified REF2VA files into the image and set
`H3_BAKED_WEIGHTS_VERIFIED=1`; otherwise preserve `WEIGHTS_DIR` on a persistent
volume. Without either option, every cold worker must download roughly 42.5GB.
