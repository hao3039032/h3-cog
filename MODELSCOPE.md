# ModelScope / single-GPU deployment

The runtime has a native single-GPU path and an explicit opt-in Raylight path:

| Visible GPUs | Backend | Memory policy |
| --- | --- | --- |
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

## Gradio Studio entry point

The ModelScope Studio entry file is `app.py`. It lazily starts ComfyUI on the
first request so the web page can become available before the 53.92GB weight set
has been checked or downloaded. Configure these environment variables:

```text
MINIMAX_H3_LICENSE_ACCEPTED=1
H3_PARALLEL_MODE=single
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
volume. Without either option, every cold worker must download roughly 53.92GB.

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
