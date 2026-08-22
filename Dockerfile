# Portable single-GPU MiniMax H3 Gradio image. This reproduces the validated
# CUDA 13 deployment directly from source instead of depending on an older
# CUDA 12.8 release image.
ARG PYTHON_IMAGE=python:3.12.3-slim-bookworm
FROM ${PYTHON_IMAGE}

ARG COMFYUI_VERSION=v0.31.0
ARG COMFYUI_REPO=https://github.com/hao3039032/ComfyUI.git
ARG COMFYUI_COMMIT=d648863a30fe8122ba609cfb7319e2b773809575
ARG SOL_ATTN_COMMIT=dfc2e31a41afd72bd53dd2137fc8b2931d5ec192
ARG SAGEATTENTION_COMMIT=eb615cf6cf4d221338033340ee2de1c37fbdba4a
ARG SAGE_CUDA_ARCH_LIST=8.9;12.0
ARG MAX_JOBS=4

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates ffmpeg git libgl1 libglib2.0-0 tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# These are the wheel URLs and hashes used by the working CUDA 13 deployment.
# torchaudio's CUDA implementation comes through the torch CUDA dependencies.
RUN python -m pip install \
    'torch@https://download-r2.pytorch.org/whl/cu130/torch-2.12.1%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=4bafc356fbb622e2756179406825c3a56c17b401196435a1487c5b40c657706c' \
    'torchvision@https://download-r2.pytorch.org/whl/cu130/torchvision-0.27.1%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=abbfc724597c16da177002a16979aa8c44c4898c97bcb731b647cc57507f5772' \
    'torchaudio@https://files.pythonhosted.org/packages/88/d8/d6d0f896e064aa67377484efef4911cdcc07bce2929474e1417cc0af18c2/torchaudio-2.11.0-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=6503c0bdb29daf2e6281bb70ea2dfe2c3553b782b619eb5d73bdadd8a3f7cecf'

RUN git clone "${COMFYUI_REPO}" /opt/ComfyUI \
    && git -C /opt/ComfyUI checkout "${COMFYUI_COMMIT}"

RUN git clone https://github.com/kijai/ComfyUI-SolAttn_triton.git \
        /opt/ComfyUI/custom_nodes/ComfyUI-SolAttn_triton \
    && git -C /opt/ComfyUI/custom_nodes/ComfyUI-SolAttn_triton \
        checkout "${SOL_ATTN_COMMIT}"

COPY docker/constraints.txt /tmp/constraints.txt
RUN python -m pip install -c /tmp/constraints.txt \
        -r /opt/ComfyUI/requirements.txt \
        'gradio==6.24.0'

# Torch's runtime toolkit omits the compiler. These versions match its CUDA
# 13.0.88 components and permit a GPU-free SageAttention build.
RUN python -m pip install \
    'ninja==1.13.0' \
        'nvidia-cuda-nvcc==13.0.88' \
    'nvidia-cuda-crt==13.0.88' \
    'nvidia-cuda-cccl==13.0.85' \
    'nvidia-nvvm==13.0.88'

# The CUDA wheel components install a complete nvcc toolchain under nvidia/cu13.
# SageAttention is built for the deployment target without requiring a GPU.
ENV CUDA_HOME=/usr/local/lib/python3.12/site-packages/nvidia/cu13
RUN export PATH="${CUDA_HOME}/bin:${PATH}" \
       TORCH_CUDA_ARCH_LIST="${SAGE_CUDA_ARCH_LIST}" \
       MAX_JOBS="${MAX_JOBS}" \
    && ln -sfn libcudart.so.13 "${CUDA_HOME}/lib/libcudart.so" \
        && python -m pip install --no-build-isolation --no-deps \
        "git+https://github.com/thu-ml/SageAttention.git@${SAGEATTENTION_COMMIT}"

COPY app.py h3_gradio.py h3_media.py h3_prompt.py h3_runtime.py \
    h3_tuning.py h3_workflow.py weights.py /src/
COPY scripts/container/start_gradio.sh /src/start_gradio.sh
RUN chmod 755 /src/start_gradio.sh \
    && mkdir -p /weights /opt/ComfyUI/input /opt/ComfyUI/output /tmp/gradio

ENV COMFY_ROOT=/opt/ComfyUI \
    WEIGHTS_DIR=/weights \
    H3_VIDEO_VAE_PRECISION=fp32 \
    PORT=7860

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c 'import os, urllib.request; urllib.request.urlopen("http://127.0.0.1:" + os.getenv("PORT", "7860") + "/config", timeout=3).read()'

ENTRYPOINT ["/usr/bin/tini", "--", "/src/start_gradio.sh"]
