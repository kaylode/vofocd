# syntax = docker/dockerfile:experimental
#
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference:
#           https://docs.docker.com/develop/develop-images/build_enhancements/
ARG BASE_IMAGE=ubuntu:18.04

# Instal basic utilities
FROM ${BASE_IMAGE} as dev-base
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && apt-get upgrade && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    gcc \
    wget \
    libjpeg-dev \
    zip \
    unzip bzip2 ffmpeg libsm6 libxext6 \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda-installs
ARG PYTHON_VERSION=3.9
ARG CUDA_VERSION=11.3
ARG PYTORCH_VERSION=1.12.1
ARG CUDA_CHANNEL=nvidia
ARG INSTALL_CHANNEL=pytorch
ENV CONDA_OVERRIDE_CUDA=${CUDA_VERSION}
RUN curl -fsSL -v -o ~/mambaforge.sh -O https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh && \
    chmod +x ~/mambaforge.sh && \
    ~/mambaforge.sh -b -p /opt/mamba && \
    rm ~/mambaforge.sh && \
    /opt/mamba/bin/mamba install -c "${INSTALL_CHANNEL}" -c "${CUDA_CHANNEL}" -y \
    python=${PYTHON_VERSION} \
    pytorch=${PYTORCH_VERSION} torchvision "cudatoolkit=${CUDA_VERSION}" && \
    /opt/mamba/bin/mamba clean -ya

# Install repo
# COPY ./ /workspace/
RUN git clone https://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@github.com/kaylode/vocal-folds.git /workspace/vocal-folds/
WORKDIR /workspace/vocal-folds/
RUN /opt/mamba/bin/python -m pip install -r requirements.txt
RUN /opt/mamba/bin/python -m pip install -U timm
RUN chmod +x scripts/misc/*
# RUN scripts/misc/download_kvasir.sh
RUN scripts/misc/download_weights.sh
RUN scripts/misc/download_vocals.sh

FROM ${BASE_IMAGE} as official
SHELL ["/bin/bash", "-c"]
ENV TZ=Asia/Ho_Chi_Minh
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
WORKDIR /workspace
ARG PYTORCH_VERSION
LABEL com.nvidia.volumes.needed="nvidia_driver"
RUN --mount=type=cache,id=apt-final,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg libsm6 libxext6 \
    libjpeg-dev \
    tmux \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/*

COPY --from=conda-installs /opt/mamba /opt/mamba
# copy packages installed by pip
COPY --from=conda-installs /workspace /workspace

ENV PATH /opt/mamba/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PYTORCH_VERSION ${PYTORCH_VERSION}

WORKDIR /workspace