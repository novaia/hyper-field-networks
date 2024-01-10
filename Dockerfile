FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
ARG NO_RECS="--no-install-recommends"
ARG BLENDER_PACKAGE_NAME="blender-3.6.3-linux-x64"
ARG BLENDER_PATH="/usr/local/blender"
ENV PATH="$PATH:$BLENDER_PATH"
COPY ./binaries/ /tmp/

RUN apt-get update -y \ 
    && apt-get install $NO_RECS -y \
        # general dependencies
        wget \
        # blender dependencies
        libfreetype6 \
        libglu1-mesa \
        libxi6 \
        libxrender1 \
        xz-utils \
        xvfb \
        libxkbcommon-x11-0 \
        # volume-rendering-jax dependencies
        libfmt-dev \
        # python3 and pip
        python3-dev \
        python3-pip \
    # Upgrade to gcc-11 and g++-11, then install cmake.
    && DEBIAN_FRONTEND=noninteractive apt-get install $NO_RECS -y software-properties-common \
    && DEBIAN_FRONTEND=noninteractive add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update -y \
    && apt-get install $NO_RECS -y gcc-11 g++-11 \
    # Set gcc-11 as the default gcc version and g++-11 as the default g++ version.
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 \
    # Download kitware's signing key.
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
        | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
    # Install cmake from kitware.
    && apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" \
    && apt-get update -y \
    && apt-get install $NO_RECS -y cmake \
    # Cleanup lists.
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    # Install blender.
    && tar -xJf /tmp/${BLENDER_PACKAGE_NAME}.tar.xz -C /tmp/ \
    && rm -f /tmp/${BLENDER_PACKAGE_NAME}.tar.xz \
    && mv /tmp/${BLENDER_PACKAGE_NAME} ${BLENDER_PATH}


# Install python packages.
ARG NO_CACHE="--no-cache-dir"

ARG JAX_PACKAGE_URL="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
RUN python3 -m pip install $NO_CACHE "jax[cuda11_cudnn86]" -f $JAX_PACKAGE_URL

ARG DALI_PACKAGE_URL="https://developer.download.nvidia.com/compute/redist "
RUN python3 -m pip install $NO_CACHE \
    --extra-index-url $DALI_PACKAGE_URL \
    --upgrade nvidia-dali-cuda120

COPY requirements.txt requirements.txt
RUN python3 -m pip install $NO_CACHE --upgrade pip \
    && python3 -m pip install $NO_CACHE -r requirements.txt
        
WORKDIR project
