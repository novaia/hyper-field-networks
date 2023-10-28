FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 AS base-dependencies
RUN apt-get update -y \ 
    && apt-get install -y \
        # general dependencies
        wget=1.20.3-1ubuntu2 \
        # blender dependencies
        libfreetype6=2.10.1-2ubuntu0.3 \
        libglu1-mesa=9.0.1-1build1 \
        libxi6=2:1.7.10-0ubuntu1 \
        libxrender1=1:0.9.10-1 \
        xz-utils=5.2.4-1ubuntu1.1 \
        xvfb \
        libxkbcommon-x11-0=0.10.0-1 \
        # volume-rendering-jax dependencies
        libfmt-dev=6.1.2+ds-2 \
        # python3 and pip
        python3=3.8.2-0ubuntu2 -y \
        python3-pip=20.0.2-5ubuntu1.9 -y \
    && apt-get clean

# Upgrade to gcc-11 and g++-11, then install cmake.
RUN DEBIAN_FRONTEND=noninteractive apt-get install software-properties-common=0.99.9.12 -y \
    && DEBIAN_FRONTEND=noninteractive add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update -y \
    && apt-get install -y \
        gcc-11=11.4.0-2ubuntu1~20.04 \
        g++-11=11.4.0-2ubuntu1~20.04 \
    # Set gcc-11 as the default gcc version and g++-11 as the default g++ version.
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 \
    # Download kitware's signing key.
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
        | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
    # Install cmake from Kitware.
    && apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" \
    && apt-get update -y \
    && apt-get install -y cmake=3.27.7-0kitware1ubuntu20.04.1 \
    && apt-get clean

FROM base-dependencies AS blender-install
# Install Blender.
ARG BLENDER_PACKAGE_NAME="blender-3.6.3-linux-x64"
ARG BLENDER_PACKAGE_URL="https://download.blender.org/release/Blender3.6/blender-3.6.3-linux-x64.tar.xz"
ARG BLENDER_PATH="/usr/local/blender"
COPY ./data/blender_versions/ /tmp/
RUN if [ -f /tmp/$BLENDER_PACKAGE_NAME* ]; then \
        echo "Copying $BLENDER_PACKAGE_NAME from cache"; \
    else \
        echo "Downloading $BLENDER_PACKAGE_NAME from the internet"; \
        wget $BLENDER_PACKAGE_URL -O /tmp/${BLENDER_PACKAGE_NAME}.tar.xz; \
    fi
RUN tar -xJf /tmp/${BLENDER_PACKAGE_NAME}.tar.xz -C /tmp/ \
    && rm -f /tmp/${BLENDER_PACKAGE_NAME}.tar.xz \
    && mv /tmp/${BLENDER_PACKAGE_NAME} ${BLENDER_PATH}
ENV PATH="$PATH:$BLENDER_PATH"

FROM blender-install AS tiny-cuda-nn-build
COPY ./dependencies/ /project/dependencies/
# Build tiny-cuda-nn.
RUN cd /project/dependencies/tiny-cuda-nn \
    && cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    && cmake --build build --config RelWithDebInfo -j \
    # Symlink the static libary to /usr/lib/.
    && ln -s \
        /project/dependencies/tiny-cuda-nn/build/libtiny-cuda-nn.a \
        /usr/lib/libtiny-cuda-nn.a

FROM tiny-cuda-nn-build AS final
ARG JAX_PACKAGE_URL="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
COPY requirements.txt requirements.txt
RUN python3 -m pip install --upgrade pip \
    # Install Jax with GPU support.
    && python3 -m pip install \
        "jax[cuda11_cudnn86]" -f $JAX_PACKAGE_URL \
    && python3 -m pip install \
        -r requirements.txt \
    # Compile/install volume-rendering-jax and jax-tcnn.
    && python3 -m pip install \
        /project/dependencies/volume-rendering-jax \
        /project/dependencies/jax-tcnn

WORKDIR project
EXPOSE 7070
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]
