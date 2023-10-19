# Check if Blender is cached.
FROM alpine as cache_checker
ARG BLENDER_PACKAGE_NAME="blender-3.6.3-linux-x64"
COPY ./data/blender_versions/$BLENDER_PACKAGE_NAME* ./
RUN if [ -f $BLENDER_PACKAGE_NAME* ]; then \
        echo "BLENDER_CACHE_EXISTS=1" > /tmp/blender_cache_exists; \
    else \
        echo "BLENDER_CACHE_EXISTS=0" > /tmp/blender_cache_exists; \
    fi

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 as final

# Install Blender.
ARG BLENDER_PACKAGE_NAME="blender-3.6.3-linux-x64"
ARG BLENDER_PACKAGE_URL= \
    "https://download.blender.org/release/Blender3.6/blender-3.6.3-linux-x64.tar.xz"
ARG BLENDER_PATH="/usr/local/blender"
COPY --from=cache_checker /tmp/blender_cache_exists /tmp/blender_cache_exists
COPY ./data/blender_versions/$BLENDER_PACKAGE_NAME* ../tmp/
RUN if [ "$(cat /tmp/blender_cache_exists)" = "BLENDER_CACHE_EXISTS=1" ]; then \
        echo "Copying $BLENDER_PACKAGE_NAME from cache"; \
    else \
        echo "Downloading $BLENDER_PACKAGE_NAME from the internet"; \
        wget $BLENDER_PACKAGE_URL /tmp/; \
        cp $BLENDER_PACKAGE_NAME* /project/data/blender_versions/; \
    fi
RUN tar -xJf /tmp/${BLENDER_PACKAGE_NAME}.tar.xz -C /tmp/ \
    && rm -f /tmp/${BLENDER_PACKAGE_NAME}.tar.xz \
    && mv /tmp/${BLENDER_PACKAGE_NAME} ${BLENDER_PATH}
ENV PATH="$PATH:$BLENDER_PATH"

# Install Blender dependencies.
RUN apt update -y
RUN apt install -y \
    libfreetype6 \
    libglu1-mesa \
    libxi6 \
    libxrender1 \
    xz-utils \
    libxkbcommon-x11-0

# Install volume-render-jax dependencies.
RUN apt install -y libfmt-dev
# Upgrade to GCC 11 for volume-render-jax. compilation.
RUN DEBIAN_FRONTEND=noninteractive apt install software-properties-common -y
RUN DEBIAN_FRONTEND=noninteractive add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt update -y
RUN apt install -y gcc-11 g++-11
# Set gcc-11 as the default gcc version and g++-11 as the default g++ version.
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

RUN apt install python3 -y 
RUN apt install python3-pip -y 
RUN apt install xvfb -y
RUN python3 -m pip install "jax[cuda11_cudnn86]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN python3 -m pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

WORKDIR project

EXPOSE 7070

ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]
