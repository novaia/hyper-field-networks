FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Install Blender dependencies.
RUN apt update -y
RUN apt install -y libfreetype6=2.10.1-2ubuntu0.3
RUN apt install -y libglu1-mesa=9.0.1-1build1
RUN apt install -y libxi6=2:1.7.10-0ubuntu1
RUN apt install -y libxrender1=1:0.9.10-1
RUN apt install -y xz-utils=5.2.4-1ubuntu1.1
RUN apt install -y xvfb=2:1.20.13-1ubuntu1~20.04.8
#RUN apt install -y \
#    libfreetype6=2.10.1-2ubuntu0.3 \
#    libglu1-mesa=9.0.1-1build1 \
#    libxi6=2:1.7.10-0ubuntu1 \
#    libxrender1=1:0.9.10-1 \
#    xz-utils=5.2.4-1ubuntu1.1 \
#    xvfb=2:1.20.13-1ubuntu1~20.04.8
#    #libxkbcommon-x11-0=0.10.0-1

# Install volume-render-jax dependencies.
RUN apt install -y libfmt-dev=6.1.2+ds-2
# Upgrade to GCC 11 for volume-render-jax. compilation.
RUN DEBIAN_FRONTEND=noninteractive apt install software-properties-common=0.99.9.12 -y
RUN DEBIAN_FRONTEND=noninteractive add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt update -y
RUN apt install -y gcc-11=11.4.0-2ubuntu1~20.04 g++-11=11.4.0-2ubuntu1~20.04
# Set gcc-11 as the default gcc version and g++-11 as the default g++ version.
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

RUN apt install python3=3.8.2-0ubuntu2 -y 
RUN apt install python3-pip=20.0.2-5ubuntu1.9 -y 
RUN python3 -m pip install "jax[cuda11_cudnn86]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN python3 -m pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

# Install volume-rendering-jax.
COPY ./dependencies/ /project/dependencies/
RUN python3 -m pip install /project/dependencies/volume-rendering-jax
#RUN rm /project/dependencies/volume-rendering-jax

# Install Blender.
ARG BLENDER_PACKAGE_NAME="blender-3.6.3-linux-x64"
ARG BLENDER_PACKAGE_URL= \
    "https://download.blender.org/release/Blender3.6/blender-3.6.3-linux-x64.tar.xz"
ARG BLENDER_PATH="/usr/local/blender"
COPY ./data/blender_versions/$BLENDER_PACKAGE_NAME* /tmp/
RUN if [ -f /tmp/$BLENDER_PACKAGE_NAME* ]; then \
        echo "Copying $BLENDER_PACKAGE_NAME from cache"; \
    else \
        echo "Downloading $BLENDER_PACKAGE_NAME from the internet"; \
        wget $BLENDER_PACKAGE_URL -O /tmp/$BLENDER_PACKAGE_NAME.tar.xz; \
        cp /tmp/$BLENDER_PACKAGE_NAME.tar.xz /project/data/blender_versions/; \
    fi
RUN tar -xJf /tmp/${BLENDER_PACKAGE_NAME}.tar.xz -C /tmp/ \
    && rm -f /tmp/${BLENDER_PACKAGE_NAME}.tar.xz \
    && mv /tmp/${BLENDER_PACKAGE_NAME} ${BLENDER_PATH}
ENV PATH="$PATH:$BLENDER_PATH"

WORKDIR project

EXPOSE 7070

ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]
