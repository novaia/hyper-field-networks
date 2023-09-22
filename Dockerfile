FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

RUN apt update -y
RUN apt install -y \
    curl \
    libfreetype6 \
    libglu1-mesa \
    libxi6 \
    libxrender1 \
    xz-utils \
    libxkbcommon-x11-0

ARG blender_package_name="blender-3.6.3-linux-x64"
ARG blender_package_url="https://download.blender.org/release/Blender3.6/blender-3.6.3-linux-x64.tar.xz"
ARG blender_path="/usr/local/blender"

WORKDIR tmp
RUN curl -OL $blender_package_url
WORKDIR ..
RUN tar -xJf /tmp/${blender_package_name}.tar.xz -C /tmp \
    && rm -f /tmp/${blender_package_name}.tar.xz \
    && mv /tmp/${blender_package_name} ${blender_path}

ENV PATH="$PATH:$blender_path"

RUN apt install python3 -y 
RUN apt install python3-pip -y 
RUN apt install xvfb -y
RUN python3 -m pip install jupyterlab

WORKDIR project

EXPOSE 7070

ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]