FROM ubuntu:latest

RUN apt update -y
RUN apt install python3 python3-pip blender xvfb -y
RUN python3 -m pip install jupyterlab

WORKDIR project

EXPOSE 7070

ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]