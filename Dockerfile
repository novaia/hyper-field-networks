FROM python:latest

WORKDIR project

RUN apt update
RUN apt install blender -y
RUN python3 -m pip install jupyterlab

EXPOSE 7070

ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]