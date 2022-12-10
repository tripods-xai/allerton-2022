FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN apt-get update && \
    apt-get install -y vim && \
    apt-get install -y sudo
RUN adduser --disabled-password --gecos '' docker
RUN adduser docker sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER docker

COPY docker_requirements.txt /opt/app/requirements.txt
RUN sudo pip install --upgrade pip && \ 
    sudo pip uninstall -y setuptools && sudo pip install setuptools && \
    sudo pip install -r /opt/app/requirements.txt
# ENV TZ America/Chicago
