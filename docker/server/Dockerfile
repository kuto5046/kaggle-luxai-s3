FROM nvcr.io/nvidia/pytorch:24.11-py3
WORKDIR /kaggle

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

ARG UID
ARG USER

RUN apt update && apt install -y sudo zip

RUN useradd $USER -l -u $UID -G sudo -s /bin/bash -m
RUN echo 'Defaults visiblepw' >> /etc/sudoers
RUN echo "$USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER $USER
ENV PATH="/home/$USER/.local/bin:$PATH"
RUN python -m pip install -U pip
COPY requirements.txt /tmp
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade luxai-s3