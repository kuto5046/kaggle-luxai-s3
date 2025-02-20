# pytorch versionに注意
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# 時間設定
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

# install basic dependencies
RUN apt-get -y update && apt-get install -y \
    sudo \
    wget \
    cmake \
    vim \
    git \
    tmux \
    zip \
    unzip \
    gcc \
    g++ \
    build-essential \
    ca-certificates \
    software-properties-common \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libpng-dev \
    libfreetype6-dev \
    libgl1-mesa-dev \
    libsndfile1 \
    libffi-dev \
    libsqlite3-dev \
    libreadline6-dev \
    libbz2-dev \
    libssl-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    libffi-dev \
    libdb-dev \
    libexpat1-dev \
    zlib1g-dev \
    liblzma-dev \
    libgdbm-dev \
    libmpdec-dev \
    zsh \
    xonsh \
    nodejs \
    npm \
    curl \
    fd-find \
    python3.10-venv \
    htop

# node js を最新Verにする
RUN npm -y install n -g && \
    n stable && \
    apt purge -y nodejs npm

ARG UID
ARG USER

RUN useradd $USER -l -u $UID -G sudo -s /bin/bash -m
RUN echo 'Defaults visiblepw' >> /etc/sudoers
RUN echo "$USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER $USER
WORKDIR /kaggle

# install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/home/$USER/.local/bin:$PATH 
COPY uv.lock .
COPY pyproject.toml .
RUN uv sync
