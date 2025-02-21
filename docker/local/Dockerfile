FROM nvcr.io/nvidia/pytorch:24.11-py3
WORKDIR /kaggle

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

RUN python -m pip install -U pip
COPY requirements.txt /tmp
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade luxai-s3