#!/bin/bash

cmd=$1
srun --gpus=0 /bin/bash -c "docker run -i --rm \
    --ipc=host \
    --env-file .env \
    -e CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES \
    -u \$(id -u):\$(id -g) \
    -v .:/kaggle \
    --cpus 128 \
    --gpus all \
    yuki.okumura/kaggle-luxai-s3 $cmd"