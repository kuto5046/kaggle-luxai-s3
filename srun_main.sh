#!/bin/bash

cmd=$1
srun --gpus=1 /bin/bash -c "docker run -i --rm \
    --ipc=host \
    --env-file .env \
    -e CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES \
    -u \$(id -u):\$(id -g) \
    -v .:/home/user/work/ \
    --cpus 32 \
    --gpus all \
    yuki.okumura/kaggle-luxai-s3-main $cmd"