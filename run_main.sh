#!/bin/bash

cmd=$1
docker run -it --rm \
    --ipc=host \
    --shm-size 64G \
    --env-file .env \
    -v .:/home/user/work/ \
    --gpus all \
    okumura/kaggle-luxai-s3-main $cmd