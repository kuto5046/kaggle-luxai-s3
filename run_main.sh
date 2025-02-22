#!/bin/bash

cmd=$1
docker run -i --rm \
    --ipc=host \
    --shm-size 64G \
    --env-file .env \
    -v .:/kaggle \
    --gpus all \
    okumura/kaggle-luxai-s3-main $cmd