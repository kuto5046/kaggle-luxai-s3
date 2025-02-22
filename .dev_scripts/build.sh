#!/bin/bash

docker build --no-cache -f docker/server/Dockerfile \
    --build-arg UID=$(id -u) --build-arg USER=$USER \
    --network host --rm -t $USER/kaggle-luxai-s3 .