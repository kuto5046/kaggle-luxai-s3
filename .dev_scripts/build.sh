#!/bin/bash

docker build -f Dockerfile \
    --build-arg UID=$(id -u) --build-arg USER=$USER \
    --network host --rm -t $USER/kaggle-luxai-s3-ir .