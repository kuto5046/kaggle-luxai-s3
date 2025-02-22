#! /bin/bash

docker build --no-cache -f Dockerfile --build-arg DOCKER_UID=$(id -u) --network host --rm -t $USER/kaggle-luxai-s3-main .