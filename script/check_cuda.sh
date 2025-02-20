#!/bin/bash

nvidia-smi
srun --gpus=1 /bin/bash -c "docker run -i --rm \
    --ipc=host \
    --env-file .env \
    -e CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES \
    -u \$(id -u):\$(id -g) \
    -v .:/kaggle \
    --gpus all \
    yuki.okumura/kaggle-luxai-s3-ir \
    /bin/bash -c 'echo \$CUDA_VISIBLE_DEVICES && python -c \"import torch; print(torch.cuda.is_available())\"'"