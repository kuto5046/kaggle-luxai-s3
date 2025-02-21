#!/bin/bash

export JAX_TRACEBACK_FILTERING=off 
cd /kaggle
luxai-s3 ./exp/okumura/exp001/main.py agents/exp017/main.py --tournament