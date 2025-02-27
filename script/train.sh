#!/bin/bash 

exp_id=$1
cd /kaggle
python exp/$exp_id/data_processor.py
python exp/$exp_id/train.py