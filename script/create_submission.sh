#!/bin/bash

sub_name=$1
exp_dir=$2
out_file=$sub_name.tar.gz

cd /kaggle
mkdir ./submissions/$sub_name
cd $exp_dir
rsync -av --exclude='__pycache__' --exclude='dataset-metadata.json' . ../../../submissions/$sub_name/
cd ../../../submissions/$sub_name
cp -r /usr/local/lib/python3.12/dist-packages/lightning .
tar -czvf $out_file \
    --exclude='*/__pycache__'  \
    --exclude='__pycache__' \
    --exclude='dataset-metadata.json' *
mv $out_file ../