#!/bin/bash

submission_id=$1
mkdir kaggle_datasets/$submission_id
cd data/$submission_id
zip -r /kaggle/kaggle_datasets/$submission_id/$submission_id.zip *