#!/bin/bash

submission_id=$1
du -h data/$submission_id/episodes/$submission_id/*  --max-depth=1 | sort -h -r
wc -l data/$submission_id/episodes.csv
du -h data/$submission_id/episodes/$submission_id/*  --max-depth=1 | wc -l
