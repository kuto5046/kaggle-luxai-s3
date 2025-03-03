#!/bin/bash

agent1_path=$1
agent2_path=$2

uv run luxai-s3-fast $1 $2 --tournament --tournament-cfg-concurrent 4 --tournament-cfg-ranking-system wins