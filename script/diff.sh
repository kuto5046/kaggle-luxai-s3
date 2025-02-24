#!/bin/bash
exp1=$1
exp2=$2

diff -ruN --exclude="__pycache__" exp/$exp1 exp/$exp2