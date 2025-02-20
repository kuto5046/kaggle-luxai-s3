#!/bin/bash

line_length=140 
ruff format --line-length $line_length src/
isort --line-length $line_length src/ 