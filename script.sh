#!/bin/bash

python3 main.py \
    --data-dir ./data \
    --columns FREQ \
    --window-length 500 \
    --stride 300 \
    --batch-size 32 \
    --epochs-per-run 100 \
    --head-learning-rate 1e-3 \
    --backbone-learning-rate 1e-5 \
    --seed 42 \
    --gpu 0 \
    --patch-size 10 \
    --d-model 2048 \
    --dropout 0.1 \
    --patch-stride 10 \
    --patience 30 \
    --experiment \
    --model gemma \
    --normalise window