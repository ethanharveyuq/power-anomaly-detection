#!/bin/bash

python3 main.py \
    --data-dir ./data \
    --window-length 500 \
    --stride 250 \
    --columns FREQ \
    --batch-size 32 \
    --epochs 20 \
    --learning-rate 1e-4 \
    --seed 42 \
    --gpu 0 \
    --patch-size 10 \
    --d-model 768 \
    --dropout 0.1