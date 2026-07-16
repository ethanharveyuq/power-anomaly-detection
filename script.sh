#!/bin/bash

python3 main.py \
    --data-dir ./data \
    --columns FREQ UA:MAG UA:ANG \
    --window-length 600 \
    --stride 480 \
    --batch-size 32 \
    --epochs-per-run 50 \
    --head-learning-rate 1e-3 \
    --backbone-learning-rate 1e-5 \
    --seed 42 \
    --gpu 0 \
    --patch-size 10 \
    --d-model 768 \
    --dropout 0.1 \
    --patch-stride 10 \
    --patience 30 \
    --resume \
    --experiment