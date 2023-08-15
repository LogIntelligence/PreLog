#!/bin/bash

MODEL_PATH="../PreLog"
DATASET="BGL"

python train.py \
    --dataset $DATASET \
    --model-path $MODEL_PATH \
    --train-file anomaly_detection/data/$DATASET/train/1.json \
    --test-file anomaly_detection/data/$DATASET/test.json \
    --prompt-template prompt_template.txt \
    --verbalizer anomaly_detection/verbalizer.txt \
    --batch-size 16 \
    --lr 3e-5 \
    --max-step 2000 \
    --lr-scheduler-type polynomial \
    --max-length 1024 \
    --do-train \
    --do-eval