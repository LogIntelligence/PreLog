#!/bin/bash

MODEL_PATH="../PreLog"
DATASET="OpenStack"

python train.py \
    --dataset $DATASET \
    --model-path $MODEL_PATH \
    --train-file failure_identification/data/OpenStack/train.json \
    --test-file failure_identification/data/OpenStack/test.json \
    --prompt-template prompt_template.txt \
    --verbalizer failure_identification/verbalizer.txt \
    --batch-size 16 \
    --lr 3e-5 \
    --max-step 2000 \
    --lr-scheduler-type polynomial \
    --max-length 1024 \
    --do-train \
    --do-eval