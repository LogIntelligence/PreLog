#!/bin/bash

DATA_DIR="data/logs"
SAVE_DIR=prelog_pretrain

MAX_UPDATE=100000
WARMUP_UPDATES=10000
MAX_SENTENCES=8
MAX_TOKENS=2048
UPDATE_FREQ=1


fairseq-train $DATA_DIR \
    --dataset-impl 'mmap' \
    --arch bart_base \
    --max-tokens $MAX_TOKENS \
    --max-sentences $MAX_SENTENCES \
    --update-freq $UPDATE_FREQ \
    --layernorm-embedding \
    --train-subset train \
    --valid-subset valid \
    --required-batch-size-multiple 8 \
    --insert 0.1 \
    --permute-sentences 0 \
    --poisson-lambda 3.5 \
    --mask 0.3 \
    --mask-length "span-poisson" \
    --replace-length 0 \
    --rotate 0 \
    --mask-random 0.1 \
    --task prelog \
    --negative-sampling \
    --contrastive-weight 0.1 \
    --sequence-insert-delete 0.1 \
    --sequence-disorder 0.5 \
    --criterion prelog_contrastive \
    --contrastive-weight 0.1 \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --relu-dropout 0.0 \
    --weight-decay 0.01 \
    --optimizer adam \
    --adam-eps 1e-06 \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 1e-4 \
    --lr-scheduler polynomial_decay \
    --warmup-updates $WARMUP_UPDATES \
    --total-num-update $MAX_UPDATE \
    --max-update $MAX_UPDATE \
    --no-epoch-checkpoints \
    --save-dir $SAVE_DIR \
    --entry-level \
    --sequence-level \
    --skip-invalid-size-inputs-valid-test \
    --ddp-backend=legacy_ddp \
    --log-format tqdm \
    --log-interval 1 \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --checkpoint-activations