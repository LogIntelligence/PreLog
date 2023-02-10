#!/bin/bash

DATA_DIR="data/logs"

for SPLIT in train valid; do \
    python multiprocessing_bpe_encoder.py \
            --encoder-json models/gpt2_bpe/encoder.json \
            --vocab-bpe models/gpt2_bpe/vocab.bpe \
            --inputs $DATA_DIR/${SPLIT}_sequence_data.txt \
            --outputs $DATA_DIR/${SPLIT}_sequence_data.bpe \
            --workers 4;
done

fairseq-preprocess \
    --only-source \
    --srcdict models/checkpoints/dict.txt \
    --trainpref $DATA_DIR/train_sequence_data.bpe \
    --validpref $DATA_DIR/valid_sequence_data.bpe \
    --destdir $DATA_DIR \
    --workers 4
