#!/bin/bash

for SPLIT in train valid; do \
    python multiprocessing_bpe_encoder.py \
            --encoder-json models/gpt2_bpe/encoder.json \
            --vocab-bpe models/gpt2_bpe/vocab.bpe \
            --inputs data/logs/${SPLIT}_sequence_data.txt \
            --outputs data/logs/${SPLIT}_sequence_data.txt \
            --workers 30;
done

fairseq-preprocess \
    --only-source \
    --srcdict models/checkpoints/dict.txt \
    --trainpref data/logs/train_sequence_data.bpe \
    --validpref data/logs/valid_sequence_data.bpe \
    --destdir data/se \
    --workers 30
