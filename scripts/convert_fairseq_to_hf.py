#!/usr/bin/env python
# coding: utf-8

CHECKPOINT_PATH = "PreLog/checkpoint_best.pt"

import torch
from transformers import BartConfig, BartForConditionalGeneration, AutoTokenizer

config = BartConfig.from_pretrained("facebook/bart-base")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
gbart = BartForConditionalGeneration(config=config)
gbart.eval()

gbart_state_dict = gbart.state_dict()

model = torch.load(CHECKPOINT_PATH)

state_dict = model['model']

for k in gbart_state_dict.keys():
    if k[:5] == 'model':
        if k[6:] in state_dict.keys():
            gbart_state_dict[k] = state_dict[k[6:]]

gbart_state_dict['model.encoder.embed_tokens.weight'][50264] = gbart_state_dict['model.encoder.embed_tokens.weight'][51200]
gbart_state_dict['model.encoder.embed_tokens.weight'] = gbart_state_dict['model.encoder.embed_tokens.weight'][:50265, :]
gbart_state_dict['model.decoder.embed_tokens.weight'] = gbart_state_dict['model.encoder.embed_tokens.weight']
gbart_state_dict['model.shared.weight'] = gbart_state_dict['model.encoder.embed_tokens.weight']
gbart_state_dict['lm_head.weight'] = state_dict['decoder.output_projection.weight']
gbart_state_dict['lm_head.weight'][50264] = gbart_state_dict['lm_head.weight'][51200]
gbart_state_dict['lm_head.weight'] = gbart_state_dict['lm_head.weight'][:50265, :]


gbart.load_state_dict(gbart_state_dict)
gbart.save_pretrained("PreLog_hf")
tokenizer.save_pretrained("PreLog_hf")