import argparse
import pandas as pd
import os
import nltk
from accelerate import Accelerator

from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Model,
    # AutoModelFor,
)
from datasets import load_dataset
import evaluate
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
import datasets
from collections import Counter
from data_loader import parsing_v1, map_template_v3, preprocess, generate_template
from datasets import disable_caching
import logging
from logging import getLogger
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

accelerator = Accelerator()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = getLogger(__name__)

logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
logger.info(accelerator.state)
device = accelerator.device

disable_caching()

metric = evaluate.load("rouge")


# device = 'cuda' if torch.cuda.is_available() else 'cpu'


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # print(decoded_preds)
    # print(labels)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(
        decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(
        pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result


def add_parameter_token(tokenizer, model):
    parameter_token = "<*>"
    tokenizer.add_tokens(parameter_token)
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model, tokenizer.convert_tokens_to_ids(parameter_token)


def assign_embedding_for_parameter_token(tokenizer, model, variable_list):
    parameter_token = "<*>"
    # model.resize_token_embeddings(len(tokenizer))
    parameter_id = tokenizer.convert_tokens_to_ids(parameter_token)
    model.shared.weight.data[-1] = model.shared.weight.data[1]
    parameter_embs = []
    for v in variable_list:
        inp = tokenizer(v, return_tensors='pt')
        out = model.encoder(**inp)
        parameter_embs.append(out['last_hidden_state'][0][-1].data)
    e_para = parameter_embs[0]
    for e in parameter_embs[1:]:
        e_para += e
    model.shared.weight.data[parameter_id] = e_para / len(parameter_embs)
    return model


if __name__ == '__main__':

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    parser = argparse.ArgumentParser(
        description="Simple example of training script.")
    parser.add_argument("--dataset", type=str, default="HDFS",
                        help="Dataset")
    parser.add_argument("--model-path", type=str, default="last_model",
                        help="Path to the trained model")
    parser.add_argument("--train-file", type=str, default='./data/HDFS/train.json',
                        help="Path to the train file")
    parser.add_argument("--test-file", type=str, default='./data/HDFS/test.json',
                        help="Path to the test file")
    parser.add_argument("--outdir", type=str, default='PreLog',
                        help="Path to the test file")
    parser.add_argument(
        "--fp16",
        default=False,
        action='store_true',
        help="Whether to use mixed precision. Choose"
             "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
             "and an Nvidia Ampere GPU.",
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                              use_fast=False,
                                              add_prefix_space=True,
                                              do_lower_case=False)
    tokenizer.model_max_length = 256
    model = BartForConditionalGeneration.from_pretrained(args.model_path)
    tokenizer, model, p_token_ids = add_parameter_token(tokenizer, model)

    train_raw_dataset = load_dataset(
        'json', data_files={'train': args.train_file})
    test_raw_dataset = load_dataset(
        'json', data_files={'validation': args.test_file})
    # dataset = parsing_tokenize_dataset(
    #     tokenizer, raw_dataset, 256, False, p_token_ids)
    train_dataset, variable_list = parsing_v1(tokenizer, train_raw_dataset)
    test_dataset, _ = parsing_v1(tokenizer, test_raw_dataset)
    train_dataset = train_dataset['train']
    test_dataset = test_dataset['validation']
    for name, param in model.named_parameters():
        if "embed" in name or "share" in name:
            param.require_grad = False
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./p_models/{args.outdir}/{args.dataset}_full/",
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        # per_device_eval_batch_size=8,
        max_steps=2000,
        weight_decay=1e-4,
        do_train=True,
        do_eval=False,
        # eval_steps=1,
        # evaluation_strategy='epoch',
        save_strategy='no',
        # fp16=args.fp16,
        lr_scheduler_type='polynomial',
        warmup_ratio=0.1,
        optim='adamw_torch',
        gradient_accumulation_steps=8,
        # label_smoothing_factor=0.1,
        predict_with_generate=True,
        generation_max_length=256,
        generation_num_beams=10,
        # logging_steps=100,
        logging_strategy='no',
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, label_pad_token_id=-100)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    outputs = trainer.train()
    logger.info(outputs)
    trainer.save_model(f"./p_models/{args.outdir}/{args.dataset}_full/last/")
    logger.info(trainer.predict(train_dataset,
                                max_length=256))
    # model = trainer.model

    '''
    Test
    '''
    tokenizer = AutoTokenizer.from_pretrained(f"./p_models/{args.outdir}/{args.dataset}_full/last/",
                                              use_fast=False,
                                              add_prefix_space=True,
                                              do_lower_case=False)
    tokenizer.model_max_length = 256
    model = BartForConditionalGeneration.from_pretrained(
        f"./p_models/{args.outdir}/{args.dataset}_full/last/")

    # test_raw_dataset = load_dataset(
    #     'json', data_files={'validation': args.test_file})
    # test_dataset, _ = parsing_v1(tokenizer, test_raw_dataset)
    # test_dataset = test_dataset['validation']
    #
    # data_collator = DataCollatorForSeq2Seq(
    #     tokenizer=tokenizer, label_pad_token_id=-100)
    # data_loader = DataLoader(
    #     test_dataset, batch_size=8, collate_fn=data_collator, shuffle=False)

    res = generate_template(tokenizer, model, f"logs/{args.dataset}/{args.dataset}_2k.log_structured.csv", accelerator)
    log_df = pd.read_csv(
        f"logs/{args.dataset}/{args.dataset}_2k.log_structured_corrected.csv")
    gf = log_df.EventTemplate.tolist()
    gf = [" ".join(x.split()) for x in gf]
    rs = [x.strip() for x in res]
    log_df.EventTemplate = pd.Series([x.strip() for x in res])
    os.makedirs(
        f"benchmark_results/{args.outdir}/", exist_ok=True)
    log_df.to_csv(
        f"benchmark_results/{args.outdir}/{args.dataset}_2k.log_structured.csv")
    templates = [(i + 1, t, c) for i, (t, c) in enumerate(
        sorted(Counter(res).items(), key=lambda x: x[1], reverse=True))]
    template_df = pd.DataFrame(
        templates, columns=['EventId', 'EventTemplate', 'Occurences'])
    template_df.to_csv(
        f"benchmark_results/{args.outdir}/{args.dataset}_2k.log_templates.csv")

    # gf, rs = postprocess_text(gf, rs)
    # result = metric.compute(predictions=rs, references=gf, use_stemmer=True)
    # result = {k: round(v * 100, 2) for k, v in result.items()}
    # logger.info(result)
