import argparse
import sys

sys.path.append("../..")

from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, MixedTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt.pipeline_base import PromptForClassification
from openprompt import PromptDataLoader
from datasets import load_dataset
import re
import torch
from torch.optim import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import classification_report
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from logging import getLogger
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')

batch_size = 2

max_steps = 500
gradient_accumulation_steps = 16
lr = 5e-5
adam_epsilon = 1e-8
max_grad_norm = 1.0
# ----------------
no_decay = ['bias', 'LayerNorm.weight']


def preprocess(line):
    line = re.sub(r'(blk_-?\d+)', " ", line)
    return " ".join(line.split())


def main(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    logger = getLogger(__name__)
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.DEBUG)
    logger.info(accelerator.state)
    device = accelerator.device

    seq_dataset = load_dataset("json", data_files={"train": args.train_file})

    dataset = [InputExample(guid=x['labels'], text_a=preprocess("</s>".join(x['text']))) for x in
               seq_dataset['train']]
    # classes = ["normal", "abnormal"]  # for anomaly detection
    classes = ["remove", "destroy", "timeout"]  # for failure identification

    plm, tokenizer, model_config, WrapperClass = load_plm(args.model_name, args.model_path)

    max_length = max(tokenizer.max_model_input_sizes.values())

    promptTemplate = ManualTemplate(tokenizer=tokenizer).from_file(args.prompt_template)
    promptVerbalizer = ManualVerbalizer(classes=classes, tokenizer=tokenizer).from_file(args.verbalizer)

    prompt_model = PromptForClassification(plm=plm, template=promptTemplate, verbalizer=promptVerbalizer)

    data_loader = PromptDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        shortenable=True,
        decoder_max_length=5,
        max_seq_length=max_length,
        shuffle=True,
    )

    optimizer_grouped_parameters = [
        {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps=max_steps * 0.1,
                                                          num_training_steps=max_steps)
    logger.info("***** Running training *****")
    logger.info("  Num examples = ", len(dataset))

    logger.info("  Total optimization steps = ", max_steps)
    global_step = 0
    prompt_model.to(device)
    prompt_model.train()
    total_step = 0
    progress_bar = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)
    criterion = CrossEntropyLoss()

    prompt_model.to(device)

    prompt_model, optimizer, scheduler, data_loader = accelerator.prepare(prompt_model, optimizer, scheduler,
                                                                          data_loader)

    for idx in range(0, max_steps):
        total_loss = 0.0
        sum_loss = 0.0
        for batch_idx, batch in enumerate(data_loader):
            # batch = {k: v.to(device) for k, v in batch.items()}
            total_step += 1
            labels = batch['guid']
            logits, _ = prompt_model(batch)
            loss = criterion(logits, labels)
            sum_loss += loss
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), max_grad_norm)
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(data_loader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += sum_loss
                progress_bar.update(1)
                progress_bar.set_description(f"Loss: {float(sum_loss)}")
                sum_loss = 0.
                global_step += 1
            if global_step >= max_steps:
                break
        if global_step >= max_steps:
            break
    progress_bar.close()

    test_dataset = load_dataset(
        "json", data_files={"test": args.test_file})['test']
    test_dataset = [
        InputExample(guid=test_dataset[i]['labels'], text_a=preprocess("</s>".join(test_dataset[i]['text']))) for i in
        range(len(test_dataset))]

    test_data_loader = PromptDataLoader(
        dataset=test_dataset,
        batch_size=2,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        shortenable=True,
        max_seq_length=max_length,
        decoder_max_length=5,
        shuffle=True,
    )

    prompt_model.eval()

    prompt_model, test_data_loader = accelerator.prepare(prompt_model, test_data_loader)

    ground_truth = []
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_data_loader, disable=not accelerator.is_local_main_process):
            # batch = {k: v.to(device) for k, v in batch.items()}
            ground_truth.extend(accelerator.gather(batch['guid']).detach().clone().cpu().tolist())
            logits, _ = prompt_model(batch)
            preds = torch.argmax(logits, dim=-1)
            predictions.extend([classes[x] for x in accelerator.gather(preds).detach().clone().cpu().tolist()])

    ground_truth = [classes[x] for x in ground_truth]

    logger.info('******* results *******')
    logger.info(classification_report(ground_truth[:len(predictions)], predictions, digits=3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Simple example of training script.")
    parser.add_argument("--dataset", type=str, default="HDFS",
                        help="Dataset")
    parser.add_argument("--rtime", type=int, default=1,
                        help="iteration")
    parser.add_argument("--model-name", type=str, default="bart",
                        help="Name of the trained model")
    parser.add_argument("--model-path", type=str, default="last_model",
                        help="Path to the trained model")
    parser.add_argument("--train-file", type=str, default="train.json",
                        help="Path to the train file")
    parser.add_argument("--test-file", type=str, default="test.json",
                        help="Path to the test file")
    parser.add_argument("--prompt-template", type=str, default="prompt_template.txt",
                        help="Prompt template file")
    parser.add_argument("--verbalizer", type=str, default="verbalizer.txt",
                        help="Verbalizer file")
    args = parser.parse_args()
    main(args)
