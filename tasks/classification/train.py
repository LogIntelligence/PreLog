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
from transformers import get_scheduler
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import classification_report
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from logging import getLogger
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

logger = getLogger(__name__)
logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
logger.info(accelerator.state)
device = accelerator.device

no_decay = ['bias', 'LayerNorm.weight']


def preprocess(line):
    line = re.sub(r'(blk_-?\d+)', " ", line)
    return " ".join(line.split())


def grouping(data, window_size=50):
    x = []
    for i in range(0, len(data) // window_size + 1):
        x.append(preprocess("</s>".join(data[i * window_size:(i + 1) * window_size])))
    return len(x), x


def train(tokenizer, model, template, max_length, WrapperClass, args):
    train_dataset = load_dataset("json", data_files={"train": args.train_file})['train']
    X, Y = [], []
    for inp in train_dataset:
        if args.grouping:
            n_samples, x = grouping(inp['text'], window_size=args.window_size)
            X.extend(x)
            Y.extend([inp['labels']] * n_samples)
        else:
            X.append(preprocess("</s>".join(inp['text'])))
            Y.append(inp['labels'])

    dataset = [InputExample(guid=y, text_a=x) for x, y in zip(X, Y)]

    data_loader = PromptDataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        template=template,
        tokenizer_wrapper_class=WrapperClass,
        shortenable=True,
        decoder_max_length=5,
        max_seq_length=max_length,
        shuffle=True,
    )

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = get_scheduler(args.lr_scheduler_type,
                              optimizer,
                              num_warmup_steps=args.max_steps * 0.1,
                              num_training_steps=args.max_steps)
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")

    logger.info(f"  Total optimization steps = {args.max_steps}")
    global_step = 0
    model.to(device)
    model.train()
    total_step = 0
    progress_bar = tqdm(range(args.max_steps), disable=not accelerator.is_local_main_process)
    criterion = CrossEntropyLoss()

    model, optimizer, scheduler, data_loader = accelerator.prepare(model, optimizer, scheduler,
                                                                   data_loader)

    for idx in range(0, args.max_steps):
        total_loss = 0.0
        sum_loss = 0.0
        for batch_idx, batch in enumerate(data_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            total_step += 1
            labels = batch['guid']
            logits, _, _ = model(batch)
            loss = criterion(logits, labels)
            sum_loss += loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or batch_idx == len(data_loader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += sum_loss
                progress_bar.update(1)
                progress_bar.set_description(f"Loss: {float(sum_loss)}")
                sum_loss = 0.
                global_step += 1
            if global_step >= args.max_steps:
                break
        if global_step >= args.max_steps:
            break
    progress_bar.close()
    # save model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_local_main_process:
        unwrapped_model.plm.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    return model


def evaluation(tokenizer, model, dataset, template, WrapperClass, max_length, classes, args):
    test_data_loader = PromptDataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        template=template,
        tokenizer_wrapper_class=WrapperClass,
        shortenable=True,
        max_seq_length=max_length,
        decoder_max_length=5,
        shuffle=True,
    )

    model, test_data_loader = accelerator.prepare(model, test_data_loader)

    ground_truth = []
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_data_loader, disable=not accelerator.is_local_main_process):
            batch = {k: v.to(device) for k, v in batch.items()}
            gt_gathered = accelerator.gather(batch['guid'])
            logits, _, _ = model(batch)
            preds = torch.argmax(logits, dim=-1)
            pr_gathered = accelerator.gather(preds)
            if accelerator.is_local_main_process:
                ground_truth.extend(gt_gathered.detach().clone().cpu().tolist())
                predictions.extend([classes[x] for x in pr_gathered.detach().clone().cpu().tolist()])
    accelerator.wait_for_everyone()
    ground_truth = [classes[x] for x in ground_truth]
    if accelerator.is_local_main_process:
        logger.info('******* results *******')
        logger.info(classification_report(ground_truth[:len(predictions)], predictions, digits=3))


def explanation(tokenizer, model, dataset, template, WrapperClass, max_length, classes, args):
    logger.info("***** Running explanation *****")
    ground_truth = [x.guid for x in dataset]
    for i in range(len(dataset)):
        dataset[i].guid = max(dataset[i].guid)
    test_data_loader = PromptDataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        template=template,
        tokenizer_wrapper_class=WrapperClass,
        shortenable=True,
        max_seq_length=max_length,
        decoder_max_length=5,
        shuffle=False
    )

    model, test_data_loader = accelerator.prepare(model, test_data_loader)
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_data_loader, disable=not accelerator.is_local_main_process):
            #label = accelerator.gather(batch['guid'])
            # del batch['guid']
            batch = {k: v.to(device) for k, v in batch.items()}
            label = accelerator.gather(batch['guid'])
            # gt_gathered = accelerator.gather(batch['guid'])
            inp_ids_gathered = accelerator.gather(batch['input_ids'])
            logits, _, attn_scores = model(batch)
            preds = torch.argmax(logits, dim=-1)
            pr_gathered = accelerator.gather(preds)
            attn_scores_gathered = accelerator.gather(attn_scores)
            if accelerator.is_local_main_process:
                inp_ids_gathered = (inp_ids_gathered == tokenizer.eos_token_id).view(-1, inp_ids_gathered.shape[-1])
                attn_scores_gathered = attn_scores_gathered.mean(dim=1)[:, 1, :].view(-1, inp_ids_gathered.shape[-1])
                for i in range(attn_scores_gathered.shape[0]):
                    #if label[i] == pr_gathered[i] == 1:
                    inp_ids = inp_ids_gathered[i]
                    attn_scores = attn_scores_gathered[i][inp_ids][:-1]
                    topk = torch.topk(attn_scores, k=5)
                    predictions.append(topk.indices.tolist())
                        # ground_truth.append(failure_messages[i])
    logger.info(len(ground_truth))
    logger.info(len(predictions))
    logger.info('******* results *******')
    for k in [1, 3, 5]:
        hit = 0
        for i in range(len(predictions)):
            for p in predictions[i][:k]:
                try:
                    if ground_truth[i][p]:
                        hit += 1
                        break
                except:
                    print(ground_truth[i])
        logger.info(f'Top@{k}: {hit / len(predictions)}')


def main(args):
    with open(args.verbalizer, 'r') as f:
        verbalizer = f.readlines()
        classes = [x.strip() for x in verbalizer]
    plm, tokenizer, model_config, WrapperClass = load_plm(args.model_name, args.model_path)

    promptTemplate = ManualTemplate(tokenizer=tokenizer).from_file(args.prompt_template)
    promptVerbalizer = ManualVerbalizer(classes=['normal', 'abnormal'], tokenizer=tokenizer).from_file(args.verbalizer)

    max_length = max(tokenizer.max_model_input_sizes.values())
    prompt_model = PromptForClassification(plm=plm, template=promptTemplate, verbalizer=promptVerbalizer)
    if args.do_train:
        prompt_model = train(tokenizer, prompt_model, promptTemplate, max_length, WrapperClass, args)

    prompt_model.eval()
    test_dataset = load_dataset(
        "json", data_files={"test": args.test_file})['test']

    if args.do_eval:
        test_dataset = [
            InputExample(
                guid=test_dataset[i]['labels'],
                text_a=preprocess("</s>".join(test_dataset[i]['text'])),
                # meta=test_dataset[i]['meta']
            )
            for i in range(len(test_dataset))
        ]
        evaluation(tokenizer, prompt_model, test_dataset, promptTemplate, WrapperClass, max_length, classes, args)

    elif args.do_explain:
        test_dataset = [
            InputExample(
                guid=test_dataset[i]['meta'],
                text_a=preprocess("</s>".join(test_dataset[i]['text'])),
                # meta=test_dataset[i]['meta']
            )
            for i in range(len(test_dataset)) if test_dataset[i]['labels'] == 1
        ]
        logger.info(len([len(x.guid) for x in test_dataset if len(x.guid) < 20]))
        explanation(tokenizer, prompt_model, test_dataset, promptTemplate, WrapperClass, max_length, classes, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Simple example of training script.")
    parser.add_argument("--dataset", type=str, default="HDFS",
                        help="Dataset")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Max training steps")
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
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--lr-scheduler-type", type=str, default="polynomial", help="Learning rate scheduler type")
    parser.add_argument("--max-length", type=int, default=512, help="Max length")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max grad norm")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--num-workers", type=int, default=4, help="Num workers")

    parser.add_argument("--grouping", default=False, help="Grouping", action='store_true')
    parser.add_argument("--window-size", type=int, default=10, help="Window size")
    parser.add_argument("--do-train", default=False, help="Do train", action='store_true')
    parser.add_argument("--do-eval", default=False, help="Do eval", action='store_true')
    parser.add_argument("--do-explain", default=False, help="Do explanation", action='store_true')
    parser.add_argument("--output-dir", type=str, default="output", help="Output dir")

    args = parser.parse_args()
    main(args)
