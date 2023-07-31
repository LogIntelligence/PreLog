import re
from collections import Counter
import torch
from tqdm import tqdm
from datasets import load_dataset

from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader

masking = [
    {"regex_pattern": "((?<=[^A-Za-z0-9])|^)(\\/\S\\.[\\S]+)((?=[^A-Za-z0-9])|$)",
     "mask_with": "<*>"},
    {"regex_pattern": "((?<=[^A-Za-z0-9])|^)(\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3})((?=[^A-Za-z0-9])|$)",
     "mask_with": "<*>"},
    {"regex_pattern": "((?<=[^A-Za-z0-9])|^)([0-9a-f]{6,} ?){3,}((?=[^A-Za-z0-9])|$)",
     "mask_with": "<*>"},
    {"regex_pattern": "((?<=[^A-Za-z0-9])|^)([0-9A-F]{4} ?){4,}((?=[^A-Za-z0-9])|$)",
     "mask_with": "<*>"},
    {"regex_pattern": "((?<=[^A-Za-z0-9])|^)(0x[a-f0-9A-F]+)((?=[^A-Za-z0-9])|$)",
     "mask_with": "<*>"},
    {"regex_pattern": "((?<=[^A-Za-z0-9])|^)([\\-\\+]?\\d+)((?=[^A-Za-z0-9])|$)", "mask_with": "<*>"},
    # {"regex_pattern": "((?<=[^A-Za-z0-9])|^)([a-f0-9A-F]+)((?=[^A-Za-z0-9])|$)", "mask_with": "<*>"},
    {"regex_pattern": "(?<=executed cmd )(\".+?\")", "mask_with": "<*>"}
]


def LCS(seq1, seq2):
    lengths = [[0 for j in range(len(seq2) + 1)] for i in range(len(seq1) + 1)]
    # row 0 and column 0 are initialized to 0 already
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            if seq1[i][:len(seq2[j])] == seq2[j] or (seq1[i][1:len(seq2[j]) + 1] == seq2[j] and seq1[i][0] == "Ġ"):
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j +
                               1] = max(lengths[i + 1][j], lengths[i][j + 1])

    # read the substring out from the matrix
    result = []
    lenOfSeq1, lenOfSeq2 = len(seq1), len(seq2)
    while lenOfSeq1 != 0 and lenOfSeq2 != 0:
        if lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1 - 1][lenOfSeq2]:
            lenOfSeq1 -= 1
        elif lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1][lenOfSeq2 - 1]:
            lenOfSeq2 -= 1
        else:
            if seq1[lenOfSeq1 - 1] == seq2[lenOfSeq2 - 1]:
                pre = " " if seq1[lenOfSeq1 - 1][0] == "Ġ" else ""
                try:
                    suf = " " if seq1[lenOfSeq1][0] == "Ġ" else ""
                except:
                    suf = ""
            else:
                pre = " " if seq1[lenOfSeq1 - 1][0] == "Ġ" else ""
                try:
                    if len(seq1[lenOfSeq1 - 1]) > len(seq2[lenOfSeq2 - 1]) + 1:
                        suf = " " if seq1[lenOfSeq1 -
                                          1][len(seq2[lenOfSeq2 - 1]) + 1] == " " else ""
                    else:
                        suf = " " if seq1[lenOfSeq1][0] == "Ġ" else ""
                except:
                    suf = ""
            result = [(seq2[lenOfSeq2 - 1], pre, suf)] + result
            lenOfSeq1 -= 1
            lenOfSeq2 -= 1
    return result


def map_template(tokenizer, c, t):
    content = tokenizer.convert_ids_to_tokens(c)
    template = tokenizer.convert_ids_to_tokens(t)
    content = content[: content.index("</s>")]
    template = template[: template.index("</s>")]
    lcs = LCS(content, template)
    # print(content)
    # print(template)
    # print(lcs)
    # pre_suf = []

    # for i, tok in enumerate(template):
    #     is_exist = False
    #     for j, con in enumerate(content):
    #         if tok == con[:len(tok)]:
    #             is_exist = True
    #             break
    #     if not is_exist:
    #         template[i] == "<*>"
    # for i, tok in enumerate(template):
    # try:
    #     next_var_token = template[i + 1] == "<*>"
    # except:
    #     next_var_token = False
    # pre, suf = "", ""
    # for j, con in enumerate(content):
    #     if tok == con[:len(tok)] or tok == con[1:len(tok) + 1]:
    #         if con[0] == "Ġ" or tok[0] == "Ġ":
    #             pre = " "
    #         if next_var_token:
    #             if len(con) > len(tok):
    #                 if con[len(tok)] == " ":
    #                     suf = " "
    #             else:
    #                 if j < len(content) - 1:
    #                     if content[j + 1][0] == "Ġ":
    #                         suf = " "
    #         break
    # pre_suf.append((pre, suf))

    # template = [tokenizer.convert_tokens_to_string(x).strip() for x in template]
    # print(tokens)
    res = [" "]
    # print(t)
    for i in range(0, len(template)):
        if len(lcs) > 0 and template[i] == lcs[0][0]:
            res.append(
                lcs[0][1] + tokenizer.convert_tokens_to_string(template[i]).strip() + lcs[0][2])
            lcs = lcs[1:]
        else:
            # if "Ġ" in tokens[i]:
            #     if "<*>" not in res[-1]:
            #         # print(tokens[i])
            #         res.append("Ġ<*>")
            if "<*>" not in res[-1]:
                res.append("<*>")
    # print(res)
    # print("*" * 10)
    r = "".join(res)
    while "<*><*>" in r:
        r = r.replace("<*><*>", "<*>")
    while "<*>:<*>" in r:
        r = r.replace("<*>:<*>", "<*>")
    while "<*>.<*>" in r:
        r = r.replace("<*>.<*>", "<*>")
    while "<*>,<*>" in r:
        r = r.replace("<*>,<*>", "<*>")
    while "<*>_<*>" in r:
        r = r.replace("<*>_<*>", "<*>")
    return " ".join(r.strip().split())


def map_template_v2(tokenizer, c, t):
    content = tokenizer.convert_ids_to_tokens(c)
    template = tokenizer.convert_ids_to_tokens(t)
    content = content[: content.index("</s>")]
    template = template[: template.index("</s>")]
    p = 0
    for i, tok in enumerate(template):
        is_exist = False
        for j in range(p, len(content)):
            con = content[j]
            if tok == con[:len(tok)]:
                is_exist = True
                template[i] = tokenizer.convert_tokens_to_string(tok).strip()
                if len(con) <= len(tok) + 1:
                    p = j
                    try:
                        if content[j + 1][0] == "Ġ":
                            template[i] = template[i] + " "
                    except:
                        pass
                else:
                    if con[con.find(tok) + len(tok) + 1:] == " ":
                        template[i] = template + " "
                content[j] = content[j][len(tok):]
                break
            # if tok == con[-len(tok):]:
            #     is_exist = True
            #     p = j
            #     try:
            #         if content[j + 1][0] == "Ġ":
            #             template[i] = template[i] + " "
            #     except:
            #         pass
            #     break
        if not is_exist:
            template[i] == "<*>"
    # print(template)
    # print("*" * 10)
    # for i, tok in enumerate(template):
    #     try:
    #         next_var_token = template[i + 1] == "<*>"
    #     except:
    #         next_var_token = False
    #     pre, suf = "", ""
    #     for j, con in enumerate(content):
    #         if tok == con[:len(tok)] or tok == con[1:len(tok) + 1]:
    #             if con[0] == "Ġ" or tok[0] == "Ġ":
    #                 pre = " "
    #             if next_var_token:
    #                 if len(con) > len(tok):
    #                     if con[len(tok)] == " ":
    #                         suf = " "
    #                 else:
    #                     if j < len(content) - 1:
    #                         if content[j + 1][0] == "Ġ":
    #                             suf = " "
    #             break
    #     pre_suf.append((pre, suf))

    r = "".join(template)
    return " ".join(r.strip().split())


def map_template_v3(content, template):
    # content = " ".join(content)
    # print(content)
    template_char = list(template)
    i = 0
    while i < len(template):
        if template[i: i + 3] == "<*>":
            i = i + 3
        else:
            if template_char[i] not in content:
                template_char[i] = " "
            i = i + 1

    template = "".join(template_char)
    template = " ".join(template.split())
    t_tokens = re.split("(\<\*\>| )", template)
    t_tokens = [x for x in t_tokens if len(x.strip()) > 0]
    # t_tokens = [x for x in t_tokens if x ==
    #             "<*>" or f"{x} " in content or f" {x}" in content]
    # print(t_tokens, content)
    res = ""
    for t in t_tokens:
        p = content.find(t)
        if p < 0 and t != "<*>":
            continue
        if content.find(" " + t) >= 0:
            res = res + " "
        res = res + t
        if t != "<*>" and content.find(t + " ") >= 0:
            res = res + " "
        if p >= 0:
            content = content[p + len(t):]
    # print(res)
    # print("*" * 20)
    res = " ".join(res.strip().split())
    while "<*><*>" in res:
        res = res.replace("<*><*>", "<*>")
    return res


def get_parameter_list(s, template_regex):
    """
    :param s: log message
    :param template_regex: template regex with <*> indicates parameters
    :return: list of parameters
    """
    # template_regex = re.sub(r"<.{1,5}>", "<*>", template_regex)
    if "<*>" not in template_regex:
        return []
    template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
    template_regex = re.sub(r'\\ +', r'\\s+', template_regex)
    template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
    parameter_list = re.findall(template_regex, s)
    parameter_list = parameter_list[0] if parameter_list else ()
    parameter_list = list(parameter_list) if isinstance(
        parameter_list, tuple) else [parameter_list]
    # print(s, parameter_list)
    return parameter_list


def parsing_tokenize_dataset(tokenizer, dataset, max_length, padding, p_token_id):
    def tokenize_and_align_labels(examples):
        examples["text"] = [" ".join(x.strip().split())
                            for x in examples["text"]]
        tokenized_inputs = tokenizer(
            examples["text"],
            max_length=max_length,
            padding=padding,
            truncation=True,
            is_split_into_words=False,
        )
        target_tokens = []
        for i, label in enumerate(examples["label"]):
            content = examples["text"][i]
            label = " ".join(label.strip().split())
            variable_list = get_parameter_list(content, label)
            input_ids = tokenized_inputs.input_ids[i]
            input_tokens = tokenizer.convert_ids_to_tokens(input_ids)

            target_token = []
            processing_variable = False
            variable_token = ""
            input_tokens = [tokenizer.convert_tokens_to_string(
                [x]) for x in input_tokens]
            # pos = 0
            for ii, (input_idx, input_token) in enumerate(zip(input_ids, input_tokens)):
                if input_idx in tokenizer.all_special_ids:
                    target_token.append(-100)
                    continue
                # Set target token for the first token of each word.
                if (label[:3] == "<*>" or label[:len(input_token.strip())] != input_token.strip()) \
                        and processing_variable is False:
                    processing_variable = True
                    variable_token = variable_list.pop(0).strip()
                    pos = label.find("<*>")
                    label = label[label.find("<*>") + 3:].strip()
                    input_token = input_token.strip()[pos:]

                if processing_variable:
                    input_token = input_token.strip()
                    if input_token == variable_token[:len(input_token)]:
                        target_token.append(p_token_id)
                        variable_token = variable_token[len(
                            input_token):].strip()
                        # print(variable_token, "+++", input_token)
                    elif len(input_token) > len(variable_token):
                        target_token.append(p_token_id)
                        label = label[len(input_token) -
                                      len(variable_token):].strip()
                        variable_token = ""
                    else:
                        raise ValueError(
                            f"error at {variable_token} ---- {input_token}")
                    if len(variable_token) == 0:
                        processing_variable = False
                else:
                    input_token = input_token.strip()
                    if input_token == label[:len(input_token)]:
                        target_token.append(input_idx)
                        label = label[len(input_token):].strip()
                    else:
                        raise ValueError(
                            f"error at {content} ---- {input_token}")

            target_tokens.append(target_token)
            tokenized_inputs.input_ids[i] = input_ids
        tokenized_inputs["labels"] = target_tokens
        return tokenized_inputs

    processed_raw_datasets = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    return processed_raw_datasets


def preprocess(line):
    line = line.strip()
    for r in masking:
        line = re.sub(r["regex_pattern"], r["mask_with"], line)
    return " ".join(line.strip().split())


def check_variable(x):
    for r in masking:
        if re.match(r['regex_pattern'], x) is not None:
            return False
    return True


def parsing_v1(tokenizer, raw_dataset):
    variable_list = []

    def convert_to_features(example_batch):
        for i in range(len(example_batch['text'])):
            variable_list.extend(get_parameter_list(
                example_batch['text'][i], example_batch['label'][i]))

        example_batch['text'] = [preprocess(x) for x in example_batch['text']]
        example_batch['label'] = [" ".join(x.split())
                                  for x in example_batch['label']]
        # for i in range(len(example_batch['text'])):
        #     print(example_batch['text'][i])
        #     print(example_batch['label'][i])
        #     print("=" * 20)
        model_inputs = tokenizer(
            example_batch['text'], truncation=True, padding=False, is_split_into_words=False)
        labels = tokenizer(
            example_batch['label'], truncation=True, padding=False, is_split_into_words=False)
        model_inputs['labels'] = labels['input_ids']

        return model_inputs

    dataset = raw_dataset.map(convert_to_features, num_proc=1, batched=True, remove_columns=[
        'text', 'label'], desc="Running tokenizer on dataset")

    variable_list = Counter(variable_list)
    variable_list = sorted(variable_list.items(),
                           key=lambda k: k[1], reverse=True)

    variable_list = [x[0] for x in variable_list]
    variable_list = [x for x in variable_list if len(x) > 2 and check_variable(x)]
    return dataset, variable_list[:8]


def generate_template(tokenizer, model, log_file, accelerator):
    res = {}
    device = accelerator.device
    model.to(device)
    model.eval()

    def tokenize_and_align_labels(examples):
        examples['Content'] = [preprocess(x) for x in examples['Content']]
        tokenized_inputs = tokenizer(
            examples['Content'],
            padding=False,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=False,
        )
        return tokenized_inputs

    dataset = load_dataset('csv', data_files=log_file)
    logs = dataset['train']['Content']
    remove_columns = list(dataset['train'].features.keys())
    remove_columns.remove("LineId")
    test_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=remove_columns,
        desc="Running tokenizer on dataset",
    )
    test_dataset = test_dataset['train']

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, label_pad_token_id=-100)

    test_loader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=32, pin_memory=True)
    model, test_loader = accelerator.prepare(
        model, test_loader
    )

    for batch in tqdm(test_loader, desc='Parsing', disable=not accelerator.is_local_main_process):
        line_id = batch.pop("LineId")
        # batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model.generate(input_ids=batch['input_ids'].to(device), max_length=512,
                                     attention_mask=batch['attention_mask'].to(device),
                                     num_beams=8,
                                     )
        predictions = accelerator.pad_across_processes(
            outputs, dim=1, pad_index=tokenizer.pad_token_id)
        predictions_gathered = accelerator.gather(predictions)
        line_id = accelerator.gather(line_id)
        if accelerator.is_local_main_process:
            templates = tokenizer.batch_decode(predictions_gathered, skip_special_tokens=True)
            for i, t in zip(line_id.detach().clone().tolist(), templates):
                res[i] = map_template_v3(" ".join(logs[i - 1].split()), t)

    accelerator.wait_for_everyone()

    res = [x for _, x in sorted(res.items(), key=lambda k: k[0])]
    return res
