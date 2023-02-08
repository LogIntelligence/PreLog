# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


@dataclass
class PCLLogCriterionConfig(FairseqDataclass):
    entry_level: bool = field(
        default=False,
        metadata={"help": "loss for log-entry level"},
    )
    sequence_level: bool = field(
        default=False,
        metadata={"help": "loss for log-sequence level"},
    )
    contrastive_weight: float = field(
        default=0.1,
        metadata={"help": "contrastive weight"},
    )
    hard_negative_weight: float = field(
        default=1.0,
        metadata={"help": "hard negative weight"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("pcllog_contrastive", dataclass=PCLLogCriterionConfig)
class PCLLogCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, entry_level=False, sequence_level=False, contrastive_weight=0.1, hard_negative_weight=1.0):
        super().__init__(task)
        self.hard_negative_weight = hard_negative_weight
        self.sentence_avg = sentence_avg
        self.contrastive_weight = contrastive_weight
        self.entry_level = entry_level
        self.sequence_level = sequence_level
        self.sim = Similarity(temp=0.05)
        if not self.entry_level and not self.sequence_level:
            raise NotImplementedError("Must choose at least one of entry_level or sequence_level")

    def pooler(self, input, output, lengths):
        output = output.transpose(0, 1)
        attn_mask = torch.tensor(
            [[1 if t == self.eos_idx and j <= lengths[i] else 0 for j, t in enumerate(inp)] for i, inp in
             enumerate(input)], device=input.device)
        return (output * attn_mask.unsqueeze(-1)).sum(1) / attn_mask.sum(-1).unsqueeze(-1)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # Compute language model loss (cross entropy)
        net_output = model(sample["net_input"]["lm_tokens"], sample["net_input"]["lm_lengths"],
                           sample["net_input"]["prev_output_tokens"])
        # print(net_output[0].shape)
        # print(len(net_output[1]['inner_states']), net_output[1]['inner_states'][-1].shape)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        if self.entry_level:
            lm_loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
            logging_output = {
                "lm_loss": lm_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
        else:
            lm_loss = -1
            logging_output = {
                "lm_loss": -1,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }


        # Compute contrastive loss
        if "noising_tokens" in sample["net_input"].keys():
            cl_o1 = net_output[2][0]
            cl_o1 = self.pooler(sample["net_input"]["lm_tokens"], cl_o1,
                                sample["net_input"]["lm_lengths"])
            cl_o2 = model(sample["net_input"]["noising_tokens"][0], sample["net_input"]["noising_lengths"][0],
                          sample["net_input"]["prev_output_tokens"])[2][0]
            cl_o2 = self.pooler(sample["net_input"]["noising_tokens"][0], cl_o2,
                                sample["net_input"]["noising_lengths"][0])
            if sample["net_input"]["noising_tokens"][1] is not None:
                # hard negative
                cl_o3 = model(sample["net_input"]["noising_tokens"][1], sample["net_input"]["noising_lengths"][1],
                              sample["net_input"]["prev_output_tokens"])[2][0]
                cl_o3 = self.pooler(sample["net_input"]["noising_tokens"][1], cl_o3,
                                    sample["net_input"]["noising_lengths"][1])
            else:
                cl_o3 = None
            cl_loss = self.compute_contrastive_loss(cl_o1, cl_o2, cl_o3)

            if lm_loss != -1:
                loss = lm_loss * self.contrastive_weight + cl_loss
            else:
                loss = cl_loss
            logging_output["cl_loss"] = cl_loss.data
        else:
            loss = lm_loss
            logging_output["cl_loss"] = -1

        logging_output["loss"] = loss.data

        return loss, sample_size, logging_output

    def compute_contrastive_loss(self, z1, z2, z3=None):
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        # print("cos_sim shape:", cos_sim.shape)
        # Hard negative
        if z3 is not None:
            z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)
            # print("cos_sim shape:", cos_sim.shape)

        labels = torch.arange(cos_sim.size(0)).long().to(z1.device)
        # print(labels)
        loss_fct = nn.CrossEntropyLoss()
        if z3 is not None:
            # Hard negative
            z3_weight = self.hard_negative_weight
            weights = torch.tensor(
                [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (
                        z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
            ).to(z3.device)
            cos_sim = cos_sim + weights
        cl_loss = loss_fct(cos_sim, labels)
        return cl_loss

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.cross_entropy(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="mean" if reduce else "none",
        )
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        # if cl_loss is None or lm_loss is None:
        metrics.log_scalar(
            "loss", loss_sum / nsentences / math.log(2), nsentences, round=3
        )

        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )
        # cl_loss, lm_loss = None, None
        if any("lm_loss" in log for log in logging_outputs):
            lm_loss = sum(log.get("lm_loss", 0) for log in logging_outputs) / nsentences / math.log(2)
            metrics.log_scalar(
                "lm_loss", lm_loss,
                sample_size, round=3
            )
        if any("cl_loss" in log for log in logging_outputs):
            cl_loss = sum(log.get("cl_loss", 0) for log in logging_outputs) / nsentences / math.log(2)
            metrics.log_scalar(
                "cl_loss", cl_loss,
                sample_size, round=3
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
