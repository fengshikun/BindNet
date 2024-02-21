# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
import pandas as pd
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from unicore.losses.cross_entropy import CrossEntropyLoss
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve, auc, roc_curve
import numpy as np
import warnings
import os


@register_loss("finetune_cross_entropy")
class FinetuneCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            # features_only=True,
            # classification_head_name=self.args.classification_head_name,
        )
        logit_output = net_output[0]
        loss = self.compute_loss(model, logit_output, sample, reduce=reduce)
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            probs = F.softmax(logit_output.float(), dim=-1).view(
                -1, logit_output.size(-1)
            )
            logging_output = {
                "loss": loss.data,
                "prob": probs.data,
                "target": sample["target"]["finetune_target"].view(-1).data,
                # "smi_name": sample["smi_name"],
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = F.log_softmax(net_output.float(), dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        targets = sample["target"]["finetune_target"].view(-1)
        loss = F.nll_loss(
            lprobs,
            targets,
            reduction="sum" if reduce else "none",
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            f"{split}_loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            acc_sum = sum(
                sum(log.get("prob").argmax(dim=-1) == log.get("target"))
                for log in logging_outputs
            )
            probs = torch.cat([log.get("prob") for log in logging_outputs], dim=0)
            metrics.log_scalar(
                f"{split}_acc", acc_sum / sample_size, sample_size, round=3
            )
            if probs.size(-1) == 2:
                targets = torch.cat(
                    [log.get("target", 0) for log in logging_outputs], dim=0
                )
                df = pd.DataFrame(
                    {
                        "probs": probs[:, 1].cpu(),
                        "targets": targets.cpu(),
                    }
                )
                auroc = roc_auc_score(df["targets"], df["probs"])
                precision, recall, thresholds = precision_recall_curve(df["targets"], df["probs"])
                auprc = auc(recall, precision)
                agg_auc = roc_auc_score(df["targets"], df["probs"])
                metrics.log_scalar(f"{split}_auc", auroc, sample_size, round=3)
                metrics.log_scalar(f"{split}_auprc", auprc, sample_size, round=3)
                metrics.log_scalar(f"{split}_agg_auc", agg_auc, sample_size, round=4)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train


@register_loss("multi_task_BCE")
class MultiTaskBCELoss(CrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            masked_tokens=None,
            features_only=True,
            classification_head_name=self.args.classification_head_name,
        )
        logit_output = net_output[0]
        is_valid = sample["target"]["finetune_target"] > -0.5
        loss = self.compute_loss(
            model, logit_output, sample, reduce=reduce, is_valid=is_valid
        )
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            probs = torch.sigmoid(logit_output.float()).view(-1, logit_output.size(-1))
            logging_output = {
                "loss": loss.data,
                "prob": probs.data,
                "target": sample["target"]["finetune_target"].view(-1).data,
                "num_task": self.args.num_classes,
                "sample_size": sample_size,
                "conf_size": self.args.conf_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True, is_valid=None):
        pred = net_output[is_valid].float()
        targets = sample["target"]["finetune_target"][is_valid].float()
        loss = F.binary_cross_entropy_with_logits(
            pred,
            targets,
            reduction="sum" if reduce else "none",
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            agg_auc_list = []
            num_task = logging_outputs[0].get("num_task", 0)
            conf_size = logging_outputs[0].get("conf_size", 0)
            y_true = (
                torch.cat([log.get("target", 0) for log in logging_outputs], dim=0)
                .view(-1, conf_size, num_task)
                .cpu()
                .numpy()
                .mean(axis=1)
            )
            y_pred = (
                torch.cat([log.get("prob") for log in logging_outputs], dim=0)
                .view(-1, conf_size, num_task)
                .cpu()
                .numpy()
                .mean(axis=1)
            )
            # [test_size, num_classes] = [test_size * conf_size, num_classes].mean(axis=1)
            for i in range(y_true.shape[1]):
                # AUC is only defined when there is at least one positive data.
                if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                    # ignore nan values
                    is_labeled = y_true[:, i] > -0.5
                    agg_auc_list.append(
                        roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
                    )
            if len(agg_auc_list) < y_true.shape[1]:
                warnings.warn("Some target is missing!")
            if len(agg_auc_list) == 0:
                raise RuntimeError(
                    "No positively labeled data available. Cannot compute Average Precision."
                )
            agg_auc = sum(agg_auc_list) / len(agg_auc_list)
            metrics.log_scalar(f"{split}_agg_auc", agg_auc, sample_size, round=4)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train


@register_loss("finetune_cross_entropy_pocket")
class FinetuneCrossEntropyPocketLoss(FinetuneCrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.args.classification_head_name,
        )
        logit_output = net_output[0]
        loss = self.compute_loss(model, logit_output, sample, reduce=reduce)
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            probs = F.softmax(logit_output.float(), dim=-1).view(
                -1, logit_output.size(-1)
            )
            logging_output = {
                "loss": loss.data,
                "prob": probs.data,
                "target": sample["target"]["finetune_target"].view(-1).data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            acc_sum = sum(
                sum(log.get("prob").argmax(dim=-1) == log.get("target"))
                for log in logging_outputs
            )
            metrics.log_scalar(
                f"{split}_acc", acc_sum / sample_size, sample_size, round=3
            )
            preds = (
                torch.cat(
                    [log.get("prob").argmax(dim=-1) for log in logging_outputs], dim=0
                )
                .cpu()
                .numpy()
            )
            targets = (
                torch.cat([log.get("target", 0) for log in logging_outputs], dim=0)
                .cpu()
                .numpy()
            )
            metrics.log_scalar(f"{split}_pre", precision_score(targets, preds), round=3)
            metrics.log_scalar(f"{split}_rec", recall_score(targets, preds), round=3)
            metrics.log_scalar(
                f"{split}_f1", f1_score(targets, preds), sample_size, round=3
            )


@register_loss("finetune_dude_cross_entropy")
class FinetuneDudeCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)
        self.fold = self.task.args.DUDE_fold
        self.task_name = self.task.args.task_name

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
        )
        logit_output = net_output[0]
        loss = self.compute_loss(model, logit_output, sample, reduce=reduce)
        sample_size = sample["target"]["affinity"].size(0)

        if not self.training:
            probs = F.softmax(logit_output.float(), dim=-1).view(
                -1, logit_output.size(-1)
            )
            logging_output = {
                "loss": loss.data,
                "prob": probs.data,
                "target": sample["target"]["affinity"].view(-1).data,
                "sample_size": sample_size,
                "bsz": sample["target"]["affinity"].size(0),
            }
        else:
            probs = F.softmax(logit_output.float(), dim=-1).view(
                -1, logit_output.size(-1)
            )
            logging_output = {
                "loss": loss.data,
                "prob": probs.data,
                "target": sample["target"]["affinity"].view(-1).data,
                "sample_size": sample_size,
                "bsz": sample["target"]["affinity"].size(0),
            }
        logging_output['fold'] = self.fold
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = F.log_softmax(net_output.float(), dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        targets = sample["target"]["affinity"].view(-1)
        loss = F.nll_loss(
            lprobs,
            targets,
            reduction="sum" if reduce else "none",
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=6
        )
        if 'train' in split:
            probs = torch.cat([log.get("prob") for log in logging_outputs], dim=0)
            targets = torch.cat(
                [log.get("target", 0) for log in logging_outputs], dim=0
            )
            acc_sum = sum(
                sum(log.get("prob").argmax(dim=-1) == log.get("target"))
                for log in logging_outputs
            )
            metrics.log_scalar(
                f"{split}_acc", acc_sum / sample_size, sample_size, round=6
            )
        if "valid" in split or "test" in split:
            acc_sum = sum(
                sum(log.get("prob").argmax(dim=-1) == log.get("target"))
                for log in logging_outputs
            )
            probs = torch.cat([log.get("prob") for log in logging_outputs], dim=0)
            metrics.log_scalar(
                f"{split}_acc", acc_sum / sample_size, sample_size, round=6
            )
            if probs.size(-1) == 2:
                targets = torch.cat(
                    [log.get("target", 0) for log in logging_outputs], dim=0
                )
                df = pd.DataFrame(
                    {
                        "probs": probs[:, 1].cpu(),
                        "targets": targets.cpu(),
                    }
                )

                def get_roce(predList, targetList, roceRate):
                    p = sum(targetList)
                    n = len(targetList) - p
                    predList = [[index, x] for index, x in enumerate(predList)]
                    predList = sorted(predList, key=lambda x:x[1], reverse=True)
                    tp1 = 0
                    fp1 = 0
                    for x in predList:
                        if(targetList[x[0]] == 1):
                            tp1 += 1
                        else:
                            fp1 += 1
                            if(fp1>((roceRate*n)/100)):
                                break
                    roce = (tp1*n)/(p*fp1)
                    return roce

                RE_05pa =  get_roce(df["probs"].to_list(), df["targets"].to_list(), 0.5)
                RE_1pa = get_roce(df["probs"].to_list(), df["targets"].to_list(), 1)
                RE_2pa = get_roce(df["probs"].to_list(), df["targets"].to_list(), 2)
                RE_5pa = get_roce(df["probs"].to_list(), df["targets"].to_list(), 5)

                auroc = roc_auc_score(df["targets"], df["probs"])

                metrics.log_scalar(f"{split}_sampleSize", sample_size, sample_size)
                metrics.log_scalar(f"{split}_auc", auroc, sample_size)
                metrics.log_scalar(f"{split}_RE_05pa", RE_05pa, sample_size)
                metrics.log_scalar(f"{split}_RE_1pa", RE_1pa, sample_size)
                metrics.log_scalar(f"{split}_RE_2pa", RE_2pa, sample_size)
                metrics.log_scalar(f"{split}_RE_5pa", RE_5pa, sample_size)


    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train