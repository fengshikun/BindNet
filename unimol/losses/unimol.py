# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
import torch.nn as nn
import wandb
import os
import numpy as np
from scipy.stats import spearmanr
import pickle as pkl

if os.environ['LOCAL_RANK'] == '0':
    wandb.login(key='a46eaf1ea4fdcf3a2a93022568aa1c730c208b50')


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


@register_loss("unimol")
class UniMolLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()
        self.seed = task.seed
        self.dist_mean = 6.312581655060595
        self.dist_std = 3.3899264663911888
        self.ctl_2d = task.args.ctl_2d # NOTE: ctrastive learning with 2D
        if self.ctl_2d:
            self.sim = Similarity(temp=0.05) # NOTE: temperature hardcore
            self.ctl_loss = nn.CrossEntropyLoss()
        self.run_name = task.args.run_name

        if os.environ['LOCAL_RANK'] == '0':
            wandb.init(project="CDMOL_Pretraining", name=self.run_name)

        self.complex_pretrain = task.args.complex_pretrain

        self.regression_cls = task.args.regression_cls

        if self.complex_pretrain:
            self.dis_cls_lst = [[0, 4], [4, 6], [6, 8], [8, 10], [10, 12], [12, 14], [14, 16], [16, 18], [18, 20], [20, 100]]
            # 0-10: interval 0.2; 10-20: inteval 1; >20
            self.dis_cls_lst = []
            dis_range = np.arange(0, 10, 0.2)
            for v in dis_range:
                self.dis_cls_lst.append([v, v+0.2])
            dis_range = np.arange(10, 20, 1)
            for v in dis_range:
                self.dis_cls_lst.append([v, v+1])
            self.dis_cls_lst.append([20, 1000]) # 61 cls
            print(f'+++++++++dis cls lst is {self.dis_cls_lst}')



            assert len(self.dis_cls_lst) == task.args.dis_clsnum
            self.dis_cls_loss = nn.CrossEntropyLoss()

            self.mask_feature = task.args.mask_feature
            self.mask_only = task.args.mask_only # only mask feature

    def quantum_dis(self, ele):
        for i, interval in enumerate(self.dis_cls_lst):
            if i == len(self.dis_cls_lst) - 1 and ele >= interval[0]:
                return i
            if ele >= interval[0] and ele < interval[1]:
                return i


    def forward(self, model, sample, reduce=True):
        input_key = "net_input"
        target_key = "target"

        if self.complex_pretrain:
            pass

        if "tokens_target" in sample[target_key]:
            masked_tokens = sample[target_key]["tokens_target"].ne(self.padding_idx)
            sample_size = masked_tokens.long().sum()

        if self.ctl_2d:
            (
                logits_encoder,
                encoder_distance,
                encoder_coord,
                x_norm,
                delta_encoder_pair_rep_norm,
                graph_2d_rep,
                graph_3d_rep,

            ) = model(**sample[input_key], encoder_masked_tokens=masked_tokens)

            cos_sim = self.sim(graph_2d_rep.unsqueeze(1), graph_3d_rep.unsqueeze(0))
            ctr_sample_num = cos_sim.shape[0]
            labels = torch.arange(ctr_sample_num).long().to(cos_sim.device)
            ctl_loss = self.ctl_loss(cos_sim, labels) + self.ctl_loss(cos_sim.t(), labels)
        elif self.complex_pretrain:
            if self.mask_feature:
                all_feat_x, all_padding_mask, dis_cls_logits, x_norm, delta_encoder_pair_rep_norm, mask_pred_target_feat = model(**sample[input_key])
            else:
                all_feat_x, all_padding_mask, dis_cls_logits, x_norm, delta_encoder_pair_rep_norm, = model(**sample[input_key])
            encoder_coord = None
            encoder_distance = None
        else:
            (
                logits_encoder,
                encoder_distance,
                encoder_coord,
                x_norm,
                delta_encoder_pair_rep_norm,
            ) = model(**sample[input_key], encoder_masked_tokens=masked_tokens)



        if self.complex_pretrain:
            # construct label for classification
            prot_num_lst = sample['target']['prot_num']
            lig_num_lst = sample['target']['lig_num']


            # get the distance
            batch_size = sample['target']['all_pos'].shape[0]
            proc_reg_distance_lst = []

            lig_proc_reg_org_dist_lst = []
            for s_idx in range(batch_size):
                all_pos = sample['target']['all_pos'][s_idx]
                proc_num = prot_num_lst[s_idx].item()
                lig_num = lig_num_lst[s_idx].item()
                proc_pos = all_pos[:proc_num, :]
                lig_pos = all_pos[proc_num:(proc_num+lig_num), :]
                proc_lig_distance = torch.cdist(proc_pos, lig_pos)

                lig_proc_distance = torch.cdist(lig_pos, proc_pos)

                lig_proc_reg_org_dist_lst.append(lig_proc_distance)


                proc_lig_quant_dis = proc_lig_distance.clone()
                proc_lig_quant_dis = proc_lig_quant_dis.cpu().apply_(self.quantum_dis).cuda()
                # quantum the list
                # proc_lig_quant_dis = self.quantum_dis(proc_lig_distance)

                proc_reg_distance_lst.append(proc_lig_quant_dis.flatten().to(torch.long))

            proc_reg_distance_lst = torch.cat(proc_reg_distance_lst)

            # calculate loss
            logging_output = {
                "sample_size": 1,
                "bsz": batch_size,
                "seq_len": all_feat_x.shape[1],
                "prot_max_len": prot_num_lst.max().item(),
                "lig_max_len": lig_num_lst.max().item()
            }

            cross_distance_predict = dis_cls_logits
            distance_loss_all = 0
            # batch_size = cross_distance_predict.shape[0]
            for idx in range(batch_size):
                lig_proc_distance = lig_proc_reg_org_dist_lst[idx] # proc num * ligand num
                proc_num = prot_num_lst[idx].item()
                lig_num = lig_num_lst[idx].item()

                lig_proc_distance_pred = cross_distance_predict[idx][1:1+lig_num, 1:1+proc_num]

                if self.regression_cls:
                    lig_proc_distance_cls_target = lig_proc_distance.cpu().apply_(self.quantum_dis).cuda()
                    distance_loss = self.dis_cls_loss(lig_proc_distance_pred.reshape(-1, len(self.dis_cls_lst)), lig_proc_distance_cls_target.flatten().to(torch.long))
                else:
                    ### distance loss
                    distance_mask = lig_proc_distance.ne(0)  # 0 is padding
                    if self.args.dist_threshold > 0:
                        distance_mask &= (
                            lig_proc_distance < self.args.dist_threshold
                        )
                    distance_predict = lig_proc_distance_pred[distance_mask]
                    distance_target = lig_proc_distance[distance_mask]
                    distance_loss = F.mse_loss(
                        distance_predict.float(), distance_target.float(), reduction="mean"
                    )


                
                
                if distance_loss.isnan().sum():
                    print('distance_loss nan') # NOTE empty
                    distance_loss = 0

                distance_loss_all += distance_loss

            distance_loss_all /= batch_size

            logging_output["dis_reg_cross_loss_value"] = distance_loss_all

            if self.mask_only:
                loss = 0
            else:
                loss = distance_loss_all
                if loss.isnan().sum():
                    print('loss nan')
            # loss = distance_loss_all




            if self.mask_feature:
                pred_feat, target_feat = mask_pred_target_feat
                mask_feat_loss = F.mse_loss(
                        pred_feat.float(), target_feat.float(), reduction="mean"
                    )
                logging_output["mask_feat_loss"] = mask_feat_loss
                loss += mask_feat_loss
            # logging_output["loss"] = loss.data
            # return loss, 1, logging_output
        else:

            target = sample[target_key]["tokens_target"]
            if masked_tokens is not None:
                target = target[masked_tokens]
            masked_token_loss = F.nll_loss(
                F.log_softmax(logits_encoder, dim=-1, dtype=torch.float32),
                target,
                ignore_index=self.padding_idx,
                reduction="mean",
            )
            masked_pred = logits_encoder.argmax(dim=-1)
            masked_hit = (masked_pred == target).long().sum()
            masked_cnt = sample_size
            loss = masked_token_loss * self.args.masked_token_loss
            logging_output = {
                "sample_size": 1,
                "bsz": sample[target_key]["tokens_target"].size(0),
                "seq_len": sample[target_key]["tokens_target"].size(1)
                * sample[target_key]["tokens_target"].size(0),
                "masked_token_loss": masked_token_loss.data,
                "masked_token_hit": masked_hit.data,
                "masked_token_cnt": masked_cnt,
            }

        if self.ctl_2d:
            loss = loss + ctl_loss # NOTE: can add loss weight
            logging_output["ctl_2d_3d_loss"] = ctl_loss

        if encoder_coord is not None:
            # real = mask + delta
            coord_target = sample[target_key]["coord_target"]
            masked_coord_loss = F.smooth_l1_loss(
                encoder_coord[masked_tokens].view(-1, 3).float(),
                coord_target[masked_tokens].view(-1, 3),
                reduction="mean",
                beta=1.0,
            )
            loss = loss + masked_coord_loss * self.args.masked_coord_loss
            # restore the scale of loss for displaying
            logging_output["masked_coord_loss"] = masked_coord_loss.data

        if encoder_distance is not None:
            dist_masked_tokens = masked_tokens
            masked_dist_loss = self.cal_dist_loss(
                sample, encoder_distance, dist_masked_tokens, target_key, normalize=True
            )
            loss = loss + masked_dist_loss * self.args.masked_dist_loss
            logging_output["masked_dist_loss"] = masked_dist_loss.data

        if self.args.x_norm_loss > 0 and x_norm is not None:
            loss = loss + self.args.x_norm_loss * x_norm
            logging_output["x_norm_loss"] = x_norm.data

        if (
            self.args.delta_pair_repr_norm_loss > 0
            and delta_encoder_pair_rep_norm is not None
        ):
            loss = (
                loss + self.args.delta_pair_repr_norm_loss * delta_encoder_pair_rep_norm
            )
            logging_output[
                "delta_pair_repr_norm_loss"
            ] = delta_encoder_pair_rep_norm.data

        logging_output["loss"] = loss.data
        return loss, 1, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("seq_len", seq_len / bsz, 1, round=3)

        masked_loss = sum(log.get("masked_token_loss", 0) for log in logging_outputs)
        if masked_loss > 0:
            metrics.log_scalar(
                "masked_token_loss", masked_loss / sample_size, sample_size, round=3
            )
            if os.environ['LOCAL_RANK'] == '0':
                wandb.log({"masked_token_loss": masked_loss / sample_size})

            masked_acc = sum(
                log.get("masked_token_hit", 0) for log in logging_outputs
            ) / sum(log.get("masked_token_cnt", 0) for log in logging_outputs)
            metrics.log_scalar("masked_acc", masked_acc, sample_size, round=3)

        masked_coord_loss = sum(
            log.get("masked_coord_loss", 0) for log in logging_outputs
        )
        if masked_coord_loss > 0:
            metrics.log_scalar(
                "masked_coord_loss",
                masked_coord_loss / sample_size,
                sample_size,
                round=3,
            )
            if os.environ['LOCAL_RANK'] == '0':
                wandb.log({"masked_coord_loss": masked_coord_loss / sample_size})

        ctl_2d_3d_loss = sum(
            log.get("ctl_2d_3d_loss", 0) for log in logging_outputs
        )
        if ctl_2d_3d_loss > 0:
            metrics.log_scalar(
                "ctl_2d_3d_loss",
                ctl_2d_3d_loss / sample_size,
                sample_size,
                round=3,
            )
            if os.environ['LOCAL_RANK'] == '0':
                wandb.log({"ctl_2d_3d_loss": ctl_2d_3d_loss / sample_size})

        masked_dist_loss = sum(
            log.get("masked_dist_loss", 0) for log in logging_outputs
        )
        if masked_dist_loss > 0:
            metrics.log_scalar(
                "masked_dist_loss", masked_dist_loss / sample_size, sample_size, round=3
            )
            if os.environ['LOCAL_RANK'] == '0':
                wandb.log({"masked_dist_loss": masked_dist_loss / sample_size})

        x_norm_loss = sum(log.get("x_norm_loss", 0) for log in logging_outputs)
        if x_norm_loss > 0:
            metrics.log_scalar(
                "x_norm_loss", x_norm_loss / sample_size, sample_size, round=3
            )
            if os.environ['LOCAL_RANK'] == '0':
                wandb.log({"x_norm_loss": x_norm_loss / sample_size})

        delta_pair_repr_norm_loss = sum(
            log.get("delta_pair_repr_norm_loss", 0) for log in logging_outputs
        )
        if delta_pair_repr_norm_loss > 0:
            metrics.log_scalar(
                "delta_pair_repr_norm_loss",
                delta_pair_repr_norm_loss / sample_size,
                sample_size,
                round=3,
            )
            if os.environ['LOCAL_RANK'] == '0':
                wandb.log({"delta_pair_repr_norm_loss": delta_pair_repr_norm_loss / sample_size})

        dis_cls_loss_value = sum(
            log.get("dis_cls_loss_value", 0) for log in logging_outputs
        )
        if dis_cls_loss_value > 0:
            metrics.log_scalar(
                "dis_cls_loss_value",
                dis_cls_loss_value / sample_size,
                sample_size,
                round=3,
            )
            if os.environ['LOCAL_RANK'] == '0':
                wandb.log({"dis_cls_loss_value": dis_cls_loss_value / sample_size})




        dis_reg_cross_loss_value = sum(
            log.get("dis_reg_cross_loss_value", 0) for log in logging_outputs
        )
        if dis_reg_cross_loss_value > 0:
            metrics.log_scalar(
                "dis_reg_cross_loss_value",
                dis_reg_cross_loss_value / sample_size,
                sample_size,
                round=3,
            )
            if os.environ['LOCAL_RANK'] == '0':
                wandb.log({"dis_reg_cross_loss_value": dis_reg_cross_loss_value / sample_size})


        mask_feat_loss = sum(
            log.get("mask_feat_loss", 0) for log in logging_outputs
        )
        if mask_feat_loss > 0:
            metrics.log_scalar(
                "mask_feat_loss",
                mask_feat_loss / sample_size,
                sample_size,
                round=3,
            )
            if os.environ['LOCAL_RANK'] == '0':
                wandb.log({"mask_feat_loss": mask_feat_loss / sample_size})

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    def cal_dist_loss(self, sample, dist, masked_tokens, target_key, normalize=False):
        dist_masked_tokens = masked_tokens
        masked_distance = dist[dist_masked_tokens, :]
        masked_distance_target = sample[target_key]["distance_target"][
            dist_masked_tokens
        ]
        non_pad_pos = masked_distance_target > 0
        if normalize:
            masked_distance_target = (
                masked_distance_target.float() - self.dist_mean
            ) / self.dist_std
        masked_dist_loss = F.smooth_l1_loss(
            masked_distance[non_pad_pos].view(-1).float(),
            masked_distance_target[non_pad_pos].view(-1),
            reduction="mean",
            beta=1.0,
        )
        return masked_dist_loss


@register_loss("unimol_infer")
class UniMolInferLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        input_key = "net_input"
        target_key = "target"
        src_tokens = sample[input_key]["src_tokens"].ne(self.padding_idx)
        (
            encoder_rep,
            encoder_pair_rep,
        ) = model(**sample[input_key], features_only=True)
        sample_size = sample[input_key]["src_tokens"].size(0)
        encoder_pair_rep_list = []
        for i in range(sample_size):  # rm padding token
            encoder_pair_rep_list.append(encoder_pair_rep[i][src_tokens[i], :][:, src_tokens[i]].data.cpu().numpy())
        logging_output = {
                "mol_repr_cls": encoder_rep[:, 0, :].data.cpu().numpy(),  # get cls token
                "pair_repr": encoder_pair_rep_list,
                "smi_name": sample[target_key]["smi_name"],
                "bsz": sample[input_key]["src_tokens"].size(0),
            }
        return 0, sample_size, logging_output
