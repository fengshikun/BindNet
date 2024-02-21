# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.modules import LayerNorm, init_bert_params
from .transformer_encoder_with_pair import TransformerEncoderWithPair
from unicore.modules import TransformerEncoderLayer
from typing import Dict, Any, List

from .graphormer import GraphormerGraphEncoder
from .torchmd_etf2d import TorchMD_ETF2D
from .output_modules import EquivariantScalar, Scalar
import yaml

logger = logging.getLogger(__name__)


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def define_gfmodel(model_config):

    graphormer_model = GraphormerGraphEncoder(
                # < for graphormer
                num_atoms=model_config.num_atoms,
                num_in_degree=model_config.num_in_degree,
                num_out_degree=model_config.num_out_degree,
                num_edges=model_config.num_edges,
                num_spatial=model_config.num_spatial,
                num_edge_dis=model_config.num_edge_dis,
                edge_type=model_config.edge_type,
                multi_hop_max_dist=model_config.multi_hop_max_dist,
                # >
                num_encoder_layers=model_config.encoder_layers,
                embedding_dim=model_config.encoder_embed_dim,
                ffn_embedding_dim=model_config.encoder_ffn_embed_dim,
                num_attention_heads=model_config.encoder_attention_heads,
                dropout=model_config.dropout,
                attention_dropout=model_config.attention_dropout,
                activation_dropout=model_config.act_dropout,
                encoder_normalize_before=model_config.encoder_normalize_before,
                pre_layernorm=model_config.pre_layernorm,
                apply_graphormer_init=model_config.apply_graphormer_init,
                activation_fn=model_config.activation_fn,
            )
    return graphormer_model


@register_model("unimol")
class UniMolModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--emb-dropout",
            type=float,
            metavar="D",
            help="dropout probability for embeddings",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-seq-len", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--post-ln", type=bool, help="use post layernorm or pre layernorm"
        )
        parser.add_argument(
            "--masked-token-loss",
            type=float,
            metavar="D",
            help="mask loss ratio",
        )
        parser.add_argument(
            "--masked-dist-loss",
            type=float,
            metavar="D",
            help="masked distance loss ratio",
        )
        parser.add_argument(
            "--masked-coord-loss",
            type=float,
            metavar="D",
            help="masked coord loss ratio",
        )
        parser.add_argument(
            "--x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--masked-coord-dist-loss",
            type=float,
            metavar="D",
            help="masked coord dist loss ratio",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "infer"],
        )
        parser.add_argument(
            "--ctl-2d",
            type=int,
            default=0,
            help='1: add 2d encoder for contrastive learning, 0: default settings',
        )

        # NOTE, complex pre-training related
        parser.add_argument(
            "--complex-pretrain",
            type=int,
            default=1,
            help='1: activate,0: deactivate',
        )

        parser.add_argument(
            "--pocket-data",
            type=str,
            default=None,
            help='',
        )
        parser.add_argument(
            "--idx-path",
            type=str,
            default=None,
            help='',
        )
        parser.add_argument(
            "--max-comnum",
            type=int,
            default=400,
            help='',
        )
        parser.add_argument(
            "--complex-layernum",
            type=int,
            default=3,
            help='',
        )
        parser.add_argument(
            "--dis-clsnum",
            type=int,
            default=61,
            help='',
        )
        parser.add_argument(
            "--online-ligfeat",
            type=int,
            default=0,
            help='',
        ) # if use ligand network to extract featrue

        parser.add_argument(
            "--lig-pretrained",
            type=str,
            default='',
            help=''
        ) # use ligand pretrained model

        parser.add_argument(
            "--proc-pretrained",
            type=str,
            default='',
            help=''
        ) # use protein pretrained model
        parser.add_argument(
            "--proc-freeze",
            type=int,
            default=1,
            help=''
        ) # freeze the pocket encoder


        parser.add_argument(
            "--complex-crnet",
            type=int,
            default=0,
            help='cross atten like unimol'
        ) # freeze the pocket encoder


        parser.add_argument(
            "--use-frad",
            type=int,
            default=0,
            help='cross atten like unimol'
        ) # freeze the pocket encoder


        parser.add_argument(
            "--cr-regression",
            type=int,
            default=0,
            help='regression like docking'
        ) # freeze the pocket encoder
        parser.add_argument(
            "--dist-threshold",
            type=float,
            default=8.0,
            help="threshold for the distance between the molecule and the pocket, valid for cr_regression",
        )

        parser.add_argument(
            "--rdkit-random",
            type=int,
            default=0,
            help='random for rdkit conformer generation'
        ) # freeze the pocket encoder


        parser.add_argument(
            "--regression-cls",
            type=int,
            default=0,
            help='regression like docking'
        )


        # NOTE for feature masking
        parser.add_argument(
            "--mask-ratio",
            type=float,
            default=0.8,
            help='random for rdkit conformer generation'
        )
        parser.add_argument(
            "--mask-feature",
            type=int,
            default=0,
            help='random for rdkit conformer generation'
        )
        parser.add_argument(
            "--mask-only",
            type=int,
            default=0,
            help='only mask feature'
        )

        parser.add_argument(
            "--recycling",
            type=int,
            default=3,
            help="recycling nums of decoder(unimol)",
        )

        # NOTE: 2d ctl
        parser.add_argument(
            "--gf-config",
            type=str,
            default='./unimol/models/graphormer/graphormer.yaml',
            help='add 2d encoder for contrastive learning',
        )

    def __init__(self, args, dictionary, lig_dictionary=None):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), args.encoder_embed_dim, self.padding_idx
        )


        self._num_updates = None
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.delta_pair_repr_norm_loss < 0,
        )

        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        if hasattr(self.args, 'online_ligfeat') and self.args.online_ligfeat:

            if hasattr(self.args, 'use_frad') and self.args.use_frad:
                # use frad as mol encoder
                shared_args ={'hidden_channels': 256, 'num_layers': 8, 'num_rbf': 64, 'rbf_type': 'expnorm', 'trainable_rbf': False, 'activation': 'silu', 'neighbor_embedding': True, 'cutoff_lower': 0.0, 'cutoff_upper': 5.0, 'max_z': 100, 'max_num_neighbors': 32}
                self.lig_encoder = TorchMD_ETF2D(
                    attn_activation="silu",
                    num_heads=8,
                    distance_influence="both",
                    layernorm_on_vec="whitened",
                    md17=False,
                    seperate_noise=False,
                    **shared_args
                )

                hidden_channels = shared_args['hidden_channels']
                self.proj_head = EquivariantScalar(hidden_channels, hidden_channels*2) # change 256 --> 512
                self.pairwise_head = Scalar(hidden_channels * 4, output_dims=64) # pair wise embeddings transfer
            else:
                self.lig_embed_tokens = nn.Embedding(
                    len(lig_dictionary), args.encoder_embed_dim, self.padding_idx
                ) # sampe padding
                self.lig_encoder = TransformerEncoderWithPair(
                    encoder_layers=args.encoder_layers,
                    embed_dim=args.encoder_embed_dim,
                    ffn_embed_dim=args.encoder_ffn_embed_dim,
                    attention_heads=args.encoder_attention_heads,
                    emb_dropout=args.emb_dropout,
                    dropout=args.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    max_seq_len=args.max_seq_len,
                    activation_fn=args.activation_fn,
                    no_final_head_layer_norm=args.delta_pair_repr_norm_loss < 0,
                ) # for encoder the ligand
                self.lig_gbf_proj = NonLinearHead(
                K, args.encoder_attention_heads, args.activation_fn
                )
                lig_n_edge_type = len(lig_dictionary) * len(lig_dictionary)
                self.lig_gbf = GaussianLayer(K, lig_n_edge_type)


        if args.masked_token_loss > 0:
            self.lm_head = MaskLMHead(
                embed_dim=args.encoder_embed_dim,
                output_dim=len(dictionary),
                activation_fn=args.activation_fn,
                weight=None,
            )

        if hasattr(self.args, 'ctl_2d') and self.args.ctl_2d:
            gf_config = self.args.gf_config
            with open(gf_config, 'r') as cr:
                gf_2d_config = yaml.safe_load(cr)

            gf_2d_config = Struct(**gf_2d_config)
            self.gf_encoder = define_gfmodel(gf_2d_config)
        # gf_config

        if hasattr(self.args, 'complex_pretrain') and self.args.complex_pretrain:
            if self.args.mask_feature:
                self.mask_token_embedding = nn.Embedding(1, args.encoder_embed_dim)

                output_dim = args.encoder_embed_dim
                if hasattr(self.args, 'use_frad') and self.args.use_frad:
                    output_dim = args.encoder_embed_dim // 2

                self.reconstruct_mask_feat_head = NonLinearHead(
                    args.encoder_embed_dim, output_dim, "relu"
                    )

            if self.args.complex_crnet: # like unimol
                self.concat_decoder = TransformerEncoderWithPair(
                    encoder_layers=4,
                    embed_dim=args.encoder_embed_dim,
                    ffn_embed_dim=args.encoder_ffn_embed_dim,
                    attention_heads=args.encoder_attention_heads,
                    emb_dropout=0.1,
                    dropout=0.1,
                    attention_dropout=0.1,
                    activation_dropout=0.0,
                    activation_fn="gelu",
                )

                if self.args.cr_regression:
                    if self.args.regression_cls:
                        self.cross_distance_project = NonLinearHead(
                        args.encoder_embed_dim * 2 + args.encoder_attention_heads, args.dis_clsnum, "relu"
                        )
                    else:
                        self.cross_distance_project = NonLinearHead(
                        args.encoder_embed_dim * 2 + args.encoder_attention_heads, 1, "relu"
                        )
                    self.holo_distance_project = DistanceHead(
                        args.encoder_embed_dim + args.encoder_attention_heads, "relu"
                    ) # NOTE: maybe exit no matter cr_regression or not
                else:
                    self.cls_dis_head = NonLinearHead(
                        args.encoder_embed_dim * 2 + args.encoder_attention_heads, args.dis_clsnum, "relu"
                    )
                # self.holo_distance_project = DistanceHead(
                #     args.encoder_embed_dim + args.encoder_attention_heads, "relu"
                # )


            else:
                self.complex_layernum = args.complex_layernum
                self.complex_layers = nn.ModuleList(
                    [
                        TransformerEncoderLayer(
                            embed_dim=args.encoder_embed_dim,
                            ffn_embed_dim=args.encoder_ffn_embed_dim,
                            attention_heads=args.encoder_attention_heads,
                            dropout=args.emb_dropout,
                            attention_dropout=args.dropout,
                            activation_dropout=args.attention_dropout,
                            activation_fn=args.activation_fn,
                            post_ln=False,
                        )
                        for _ in range(args.complex_layernum)
                    ]
                )
                self.cls_dis_head = NonLinearHead(args.encoder_embed_dim*2, args.dis_clsnum, args.activation_fn)

            # self.lig_gbf_proj = NonLinearHead(
            # K, args.encoder_attention_heads, args.activation_fn
            # )
            # self.lig_gbf = GaussianLayer(K, n_edge_type)




        self.gbf_proj = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type)

        if args.masked_coord_loss > 0:
            self.pair2coord_proj = NonLinearHead(
                args.encoder_attention_heads, 1, args.activation_fn
            )
        if args.masked_dist_loss > 0:
            self.dist_head = DistanceHead(
                args.encoder_attention_heads, args.activation_fn
            )
        self.classification_heads = nn.ModuleDict()
        self.apply(init_bert_params)

        if hasattr(self.args, 'online_ligfeat') and self.args.online_ligfeat:
            self.load_pretrained_model(self.args.lig_pretrained)
        if hasattr(self.args, 'complex_pretrain') and self.args.complex_pretrain:
            self.load_pocket_pretrained_model(self.args.proc_pretrained)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        if args.complex_pretrain:
            return cls(args, task.dictionary, task.lig_dictionary)
        return cls(args, task.dictionary)

    def load_pocket_pretrained_model(self, poc_pretrained):
        logger.info(f"Loading pocket pretraind weight from {poc_pretrained}")
        poc_state_dict = torch.load(poc_pretrained, map_location=lambda storage, loc: storage)
        missing_keys, not_matched_keys = self.load_state_dict(poc_state_dict['model'], strict=False)
        logging.info(f'loadding pocket model weight, missing_keys is {missing_keys}\n, not_matched_keys is {not_matched_keys}\n')
        filter_lig_keys = []
        for k in missing_keys:
            if not k.startswith('lig_'):
                filter_lig_keys.append(k)
        logging.info(f'loadding pocket model weight, filter lig weight missing_keys is {filter_lig_keys}\n')
        # freeze weight
        if self.args.proc_freeze:
            self.freeze_params(self.embed_tokens)
            self.freeze_params(self.gbf_proj)
            self.freeze_params(self.gbf)
            self.freeze_params(self.encoder)

    def load_pretrained_model(self, lig_pretrained):
        # load model parameter manually
        logger.info("Loading pretrained weights for ligand from {}".format(lig_pretrained))
        state_dict = torch.load(lig_pretrained, map_location=lambda storage, loc: storage)

        # load weight by hand
        if hasattr(self.args, 'use_frad') and self.args.use_frad:
            new_state_dict = {}
            for k, v in state_dict['state_dict'].items():
                if 'model.representation_model' in k:
                    new_k = k.replace('model.representation_model.', '')
                    new_state_dict[new_k] = v
            missing_keys, not_matched_keys = self.lig_encoder.load_state_dict(new_state_dict, strict=False)
            logging.info(f'loadding lig model weight, missing_keys is {missing_keys}\n, not_matched_keys is {not_matched_keys}\n')
            self.freeze_params(self.lig_encoder)
            return
        token_weight_dict = {'weight': state_dict['model']['embed_tokens.weight']}
        self.lig_embed_tokens.load_state_dict(token_weight_dict, strict=True)

        gbf_proj_weight_dict = {'linear1.weight': state_dict['model']['gbf_proj.linear1.weight'], 'linear1.bias': state_dict['model']['gbf_proj.linear1.bias'], 'linear2.weight': state_dict['model']['gbf_proj.linear2.weight'], 'linear2.bias' : state_dict['model']['gbf_proj.linear2.bias']}

        self.lig_gbf_proj.load_state_dict(gbf_proj_weight_dict, strict=True)

        gbf_weight_dict = {'means.weight': state_dict['model']['gbf.means.weight'], 'stds.weight': state_dict['model']['gbf.stds.weight'], 'mul.weight': state_dict['model']['gbf.mul.weight'], 'bias.weight': state_dict['model']['gbf.bias.weight']}
        self.lig_gbf.load_state_dict(gbf_weight_dict, strict=True)

        model_dict = {k.replace('encoder.',''):v for k, v in state_dict['model'].items()}
        missing_keys, not_matched_keys = self.lig_encoder.load_state_dict(model_dict, strict=False)
        logging.info(f'loadding lig model weight, missing_keys is {missing_keys}\n, not_matched_keys is {not_matched_keys}\n')
        # NOTE todo fix lig_encoder, fix and freeze
        # freeze weight
        # self.lig_embed_tokens.eval()
        # self.lig_gbf_proj.eval()
        # self.lig_gbf.eval()
        # self.lig_encoder.eval()
        self.freeze_params(self.lig_embed_tokens)
        self.freeze_params(self.lig_gbf_proj)
        self.freeze_params(self.lig_gbf)
        self.freeze_params(self.lig_encoder)

    def freeze_params(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def check_lig_eval(self):
        if self.lig_encoder.training:
            self.lig_encoder.eval()
        if hasattr(self.args, 'use_frad') and self.args.use_frad:
            return
        if self.lig_embed_tokens.training:
            self.lig_embed_tokens.eval()
        if self.lig_gbf_proj.training:
            self.lig_gbf_proj.eval()
        if self.lig_gbf.training:
            self.lig_gbf.eval()


    def check_pocket_eval(self):
        if self.embed_tokens.training:
            self.embed_tokens.eval()
        if self.gbf_proj.training:
            self.gbf_proj.eval()
        if self.gbf.training:
            self.gbf.eval()
        if self.encoder.training:
            self.encoder.eval()


    def crnet_forward(self, encoder_rep, encoder_pair_rep, prot_padding_mask, lig_encoder_rep, lig_graph_attn_bias, lig_padding_mask, lig_encoder_pair_rep):
        assert self.args.online_ligfeat == 1
        pocket_encoder_rep = encoder_rep
        pocket_encoder_pair_rep = encoder_pair_rep
        pocket_padding_mask = prot_padding_mask

        mol_encoder_rep = lig_encoder_rep
        mol_graph_attn_bias = lig_graph_attn_bias
        mol_padding_mask = lig_padding_mask
        mol_encoder_pair_rep = lig_encoder_pair_rep



        mol_sz = lig_encoder_rep.size(1)
        pocket_sz = pocket_encoder_rep.size(1)

        concat_rep = torch.cat(
            [mol_encoder_rep, pocket_encoder_rep], dim=-2
        )  # [batch, mol_sz+pocket_sz, hidden_dim]
        concat_mask = torch.cat(
            [mol_padding_mask, pocket_padding_mask], dim=-1
        )  # [batch, mol_sz+pocket_sz]
        attn_bs = mol_graph_attn_bias.size(0)

        concat_attn_bias = torch.zeros(
            attn_bs, mol_sz + pocket_sz, mol_sz + pocket_sz
        ).type_as(
            concat_rep
        )  # [batch, mol_sz+pocket_sz, mol_sz+pocket_sz]
        concat_attn_bias[:, :mol_sz, :mol_sz] = (
            mol_encoder_pair_rep.permute(0, 3, 1, 2)
            .reshape(-1, mol_sz, mol_sz)
            .contiguous()
        )
        concat_attn_bias[:, -pocket_sz:, -pocket_sz:] = (
            pocket_encoder_pair_rep.permute(0, 3, 1, 2)
            .reshape(-1, pocket_sz, pocket_sz)
            .contiguous()
        )

        decoder_rep = concat_rep
        decoder_pair_rep = concat_attn_bias
        for i in range(self.args.recycling):
            decoder_outputs = self.concat_decoder(
                decoder_rep, padding_mask=concat_mask, attn_mask=decoder_pair_rep
            )
            decoder_rep = decoder_outputs[0]
            decoder_pair_rep = decoder_outputs[1]
            if i != (self.args.recycling - 1):
                decoder_pair_rep = decoder_pair_rep.permute(0, 3, 1, 2).reshape(
                    -1, mol_sz + pocket_sz, mol_sz + pocket_sz
                )

        mol_decoder = decoder_rep[:, :mol_sz]
        pocket_decoder = decoder_rep[:, mol_sz:]

        return mol_decoder, pocket_decoder, decoder_pair_rep

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        mol_graph=None,
        lig_feat_input=None,
        lig_num_lst=None,
        prot_num_lst=None,
        # for molecular lig part
        lig_tokens=None,
        lig_distance=None,
        lig_coord=None,
        lig_edge_type=None,
        lig_org_distance=None,  # NOTE: for masking feature reconstruction
        feat_masking_idx=None, # NOTE: for masking feature recontruction
        frad_dataset=None, # frad dataset
        **kwargs
    ):


        if frad_dataset is not None:
            frad_dataset.to(src_tokens.device)

        if classification_head_name is not None:
            features_only = True

        padding_mask = src_tokens.eq(self.padding_idx)
        if self.args.complex_pretrain:

            self.check_lig_eval()
            if self.args.proc_freeze:
                self.check_pocket_eval()


            prot_padding_mask = padding_mask
            # lig_padding_mask = torch.zeros(lig_feat_input.shape[0], lig_feat_input.shape[1]).to(torch.bool)
            # for i, lig_n in enumerate(lig_num_lst):
            #     lig_padding_mask[i, lig_n.item():] = True
            # all_padding_mask = torch.cat([prot_padding_mask, lig_padding_mask.to(prot_padding_mask.device)], dim=1)


        if not padding_mask.any():
            padding_mask = None


        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        def get_dist_feature_lig(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.lig_gbf(dist, et)
            gbf_result = self.lig_gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        if self.args.complex_pretrain and self.args.proc_freeze:
            with torch.no_grad():
                x = self.embed_tokens(src_tokens)
                graph_attn_bias = get_dist_features(src_distance, src_edge_type)
                (
                    encoder_rep,
                    encoder_pair_rep,
                    delta_encoder_pair_rep,
                    x_norm,
                    delta_encoder_pair_rep_norm,
                ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        else:
            x = self.embed_tokens(src_tokens)
            graph_attn_bias = get_dist_features(src_distance, src_edge_type)
            (
                encoder_rep,
                encoder_pair_rep,
                delta_encoder_pair_rep,
                x_norm,
                delta_encoder_pair_rep_norm,
            ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)


        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0

        encoder_distance = None
        encoder_coord = None


        if hasattr(self.args, 'complex_pretrain') and self.args.complex_pretrain:
            if hasattr(self.args, 'online_ligfeat') and self.args.online_ligfeat:
                with torch.no_grad():

                    if hasattr(self.args, 'use_frad') and self.args.use_frad:
                        if '16' in str(x.dtype):
                            frad_dataset.pos = frad_dataset.pos.half()
                        xnew, vec, z, pos, batch = self.lig_encoder(frad_dataset.z, frad_dataset.pos, frad_dataset.batch)


                        # 256 --> 512 node embedding, special head

                        xnew_proj = self.proj_head.pre_reduce(xnew, vec) # update dims


                        x_feat_lst = []
                        batch_size = batch.max().item() + 1
                        feat_len_lst = []
                        for i in range(batch_size):
                            # insert a cls embedding, take mean of all
                            mol_embs = xnew_proj[batch==i]
                            mean_embs = mol_embs.mean(dim=0).reshape(1, -1)
                            mol_embs_final = torch.concat((mean_embs, mol_embs))
                            feat_len_lst.append(mol_embs_final.shape[0])
                            x_feat_lst.append(mol_embs_final)

                        # padding to the same length
                        max_len = max(feat_len_lst)
                        embed_size = xnew_proj.shape[1]
                        lig_encoder_rep = torch.zeros((batch_size, max_len, embed_size), device=xnew.device, dtype=xnew.dtype)
                        lig_padding_mask = torch.ones((batch_size, max_len), device=xnew.device, dtype=torch.bool)

                        for i in range(batch_size):
                            feat_len = feat_len_lst[i]
                            lig_encoder_rep[i, :feat_len, :] = x_feat_lst[i]
                            lig_padding_mask[i,:feat_len] = False


                        # pairwise embedding
                        # 512 --> concat???
                        expand_lig_a = lig_encoder_rep.unsqueeze(2)
                        expand_lig_b = lig_encoder_rep.unsqueeze(1)



                        repeat_lig_a = expand_lig_a.expand(batch_size, max_len, max_len, embed_size)
                        repeat_lig_b = expand_lig_b.expand(batch_size, max_len, max_len, embed_size)
                        lig_pair_wise = torch.cat((repeat_lig_a, repeat_lig_b), dim=3)
                        lig_encoder_pair_rep = self.pairwise_head.pre_reduce(lig_pair_wise) # get pairwise embedding

                        # print(lig_encoder_pair_rep.shape)

                        lig_graph_attn_bias = graph_attn_bias # only use attn_bs = mol_graph_attn_bias.size(0)

                    else:
                        # extract ligand feature online
                        lig_x = self.lig_embed_tokens(lig_tokens)
                        lig_graph_attn_bias = get_dist_feature_lig(lig_distance, lig_edge_type)
                        lig_padding_mask = lig_tokens.eq(self.padding_idx)
                        (
                            lig_encoder_rep,
                            lig_encoder_pair_rep,
                            lig_delta_encoder_pair_rep,
                            lig_x_norm,
                            lig_delta_encoder_pair_rep_norm,
                        ) = self.lig_encoder(lig_x, padding_mask=lig_padding_mask, attn_mask=lig_graph_attn_bias)
                    # concat the feat of lig and protein
                    # print('extract feature')



                    if lig_org_distance is not None:

                        if hasattr(self.args, 'use_frad') and self.args.use_frad:
                            if '16' in str(x.dtype):
                                frad_dataset.org_pos = frad_dataset.org_pos.half()
                            xnew_org, vec_org, z, pos, batch = self.lig_encoder(frad_dataset.z, frad_dataset.org_pos, frad_dataset.batch)
                            # xnew_org: regression target, embedding 256

                            # need to change the feat_masking_idx to adjust the shape same as lig_encoder_rep of frad

                            feat_masking_idx = feat_masking_idx.to(torch.bool)
                            feat_masking_idx = feat_masking_idx[:, :max_len] # no need for the multiple of 8 and the seq operator, only with the cls token at the first place, cutoff

                            # need to get the one dim of masking idx
                            feat_masking_idx_one_dim = torch.zeros((frad_dataset.org_pos.shape[0]), device=xnew.device, dtype=torch.bool)
                            start_idx = 0
                            for i in range(batch_size):
                                cur_len = feat_len_lst[i] - 1 # erase the cls
                                feat_masking_idx_one_dim[start_idx: start_idx+cur_len] = feat_masking_idx[i, 1:1+cur_len] # ommit the cls(start from 1)
                                start_idx += cur_len

                            lig_feature_reg_target = xnew_org[feat_masking_idx_one_dim]

                            lig_encoder_rep_unmask = lig_encoder_rep.clone()
                            lig_encoder_rep[feat_masking_idx] = self.mask_token_embedding(torch.tensor(0).to(lig_encoder_rep.device))

                        else:
                            lig_graph_attn_bias_org = get_dist_feature_lig(lig_org_distance, lig_edge_type)
                            (
                                lig_encoder_rep_org,
                                lig_encoder_pair_rep_org,
                                _,
                                _,
                                _,
                            ) = self.lig_encoder(lig_x, padding_mask=lig_padding_mask, attn_mask=lig_graph_attn_bias_org)

                            # replace masking feature
                            feat_masking_idx = feat_masking_idx.to(torch.bool)

                            lig_encoder_rep_unmask = lig_encoder_rep.clone()
                            lig_encoder_rep[feat_masking_idx] = self.mask_token_embedding(torch.tensor(0).to(lig_encoder_rep.device))

                            lig_feature_reg_target = lig_encoder_rep_org[feat_masking_idx]


                all_padding_mask = torch.cat([prot_padding_mask, lig_padding_mask], dim=1)
                # NOTE cls and sep for the ligand
                all_feat_concat = torch.cat([encoder_rep, lig_encoder_rep], dim=1) # fp16?
            else:
                all_feat_concat = torch.cat([encoder_rep, lig_feat_input.to(encoder_rep.dtype)], dim=1) # fp16?

            all_feat_x = all_feat_concat

            if self.args.complex_crnet:
                # all_padding_mask = torch.cat([lig_padding_mask, prot_padding_mask], dim=1)
                concat_mask = torch.cat(
                        [lig_padding_mask, prot_padding_mask], dim=-1
                    )
                if self.args.mask_feature:
                    mask_mol_decoder, _, _ = self.crnet_forward(encoder_rep, encoder_pair_rep, prot_padding_mask, lig_encoder_rep, lig_graph_attn_bias, lig_padding_mask, lig_encoder_pair_rep)
                    reconstruct_feat = self.reconstruct_mask_feat_head(mask_mol_decoder[feat_masking_idx])

                    mol_decoder, pocket_decoder, decoder_pair_rep = self.crnet_forward(encoder_rep, encoder_pair_rep, prot_padding_mask, lig_encoder_rep_unmask, lig_graph_attn_bias, lig_padding_mask, lig_encoder_pair_rep)

                else:
                    mol_decoder, pocket_decoder, decoder_pair_rep = self.crnet_forward(encoder_rep, encoder_pair_rep, prot_padding_mask, lig_encoder_rep, lig_graph_attn_bias, lig_padding_mask, lig_encoder_pair_rep)

                mol_sz = mol_decoder.size(1)
                pocket_sz = pocket_decoder.size(1)




                mol_pair_decoder_rep = decoder_pair_rep[:, :mol_sz, :mol_sz, :]
                mol_pocket_pair_decoder_rep = (
                    decoder_pair_rep[:, :mol_sz, mol_sz:, :]
                    + decoder_pair_rep[:, mol_sz:, :mol_sz, :].transpose(1, 2)
                ) / 2.0
                mol_pocket_pair_decoder_rep[mol_pocket_pair_decoder_rep == float("-inf")] = 0

                cross_rep = torch.cat(
                    [
                        mol_pocket_pair_decoder_rep,
                        mol_decoder.unsqueeze(-2).repeat(1, 1, pocket_sz, 1),
                        pocket_decoder.unsqueeze(-3).repeat(1, mol_sz, 1, 1),
                    ],
                    dim=-1,
                )  # [batch, mol_sz, pocket_sz, 4*hidden_size]


                if self.args.cr_regression:
                    # directly regress the dist matrix
                    if self.args.regression_cls:
                        cross_distance_predict = self.cross_distance_project(cross_rep)
                        pass
                    else:
                        cross_distance_predict = (
                        F.elu(self.cross_distance_project(cross_rep).squeeze(-1)) + 1.0
                        )  # batch, mol_sz, pocket_sz

                   
                    dis_cls_logits = cross_distance_predict# regression target
                else:
                    dis_cls_logits_matrix = self.cls_dis_head(cross_rep)
                    # collect logits:
                    batch_size = dis_cls_logits_matrix.shape[0]
                    logits_ele_lst = []
                    for s_idx in range(batch_size):
                        filter_idx = ~concat_mask[s_idx] # same as all_padding_mask
                        proc_num = prot_num_lst[s_idx].item()
                        lig_num = lig_num_lst[s_idx].item()
                        logits_ele = dis_cls_logits_matrix[s_idx]
                        logits_ele = logits_ele[1:1+lig_num, 1:1+proc_num, :]
                        # NOTE: lig_num * proc_num --> proc_num * lig_num
                        logits_ele = logits_ele.transpose(0, 1)
                        logits_ele = logits_ele.reshape(-1, logits_ele.shape[-1])

                        logits_ele_lst.append(logits_ele)
                    dis_cls_logits = torch.cat(logits_ele_lst, dim=0)


            else:
                for i in range(len(self.complex_layers)):
                    all_feat_x = self.complex_layers[i](
                        all_feat_x, padding_mask=all_padding_mask
                    )


                # get predict values
                # collect feats of proc and ligand
                batch_size = all_feat_x.shape[0]
                proc_lig_feat_lst = []
                for s_idx in range(batch_size):
                    filter_idx = ~all_padding_mask[s_idx]
                    proc_lig_feat = all_feat_x[s_idx][filter_idx]
                    proc_num = prot_num_lst[s_idx].item()
                    lig_num = lig_num_lst[s_idx].item()
                    proc_feat = proc_lig_feat[1:1+proc_num, :]
                    if self.args.online_ligfeat:
                        lig_feat = proc_lig_feat[3+proc_num:3+proc_num+lig_num, :]  # cls,proc_feat,<sep>; cls,lig_feat,<sep>
                    else:
                        lig_feat = proc_lig_feat[2+proc_num:, :]  # cls,proc_feat,<sep>,lig_feat

                    proc_feat_repeat = proc_feat.unsqueeze(1).repeat(1, lig_num, 1)
                    lig_feat_repeat = lig_feat.repeat(proc_num, 1, 1)

                    mix_concat_matrix = torch.cat([proc_feat_repeat, lig_feat_repeat], dim=2)  # Shape: (M, N, 1024)
                    mix_concat_matrix_one = mix_concat_matrix.reshape(-1, mix_concat_matrix.shape[-1])
                    proc_lig_feat_lst.append(mix_concat_matrix_one)

                # get logits
                proc_lig_feat_lst = torch.cat(proc_lig_feat_lst, dim=0)
                dis_cls_logits = self.cls_dis_head(proc_lig_feat_lst)

            if self.args.mask_feature:
                return all_feat_x, all_padding_mask, dis_cls_logits, x_norm, delta_encoder_pair_rep_norm, (reconstruct_feat, lig_feature_reg_target)

            return all_feat_x, all_padding_mask, dis_cls_logits, x_norm, delta_encoder_pair_rep_norm,

        if hasattr(self.args, 'ctl_2d') and self.args.ctl_2d:
            mol_2d_rep = self.gf_encoder(mol_graph)
            inner_state, graph_2d_rep = mol_2d_rep
            graph_3d_rep = encoder_rep[:,0,:] # cls token
            # contrastive learning for 2d and 3d
            # print("infer 2d")

        if not features_only:
            if self.args.masked_token_loss > 0:
                logits = self.lm_head(encoder_rep, encoder_masked_tokens)
            if self.args.masked_coord_loss > 0:
                coords_emb = src_coord
                if padding_mask is not None:
                    atom_num = (torch.sum(1 - padding_mask.type_as(x), dim=1) - 1).view(
                        -1, 1, 1, 1
                    )
                else:
                    atom_num = src_coord.shape[1] - 1
                delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
                attn_probs = self.pair2coord_proj(delta_encoder_pair_rep)
                coord_update = delta_pos / atom_num * attn_probs
                coord_update = torch.sum(coord_update, dim=2)
                encoder_coord = coords_emb + coord_update
            if self.args.masked_dist_loss > 0:
                encoder_distance = self.dist_head(encoder_pair_rep)

        if classification_head_name is not None:
            logits = self.classification_heads[classification_head_name](encoder_rep)
        if self.args.mode == 'infer':
            return encoder_rep, encoder_pair_rep
        elif self.args.ctl_2d:
            return (
                logits,
                encoder_distance,
                encoder_coord,
                x_norm,
                delta_encoder_pair_rep_norm,
                graph_2d_rep,
                graph_3d_rep,
            )
        else:
            return (
                logits,
                encoder_distance,
                encoder_coord,
                x_norm,
                delta_encoder_pair_rep_norm,
            )

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


class MaskLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        x[x == float('-inf')] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


@register_model_architecture("unimol", "unimol")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 15)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)


@register_model_architecture("unimol", "unimol_base")
def unimol_base_architecture(args):
    base_architecture(args)
