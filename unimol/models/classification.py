import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from .transformer_encoder_with_pair import TransformerEncoderWithPair
from unicore.modules import TransformerEncoderLayer
from unicore.modules import LayerNorm, init_bert_params
from .unimol import base_architecture
from unicore import utils
import lmdb
import os
import pickle as pkl

logger = logging.getLogger(__name__)

@register_model("classification")
class ClassificationModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--target-class",
            type=int,
            default=2,
            help='how many class to classify',
        )
        parser.add_argument(
            "--aff_finetune",
            type=int,
            default=0,
            help='1: activate,0: deactivate',
        )
        parser.add_argument(
            "--LBA_data",
            type=str,
            default="/home/admin01/limh/LBADATA/lba",
            help='',
        )
        parser.add_argument(
            "--DUDE_data",
            type=str,
            default="",
            help='',
        )
        parser.add_argument(
            "--DUDE_fold",
            type=int,
            default=0,
            help='',
        )
        parser.add_argument(
            "--log_folder",
            type=str,
            default=None,
            help='',
        )
        parser.add_argument(
            "--LEP_data",
            type=str,
            default=None,
            help='',
        )
        parser.add_argument(
            "--complex-pretrained-model",
            type=str,
            default='',
            help=''
        ) # use complex pretrained model
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
            "--ligand-freeze",
            type=int,
            default=1,
            help=''
        ) # freeze the ligand encoder
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
            "--net",
            choices = ['complex_crnet', 'old_transformer'],
            default='complex_crnet',
            help='Use cross atten like unimol or old transformer'
        )
        parser.add_argument(
            "--recycling",
            type=int,
            default=3,
            help="recycling nums of decoder(unimol)",
        )
        parser.add_argument(
            "--CLS-use",
            choices = ['complex_CLS', 'seperate_CLS'],
            default='complex_CLS',
            help='Input which kind of CLS to regression head'
        )
        parser.add_argument(
            "--complex-layernum",
            type=int,
            default=3,
            help='The transformer layer when use tranformer to concat protein ligand information',
        )
        parser.add_argument(
            "--freeze-pretrained-transformer",
            type=int,
            default=0,
            help='if freeze the pretrained transformer',
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "infer"],
        )
        parser.add_argument(
            "--all-test",
            type=int,
            default=0,
            help='Use in DUDE, if test all data',
        )
        parser.add_argument(
            "--use-frad",
            type=int,
            default=0,
            help='use frad as ligand encoder'
        )

        parser.add_argument(
            "--use-esm",
            type=int,
            default=0,
            help='use esmfold as protein encoder'
        )

        parser.add_argument(
            "--use-BindNet",
            type=int,
            default=1,
            help='use BindNet to fuse information'
        )

        parser.add_argument(
            "--random-choose-residue",
            type=int,
            default=0,
            help='random choose residue if residue num + ligand num > max_atom_num'
        )



    def __init__(self, args, dictionary, lig_dictionary=None):
        super().__init__()
        affinity_regres_architecture(args)
        self.args = args
        self.padding_idx = dictionary.pad()    # record the padding token
        self._num_updates = None
        K = 128

        # Protein Encoder
        self.embed_tokens = nn.Embedding(
            len(dictionary), args.encoder_embed_dim, self.padding_idx
        )
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
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type)

        # Ligand Encoder
        self.lig_embed_tokens = nn.Embedding(
            len(lig_dictionary), args.encoder_embed_dim, self.padding_idx
        )
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
        )
        lig_n_edge_type = len(lig_dictionary) * len(lig_dictionary)
        self.lig_gbf_proj = NonLinearHead(
        K, args.encoder_attention_heads, args.activation_fn
        )
        self.lig_gbf = GaussianLayer(K, lig_n_edge_type)

        # Transformer
        if self.args.net == 'complex_crnet': # like unimol
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

        elif self.args.net == 'old_transformer':
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
        else:
            raise KeyError("choose net from complex_crnet and old_transformer")



        # Classification Head
        if self.args.CLS_use == "complex_CLS":
            self.classification_head = NonLinearHead(args.encoder_embed_dim, args.target_class, args.activation_fn)
        elif self.args.CLS_use == "seperate_CLS":
            self.classification_head = NonLinearHead(args.encoder_embed_dim * 2, args.target_class, args.activation_fn)
        else:
            raise KeyError("choose CLS_use from complex_CLS and seperate_CLS")

        # Load Model
        self.apply(init_bert_params)
        # If use BindNet, load the whole model
        if hasattr(self.args, 'use_BindNet') and self.args.use_BindNet:
            self.load_complex_retrained_model(self.args.complex_pretrained_model)
            if self.args.freeze_pretrained_transformer > 0:
                self.freeze_params(self.concat_decoder)
        # If not, just load two encoder
        else:
            self.load_ligand_pretrained_model(self.args.lig_pretrained)
            if not (hasattr(self.args, 'use_esm') and self.args.use_esm):
                self.load_pocket_pretrained_model(self.args.complex_pretrained_model)


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary, task.lig_dictionary)

    def load_complex_retrained_model(self, complex_pretrained):
        logger.info(f"Loading complex pretraind weight from {complex_pretrained}")
        complex_state_dict = torch.load(complex_pretrained, map_location=lambda storage, loc: storage)
        missing_keys, not_matched_keys = self.load_state_dict(complex_state_dict['model'], strict=False)
        logging.info(f'loadding complex model weight, missing_keys is {missing_keys}\n, not_matched_keys is {not_matched_keys}\n')
        # freeze weight
        self.freeze_params(self.embed_tokens)
        self.freeze_params(self.gbf_proj)
        self.freeze_params(self.gbf)
        self.freeze_params(self.encoder)
        self.freeze_params(self.lig_embed_tokens)
        self.freeze_params(self.lig_gbf_proj)
        self.freeze_params(self.lig_gbf)
        self.freeze_params(self.lig_encoder)

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

    def load_ligand_pretrained_model(self, lig_pretrained):
        # load model parameter manually
        logger.info("Loading pretrained weights for ligand from {}".format(lig_pretrained))
        state_dict = torch.load(lig_pretrained, map_location=lambda storage, loc: storage)
        # load weight by hand
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
        if self.args.ligand_freeze:
            self.freeze_params(self.lig_embed_tokens)
            self.freeze_params(self.lig_gbf_proj)
            self.freeze_params(self.lig_gbf)
            self.freeze_params(self.lig_encoder)

    def freeze_params(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def check_lig_eval(self):
        if self.lig_embed_tokens.training:
            self.lig_embed_tokens.eval()
        if self.lig_gbf_proj.training:
            self.lig_gbf_proj.eval()
        if self.lig_gbf.training:
            self.lig_gbf.eval()
        if self.lig_encoder.training:
            self.lig_encoder.eval()


    def check_pocket_eval(self):
        if self.embed_tokens.training:
            self.embed_tokens.eval()
        if self.gbf_proj.training:
            self.gbf_proj.eval()
        if self.gbf.training:
            self.gbf.eval()
        if self.encoder.training:
            self.encoder.eval()


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
        idx = None,
        **kwargs
    ):
        # If freeze the ligand encoder or pocket encoder
        if self.args.ligand_freeze:
            self.check_lig_eval()
        if self.args.proc_freeze:
            self.check_pocket_eval()

        # Define the function to get the Gaussian distance feature
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

        # Protein Encoder Calculation
        prot_padding_mask = src_tokens.eq(self.padding_idx)    # Recond the padding token
        if self.args.proc_freeze:
            with torch.no_grad():
                x = self.embed_tokens(src_tokens)
                graph_attn_bias = get_dist_features(src_distance, src_edge_type)
                (
                    encoder_rep,
                    encoder_pair_rep,
                    delta_encoder_pair_rep,
                    x_norm,
                    delta_encoder_pair_rep_norm,
                ) = self.encoder(x, padding_mask=prot_padding_mask, attn_mask=graph_attn_bias)
        else:
            x = self.embed_tokens(src_tokens)
            graph_attn_bias = get_dist_features(src_distance, src_edge_type)
            (
                encoder_rep,
                encoder_pair_rep,
                delta_encoder_pair_rep,
                x_norm,
                delta_encoder_pair_rep_norm,
            ) = self.encoder(x, padding_mask=prot_padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0

        # Ligand Encoder Calculation
        lig_padding_mask = lig_tokens.eq(self.padding_idx)
        if self.args.ligand_freeze:
            with torch.no_grad():
                lig_x = self.lig_embed_tokens(lig_tokens)
                lig_graph_attn_bias = get_dist_feature_lig(lig_distance, lig_edge_type)
                (
                    lig_encoder_rep,
                    lig_encoder_pair_rep,
                    lig_delta_encoder_pair_rep,
                    lig_x_norm,
                    lig_delta_encoder_pair_rep_norm,
                ) = self.lig_encoder(lig_x, padding_mask=lig_padding_mask, attn_mask=lig_graph_attn_bias)
        else:
            lig_x = self.lig_embed_tokens(lig_tokens)
            lig_graph_attn_bias = get_dist_feature_lig(lig_distance, lig_edge_type)
            (
                lig_encoder_rep,
                lig_encoder_pair_rep,
                lig_delta_encoder_pair_rep,
                lig_x_norm,
                lig_delta_encoder_pair_rep_norm,
            ) = self.lig_encoder(lig_x, padding_mask=lig_padding_mask, attn_mask=lig_graph_attn_bias)


        # Transformer Calculation
        if self.args.net == 'complex_crnet': # like unimol:
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
                decoder_outputs = self.concat_decoder(decoder_rep, padding_mask=concat_mask, attn_mask=decoder_pair_rep)
                decoder_rep = decoder_outputs[0]
                decoder_pair_rep = decoder_outputs[1]
                if i != (self.args.recycling - 1):
                    decoder_pair_rep = decoder_pair_rep.permute(0, 3, 1, 2).reshape(
                        -1, mol_sz + pocket_sz, mol_sz + pocket_sz
                    )
            mol_decoder = decoder_rep[:, :mol_sz]
            pocket_decoder = decoder_rep[:, mol_sz:]
            # Affinity Regression Head Calculation
            if self.args.CLS_use == "complex_CLS":
                cls_token = decoder_rep[:, 0, :]
            elif self.args.CLS_use == "seperate_CLS":
                protein_cls_token = pocket_decoder[:, 0, :]
                ligand_cls_token = mol_decoder[:, 0, :]
                cls_token = torch.cat([protein_cls_token, ligand_cls_token], dim=-1)
            else:
                raise KeyError("choose CLS_use from complex_CLS and seperate_CLS")
            affinity_predict = self.classification_head(cls_token).squeeze(-1)
        elif self.args.net == 'old_transformer':
            mol_sz = lig_encoder_rep.size(1)
            pocket_sz = encoder_rep.size(1)
            all_padding_mask = torch.cat([prot_padding_mask, lig_padding_mask], dim=1)
            all_feat_x = torch.cat([encoder_rep, lig_encoder_rep], dim=1)
            for i in range(len(self.complex_layers)):
                all_feat_x = self.complex_layers[i](
                    all_feat_x, padding_mask=all_padding_mask
                )
            # Affinity Regression Head Calculation
            if self.args.CLS_use == "complex_CLS":
                cls_token = all_feat_x[:, 0, :]
            elif self.args.CLS_use == "seperate_CLS":
                pocket_decoder = all_feat_x[:, mol_sz:]
                mol_decoder = all_feat_x[:, :mol_sz]
                protein_cls_token = pocket_decoder[:, 0, :]
                ligand_cls_token = mol_decoder[:, 0, :]
                cls_token = torch.cat([protein_cls_token, ligand_cls_token], dim=-1)
            else:
                raise KeyError("choose CLS_use from complex_CLS and seperate_CLS")
            affinity_predict = self.classification_head(cls_token).squeeze(-1)
        else:
            raise KeyError("choose net from complex_crnet and old_transformer")

        return affinity_predict, x_norm

@torch.jit.script
def gaussian(x, mean, std):
    '''Gaussian function with specific mean and std'''
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianLayer(nn.Module):
    '''Gaussian layer for distance feature extraction'''
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


class NonLinearHead(nn.Module):
    """Head for simple classification or Regression tasks."""
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


@register_model_architecture("classification", "classification")
def affinity_regres_architecture(args):
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
    args.masked_coord_loss = getattr(args, "masked_coord_loss", 1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", 1.0)

    base_architecture(args)