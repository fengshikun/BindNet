# code from https://github.com/microsoft/Graphormer.git

import torch
import torch.nn as nn

from .graphormer_graph_encoder import GraphormerGraphEncoder, init_graphormer_params


# head 
class RobertaHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, num_tasks, regression=False):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        output_dim = num_tasks
        self.out_proj = nn.Linear(hidden_size, output_dim)
        self.regression = regression

        if not self.regression:
            self.norm = nn.LayerNorm(hidden_size)
            self.gelu = nn.GELU()
            # self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
            # self.norm2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x):
        x = self.dense(x)
        if self.regression:
            x = torch.relu(x)
        else:
            x = self.gelu(x)   
        if not self.regression:
            x = self.norm(x)
        x = self.out_proj(x)
        return x


class GraphormerEncoder(nn.Module):
    def __init__(self, args, task_info):
        super().__init__()
        self.max_nodes = args.max_nodes

        self.graph_encoder = GraphormerGraphEncoder(
            # < for graphormer
            num_atoms=args.num_atoms,
            num_in_degree=args.num_in_degree,
            num_out_degree=args.num_out_degree,
            num_edges=args.num_edges,
            num_spatial=args.num_spatial,
            num_edge_dis=args.num_edge_dis,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            # >
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            encoder_normalize_before=args.encoder_normalize_before,
            pre_layernorm=args.pre_layernorm,
            apply_graphormer_init=args.apply_graphormer_init,
            activation_fn=args.activation_fn,
        )

        self.share_input_output_embed = args.share_encoder_input_output_embed
        self.embed_out = None
        self.lm_output_learned_bias = None

        # Remove head is set to true during fine-tuning
        self.load_softmax = not getattr(args, "remove_head", False)

        self.masked_lm_pooler = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )

        self.lm_head_transform_weight = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        # self.activation_fn = utils.get_activation_fn(args.activation_fn)
        self.activation_fn = nn.GELU()
        self.layer_norm = nn.LayerNorm(args.encoder_embed_dim)

        self.lm_output_learned_bias = None
        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))

            if not self.share_input_output_embed:
                self.embed_out = nn.Linear(
                    args.encoder_embed_dim, args.num_classes, bias=False
                )
            else:
                raise NotImplementedError
        self.NodeEdge = task_info.NodeEdge
        self.mlabel_task = task_info.mask
        mlabel_task_vocab_size = task_info.mlabel_task_vocab_size
        self.react_indenti_task = task_info.indenti # binary classification
        self.finetune = task_info.finetune
        finetune_task_num = task_info.task_num

        if task_info.mask:
            self.mlabel_head = RobertaHead(args.encoder_embed_dim * 2, mlabel_task_vocab_size)
            self.mask_criterion = nn.CrossEntropyLoss()
        
        if task_info.NodeEdge:
            self.Node_head = RobertaHead(args.encoder_embed_dim * 2, 119)
            self.Edge_head = RobertaHead(args.encoder_embed_dim * 2, 4)
            self.mask_criterion = nn.CrossEntropyLoss()
        
        if task_info.indenti:
            self.indenti_head = RobertaHead(args.encoder_embed_dim * 2, 1)
            self.indenti_criterion = nn.BCELoss()

        
        if self.finetune:
            self.finetune_head = RobertaHead(args.encoder_embed_dim, finetune_task_num)
            self.finetune_criterion = nn.BCEWithLogitsLoss(reduction = "none")


        if getattr(args, "apply_graphormer_init", False):
            self.apply(init_graphormer_params)

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        if self.embed_out is not None:
            self.embed_out.reset_parameters()

    def forward(self, batched_data, perturb=None, masked_tokens=None, return_node_emb=False, **unused):
        inner_states, graph_rep = self.graph_encoder(
            batched_data,
            perturb=perturb,
        )

        x = inner_states[-1].transpose(0, 1)

        # project masked tokens only
        if masked_tokens is not None:
            raise NotImplementedError

        if return_node_emb:
            return self.get_reaction_node_embedding(x)

        # x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))
        if not self.finetune:
            x = self.get_reaction_node_embedding(x) # mask or indentify

        return_dict = {'mask_loss': 0, 'indenti_loss': 0, 'finetune_loss': 0, 'node_loss': 0, 'edge_loss': 0, 'mask_logits': None, 'indenti_logits': None}
        
        batch_size = x.size(0)
        if self.mlabel_task:
            mlabel_logits = self.mlabel_head(x)
            mlabel_node_logits = mlabel_logits[:,1:, :]
            logits_array = []
            mlabel_array = []
            for i in range(batch_size):
                mask_idx = batched_data['mask_atom_indices'][i]
                node_logits = mlabel_node_logits[i][mask_idx]
                logits_array.append(node_logits)
                mlabel = torch.tensor(batched_data['mlabes'][i])
                node_mlabel = mlabel[mask_idx]
                mlabel_array.append(node_mlabel)
            
            all_logits = torch.cat(logits_array, dim=0)
            all_labels = torch.cat(mlabel_array).to(all_logits.device)
            mask_loss = self.mask_criterion(all_logits, all_labels)
            return_dict['mask_loss'] = mask_loss
            return_dict['mask_logits'] = all_logits
            return_dict['mask_labels'] = all_labels
        
        if self.NodeEdge:
            logits = self.Node_head(x)
            node_logits = logits[:, 1:, :]
            logits_array = []
            node_tgt_array = []
            edge_tgt_array = []
            for i in range(batch_size):
                mask_idx = batched_data['mask_atom_indices'][i]
                node_logits = node_logits[i][mask_idx]
                logits_array.append(node_logits)
                
        if self.react_indenti_task:
            indenti_logits = self.indenti_head(x)
            indenti_logits = indenti_logits[:, 1:, :]

            logits_array = []
            ilabel_array = []
            for i in range(batch_size):
                primary_mol = (batched_data['molecule_idx'][i] == 1)
                primary_mol_len = primary_mol.sum()
                node_logits = indenti_logits[i][primary_mol]
                indenti_label = node_logits.new_zeros([primary_mol_len])
                indenti_label[batched_data['reaction_centre'][i]] = 1 # reaction centre
                logits_array.append(node_logits)
                ilabel_array.append(indenti_label)
            all_logits = torch.cat(logits_array, dim=0)
            all_labels = torch.cat(ilabel_array, dim=0).reshape((-1, 1))
            all_logits = torch.nn.Sigmoid()(all_logits)
            indenti_loss = self.indenti_criterion(all_logits, all_labels)
            return_dict['identi_loss'] = indenti_loss
            return_dict['identi_logits'] = all_logits
            return_dict['identi_labels'] = all_labels
            # print(indenti_loss)


        
        if self.finetune:
            self.finetune_head = RobertaHead(args.encoder_embed_dim, finetune_task_num)
            self.finetune_criterion = nn.BCEWithLogitsLoss(reduction = "none")
            # return_dict['finetune_loss'] = 
        return return_dict

        # project back to size of vocabulary
        if self.share_input_output_embed and hasattr(
            self.graph_encoder.embed_tokens, "weight"
        ):
            x = F.linear(x, self.graph_encoder.embed_tokens.weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)
        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias

        return x
    
    def get_reaction_node_embedding(self, node_embedding):
        batch_size = node_embedding.size(0)
        # batch_size x max_nodes_num x embeddings_size
        tensor_2_head = torch.zeros((node_embedding.size(0), node_embedding.size(1), node_embedding.size(2) * 2), dtype=node_embedding.dtype, device=node_embedding.device)
        # use the cls node as conditional embedding
        cls_emb = node_embedding[:, :1, :]
        tensor_2_head = torch.cat([node_embedding, torch.broadcast_to(cls_emb, node_embedding.shape)], dim=2)
        return tensor_2_head

    def max_nodes(self):
        """Maximum output length supported by the encoder."""
        return self.max_nodes

    def upgrade_state_dict_named(self, state_dict, name):
        if not self.load_softmax:
            for k in list(state_dict.keys()):
                if "embed_out.weight" in k or "lm_output_learned_bias" in k:
                    del state_dict[k]
        return state_dict
