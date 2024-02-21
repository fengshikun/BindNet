# graphormer assets

import torch
import pyximport
from torch_geometric.datasets import *

import numpy as np
import torch.distributed as dist
import copy
from functools import lru_cache
from torch_geometric.data import Dataset
from sklearn.model_selection import train_test_split
from unicore.data import BaseWrapperDataset
from .data_utils import mol_to_graph_data_obj_simple

pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import algos


@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item, max_node=0):
    if max_node > 0 and item.x.size(0) > max_node:
        print(f'Prune element shape: {item.x.size(0)} exceeds max_node : {max_node}')
        item.x = item.x[:max_node, :]
        edge_mask = torch.logical_or(item.edge_index[0,:] >= max_node, item.edge_index[1,:] >= max_node)
        item.edge_index = item.edge_index[:, ~edge_mask]
        item.edge_attr = item.edge_attr[~edge_mask, :]
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x


    N = x.size(0)
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr) + 1
    )

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree  # for undirected graph
    item.edge_input = torch.from_numpy(edge_input).long()

    

    return item




def preprocess_react_item(item):
    graph = item['graph']
    edge_attr, edge_index, x = graph.edge_attr, graph.edge_index, graph.x
    N = x.size(0)
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr) + 1
    )

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    graph.x = x
    graph.attn_bias = attn_bias
    graph.attn_edge_type = attn_edge_type
    graph.spatial_pos = spatial_pos
    graph.in_degree = adj.long().sum(dim=1).view(-1)
    graph.out_degree = graph.in_degree  # for undirected graph
    graph.edge_input = torch.from_numpy(edge_input).long()
    
    graph.idx_list = item['idx_list']
    graph.molecule_idx = item['molecule_idx']
    graph.reaction_centre = item['reaction_centre']
    graph.one_hop = item['one_hop']
    graph.two_hop = item['two_hop']
    graph.three_hop = item['three_hop']
    graph.mlabes = item['mlabes']

    return graph


class MyZINC(ZINC):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).process()
        if dist.is_initialized():
            dist.barrier()

class GraphormerPYGDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        seed: int = 0,
        train_idx=None,
        valid_idx=None,
        test_idx=None,
        train_set=None,
        valid_set=None,
        test_set=None,
    ):
        self.dataset = dataset
        if self.dataset is not None:
            self.num_data = len(self.dataset)
        self.seed = seed
        if train_idx is None and train_set is None:
            train_valid_idx, test_idx = train_test_split(
                np.arange(self.num_data),
                test_size=self.num_data // 10,
                random_state=seed,
            )
            train_idx, valid_idx = train_test_split(
                train_valid_idx, test_size=self.num_data // 5, random_state=seed
            )
            self.train_idx = torch.from_numpy(train_idx)
            self.valid_idx = torch.from_numpy(valid_idx)
            self.test_idx = torch.from_numpy(test_idx)
            self.train_data = self.index_select(self.train_idx)
            self.valid_data = self.index_select(self.valid_idx)
            self.test_data = self.index_select(self.test_idx)
        elif train_set is not None:
            self.num_data = len(train_set) + len(valid_set) + len(test_set)
            self.train_data = self.create_subset(train_set)
            self.valid_data = self.create_subset(valid_set)
            self.test_data = self.create_subset(test_set)
            self.train_idx = None
            self.valid_idx = None
            self.test_idx = None
        else:
            self.num_data = len(train_idx) + len(valid_idx) + len(test_idx)
            self.train_idx = train_idx
            self.valid_idx = valid_idx
            self.test_idx = test_idx
            self.train_data = self.index_select(self.train_idx)
            self.valid_data = self.index_select(self.valid_idx)
            self.test_data = self.index_select(self.test_idx)
        self.__indices__ = None

    def index_select(self, idx):
        dataset = copy.copy(self)
        dataset.dataset = self.dataset.index_select(idx)
        if isinstance(idx, torch.Tensor):
            dataset.num_data = idx.size(0)
        else:
            dataset.num_data = idx.shape[0]
        dataset.__indices__ = idx
        dataset.train_data = None
        dataset.valid_data = None
        dataset.test_data = None
        dataset.train_idx = None
        dataset.valid_idx = None
        dataset.test_idx = None
        return dataset

    def create_subset(self, subset):
        dataset = copy.copy(self)
        dataset.dataset = subset
        dataset.num_data = len(subset)
        dataset.__indices__ = None
        dataset.train_data = None
        dataset.valid_data = None
        dataset.test_data = None
        dataset.train_idx = None
        dataset.valid_idx = None
        dataset.test_idx = None
        return dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.dataset[idx]
            item.idx = idx
            item.y = item.y.reshape(-1)
            return preprocess_item(item)
        else:
            raise TypeError("index to a GraphormerPYGDataset can only be an integer.")

    def __len__(self):
        return self.num_data



class Convert2DDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        convert_func=mol_to_graph_data_obj_simple,
        max_node=128
    ):
        self.dataset = dataset
        self.convert_func = convert_func
        self.max_node = max_node

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        # for low rdkit version
        sample = self.dataset[index]
        graph_obj = self.convert_func(sample)
        graph_obj.idx = index
        data = preprocess_item(graph_obj, max_node=self.max_node)
        return data




def pad_1d_unsqueeze(x, padlen, pad_value=0):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:] = pad_value
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [
        (
            item.idx,
            item.edge_index,
            item.edge_attr,
            item.connected_edge_indices,
            item.attn_bias,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_input[:, :, :multi_hop_max_dist, :],
            # item.y,
            item.mask_node_label,
            item.mask_edge_label,
            item.masked_atom_indices,
            item.mlabes,
            item.reaction_centre,
            item.molecule_idx,
        )
        for item in items
    ]
    (
        idxs,
        edge_index,
        edge_attr,
        connected_edge_indices,
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_inputs,
        mask_node_label,
        mask_edge_label,
        mask_atom_indices,
        mlabes,
        reaction_centre,
        molecule_idx
    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
    batch_node_num_lst = torch.tensor([i.size(0) for i in xs])
    max_node_num = max(batch_node_num_lst)
    max_dist = max(i.size(-2) for i in edge_inputs)
    # y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    )
    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
    )
    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])

    # mask_atom_indices = torch.cat([pad_1d_unsqueeze(i, max_node_num, pad_value=-1) for i in mask_atom_indices])
    # reaction_centre = torch.cat([pad_1d_unsqueeze(torch.tensor(i), max_node_num, pad_value=-1) for i in reaction_centre])
    molecule_idx = torch.cat([pad_1d_unsqueeze(torch.tensor(i), max_node_num, pad_value=-1) for i in molecule_idx])

    return dict(
        idx=torch.LongTensor(idxs),
        edge_index=edge_index,
        edge_attr=edge_attr,
        connected_edge_indices=connected_edge_indices,
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        x=x,
        edge_input=edge_input,
        # y=y,
        mask_node_label=mask_node_label,
        mask_edge_label=mask_edge_label,
        batch_node_num_lst=batch_node_num_lst,
        mask_atom_indices = mask_atom_indices,
        reaction_centre = reaction_centre,
        molecule_idx = molecule_idx,
        mlabes = mlabes
    )



class BatchedDataDataset(Dataset):
    def __init__(
        self, dataset, max_node=128, multi_hop_max_dist=5, spatial_pos_max=1024
    ):
        super().__init__()
        self.dataset = dataset
        self.max_node = max_node
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max

    def __getitem__(self, index):
        item = self.dataset[int(index)]
        return item

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return collator(
            samples,
            max_node=self.max_node,
            multi_hop_max_dist=self.multi_hop_max_dist,
            spatial_pos_max=self.spatial_pos_max,
        )

def collator_finetune(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20, cliff=True):
    
    max_num = 0
    for item in items:
        if item.x.size(0) > max_node:
            max_num += 1

    if max_num > 0:
        print(f'filter number is {max_num}')
            
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    # if len(items) != 8:
    #     print(items)
    #     print([item.x.size(0) for item in items])
    if len(items) == 0:
        return None
    if cliff:
        # print("checks")
        
        # print("batch_length", len(items[0]))
        items = [
            (
                item.idx,
                item.attn_bias,
                item.attn_edge_type,
                item.spatial_pos,
                item.in_degree,
                item.out_degree,
                item.x,
                item.edge_input[:, :, :multi_hop_max_dist, :],
                item.y,
                item.cliff
            )
            for item in items
        ]
        
        (
            idxs,
            attn_biases,
            attn_edge_types,
            spatial_poses,
            in_degrees,
            out_degrees,
            xs,
            edge_inputs,
            ys,
            cliffs,
        ) = zip(*items)

        for idx, _ in enumerate(attn_biases):
            attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
        max_node_num = max(i.size(0) for i in xs)
        max_dist = max(i.size(-2) for i in edge_inputs)
        y = torch.cat(ys)
        x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
        edge_input = torch.cat(
            [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
        )
        attn_bias = torch.cat(
            [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
        )
        attn_edge_type = torch.cat(
            [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
        )
        spatial_pos = torch.cat(
            [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
        )
        in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])

        return dict(
            idx=torch.LongTensor(idxs),
            attn_bias=attn_bias,
            attn_edge_type=attn_edge_type,
            spatial_pos=spatial_pos,
            in_degree=in_degree,
            out_degree=in_degree,  # for undirected graph
            x=x,
            edge_input=edge_input,
            y=y,
            cliff=cliffs
        )
        
    else:
        items = [
            (
                item.idx,
                item.attn_bias,
                item.attn_edge_type,
                item.spatial_pos,
                item.in_degree,
                item.out_degree,
                item.x,
                item.edge_input[:, :, :multi_hop_max_dist, :],
                item.y,
            )
            for item in items
        ]
        (
            idxs,
            attn_biases,
            attn_edge_types,
            spatial_poses,
            in_degrees,
            out_degrees,
            xs,
            edge_inputs,
            ys,
        ) = zip(*items)

        for idx, _ in enumerate(attn_biases):
            attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
        max_node_num = max(i.size(0) for i in xs)
        max_dist = max(i.size(-2) for i in edge_inputs)
        if (ys[0].shape[0] > 1): # multitask
            ys = [y.reshape(1, -1) for y in ys]
            y = torch.cat(ys)
        else:
            y = torch.cat(ys)
        x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
        edge_input = torch.cat(
            [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
        )
        attn_bias = torch.cat(
            [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
        )
        attn_edge_type = torch.cat(
            [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
        )
        spatial_pos = torch.cat(
            [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
        )
        in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])

        return dict(
            idx=torch.LongTensor(idxs),
            attn_bias=attn_bias,
            attn_edge_type=attn_edge_type,
            spatial_pos=spatial_pos,
            in_degree=in_degree,
            out_degree=in_degree,  # for undirected graph
            x=x,
            edge_input=edge_input,
            y=y,
        )
    


def collator_pretrain(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    
    max_num = 0
    for item in items:
        if item.x.size(0) > max_node:
            max_num += 1

    if max_num > 0:
        print(f'filter number is {max_num}')
            
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    # if len(items) != 8:
    #     print(items)
    #     print([item.x.size(0) for item in items])
    if len(items) == 0:
        return None
    items = [
    (
        item.idx,
        item.attn_bias,
        item.attn_edge_type,
        item.spatial_pos,
        item.in_degree,
        item.out_degree,
        item.x,
        item.edge_input[:, :, :multi_hop_max_dist, :],
        # item.y,
    ) for item in items ]
    (
        idxs,
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_inputs,
        # ys,
    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    # if (ys[0].shape[0] > 1): # multitask
    #     ys = [y.reshape(1, -1) for y in ys]
    #     y = torch.cat(ys)
    # else:
    #     y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    )
    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
    )
    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])

    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        x=x,
        edge_input=edge_input,
        # y=y,
    )

class BatchedDataDataset_finetune(BaseWrapperDataset):
    def __init__(
        self, dataset, max_node=128, multi_hop_max_dist=5, spatial_pos_max=1024, cliff=True
    ):
        # super().__init__()
        self.dataset = dataset
        self.max_node = max_node
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max
        self.cliff=cliff

    def __getitem__(self, index):
        item = self.dataset[int(index)]
        return item

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        
        return collator_finetune(
            samples,
            max_node=self.max_node,
            multi_hop_max_dist=self.multi_hop_max_dist,
            spatial_pos_max=self.spatial_pos_max,
            cliff=self.cliff
        )

class BatchedDataDataset_pretrain(BaseWrapperDataset):
    def __init__(
        self, dataset, max_node=128, multi_hop_max_dist=5, spatial_pos_max=1024
    ):
        # super().__init__()
        self.dataset = dataset
        self.max_node = max_node
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max

    def __getitem__(self, index):
        item = self.dataset[int(index)]
        return item

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        
        return collator_pretrain(
            samples,
            max_node=self.max_node,
            multi_hop_max_dist=self.multi_hop_max_dist,
            spatial_pos_max=self.spatial_pos_max,
        )