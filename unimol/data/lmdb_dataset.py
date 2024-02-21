# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import lmdb
import os
import pickle
from functools import lru_cache
import logging
from torch.utils.data import Subset

logger = logging.getLogger(__name__)


class LMDBDataset:
    def __init__(self, db_path, keep_data=1000000): # NOTE: keep only first 1 million molecules
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            if keep_data > 0:
                self._keys = list(txn.cursor().iternext(values=False))[:keep_data]
            else:
                self._keys = list(txn.cursor().iternext(values=False))

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, "env"):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(f"{idx}".encode("ascii"))
        data = pickle.loads(datapoint_pickled)
        return data


# complex data for pocket pre-training

from torch_geometric.data import Data, Dataset
import numpy as np
import torch
from torch_geometric.data import (InMemoryDataset, Data)
import mmap
import pickle as pk
from scipy.spatial import distance_matrix
import random
import lmdb

atomic_number = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
                 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
                 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
                 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
                 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
                 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
                 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
                 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
                 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
                 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
                 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
                 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118, 'D': 119}

atomic_number_reverse = {}
for k, v in atomic_number.items():
    atomic_number_reverse[v] = k


def deserialize_array(serialized):
    return np.frombuffer(serialized, dtype=np.float32)

class Pocket(InMemoryDataset):
    def __init__(self, pocket_data, idx_path, transform_noise=None, mask_ratio=0, mask_strategy='random', use_lig_feat=False, feat_path='feat_path', split_num=4):
        # mask_strategy = random or dist, mask the ligand only
        # random: random or dist
        # dist: pick the nearest pair to mask

        # get file handler
        pocket_r = open(pocket_data,'r+b')
        self.pocket_handler = mmap.mmap(pocket_r.fileno(), 0)
        # read index
        with open(idx_path, 'r') as ir:
            idx_info_lst = ir.readlines()

        self.idx_lst = []
        for ele in idx_info_lst:
            ele_array = [int(i) for i in ele.strip().split()]
            self.idx_lst.append(ele_array)
        self.length = len(self.idx_lst)
        self.transform_noise = transform_noise
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy

        self.use_lig_feat = use_lig_feat
        self.split_num = split_num
        if self.use_lig_feat:
            self.env_lst = []
            self._keys_lst = []
            for sidx in range(split_num):
                env = lmdb.open(f'{feat_path}_{sidx}', readonly=True, subdir=True, lock=False)
                self.env_lst.append(env)
                with env.begin() as txn:
                    _keys = list(txn.cursor().iternext(values=False))
                    self._keys_lst.append(_keys)
            self.sub_length = self.length // split_num

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx, drop_atom_lst=['H', 'D']):

        data = Data()
        # get z, pos, and y
        # read element
        idx = idx.item()
        idx_array = self.idx_lst[idx]
        org_data = pk.loads(self.pocket_handler[idx_array[0]: idx_array[0] + idx_array[1]])
        pocket_atoms = org_data['pocket_atoms']
        pocket_coordinates = org_data['pocket_coordinates']
        # pocket_z = [atomic_number.index(ele) for ele in pocket_atoms]
        pocket_atoms = np.array(pocket_atoms)
        pocket_z = [atomic_number[ele] for ele in pocket_atoms]
        pocket_z = np.array(pocket_z)


        if len(drop_atom_lst): # erase H
            mask_idx = np.in1d(pocket_atoms, drop_atom_lst)
            pocket_z = pocket_z[~mask_idx]
            pocket_coordinates = pocket_coordinates[~mask_idx]
            pocket_atoms = pocket_atoms[~mask_idx]
            # data.z = data.z[~mask_idx]
            # data.pos = data.pos[~mask_idx]
            # data.type_mask = data.type_mask[~mask_idx]



        lig_atoms_real = org_data['lig_atoms_real']
        lig_coord_real = org_data['lig_coord_real']
        lig_atoms_real = np.array(lig_atoms_real)
        # lig_z = [atomic_number.index(ele) for ele in lig_atoms_real]
        lig_z = [atomic_number[ele] for ele in lig_atoms_real]
        lig_z = np.array(lig_z)



        if len(drop_atom_lst): # erase H
            # pocket_atoms = np.array(pocket_atoms)
            mask_idx = np.in1d(lig_atoms_real, drop_atom_lst)
            lig_z = lig_z[~mask_idx]
            lig_coord_real = lig_coord_real[~mask_idx]
            lig_atoms_real = lig_atoms_real[~mask_idx]


        num_atoms = len(pocket_atoms) + len(lig_atoms_real)
        poc_lig_id = np.zeros(num_atoms)
        poc_lig_id[len(pocket_atoms): ] = 1 # lig 1



        if self.mask_ratio > 0:
            lig_atoms_num = len(lig_atoms_real)
            sample_size = int(lig_atoms_num * self.mask_ratio + 1)
            if self.mask_strategy == 'random':
                masked_atom_indices = random.sample(range(lig_atoms_num), sample_size)

            else:
                ligand_prot_matrix = distance_matrix(lig_coord_real, pocket_coordinates) # M X N
                ligand_prot_dis = ligand_prot_matrix.min(axis=1)
                ligand_prot_dis_idx = np.argsort(ligand_prot_dis) # from small to big
                masked_atom_indices = ligand_prot_dis_idx[:sample_size]

            masked_atom_indices.sort()

            # for ligand embedding masking
            ligand_embedding_mask = np.zeros(len(lig_atoms_real))
            ligand_embedding_mask[masked_atom_indices] = 1 # indicate where to replace the mask token feature
            data.ligand_embedding_mask = torch.tensor(ligand_embedding_mask)

            data.mask_node_label = lig_atoms_real[masked_atom_indices]
            # lig_z[masked_atom_indices] = self.num_atom_type
            data.masked_atom_indices = torch.tensor(masked_atom_indices) + len(pocket_atoms) # pocket first


        all_z = np.concatenate([pocket_z, lig_z])
        all_pos = np.concatenate([pocket_coordinates, lig_coord_real])



        data.z = torch.tensor(all_z, dtype=torch.long)
        data.pos = torch.tensor(all_pos, dtype=torch.float32)
        data.type_mask = torch.tensor(poc_lig_id, dtype=torch.long)


        if self.use_lig_feat:
            # decide which env
            if idx < self.sub_length:
                e_idx = 0
            elif idx < self.sub_length * 2:
                e_idx = 1
            elif idx < self.sub_length * 3:
                e_idx = 2
            else:
                e_idx = 3

            ky = str(idx).encode()
            # assert ky in self._keys_lst[e_idx]
            buffer = self.env_lst[e_idx].begin().get(ky)
            if buffer is None:
                print('debug')
            feat = deserialize_array(buffer).reshape(-1, 512)
            data.lig_feat = feat[1:-1]
            if self.mask_ratio > 0:
                regression_target = data.lig_feat[masked_atom_indices]
                data.regression_target = regression_target



        if self.transform_noise is not None:
            data = self.transform_noise(data) # noisy node

        return data


def make_pocket_dataset(pocket_data, idx_path, max_num, split, split_ratio=0.95):
    raw_dataset = Pocket(pocket_data, idx_path)
    atom_num_lst = np.load(os.path.join(os.path.dirname(pocket_data), 'pocket_atom_num_new.npy'))

    org_data_len = len(raw_dataset)
    idx_mask = (atom_num_lst < max_num)
    org_idx = np.array([idx for idx in range(org_data_len)])
    filter_idx = org_idx[idx_mask]

    data_len = len(filter_idx)
    train_len = int(data_len * split_ratio)
    if split == 'train':
        data_idx = filter_idx[:train_len]
    else:
        data_idx = filter_idx[train_len:]


    return Subset(raw_dataset, data_idx)




if __name__ == '__main__':
    db_path = '/feat_path'
    test_lmdb = LMDBDataset(db_path=db_path)