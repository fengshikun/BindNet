from torch_geometric.data import Data, Dataset
import numpy as np
import torch
from torch_geometric.data import (InMemoryDataset, Data)
from .lmdb_dataset import atomic_number_reverse
import pickle as pk
import json, io
import msgpack
import gzip, logging
import pandas as pd
from pathlib import Path
import lmdb
import random
import dgl
import os
from torch.utils.data import Subset
import sys
from functools import lru_cache
from unicore.data import BaseWrapperDataset
import random


class DUDEDataset(InMemoryDataset):
    def __init__(self, data_path, split,  transform_noise=None, lp_sep=False):
        self.split = split
        if self.split == 'train':
            self.lmdb_data = pk.load(open(os.path.join(data_path, 'new_data.pkl'), 'rb'))
            self.length = max(self.lmdb_data.keys()) + 1
        else:
            self.lmdb_data = pk.load(open(os.path.join(data_path, 'data.pkl'), 'rb'))
            self.length = max(self.lmdb_data.keys()) + 1

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        if not 0 <= idx < self.length:
            raise IndexError(idx)

        if self.split == 'train':
            self.data_dict_all = self.lmdb_data[idx]
            self.data_dict = random.choice(self.data_dict_all)
            data = Data()
            pocket_atomsnum = len(self.data_dict['pocket_atoms'])
            ligand_atomsnum = len(self.data_dict['lig_atoms_real'])
            num_atoms = pocket_atomsnum + ligand_atomsnum
            data.pocketAtom = np.array(self.data_dict['pocket_atoms'], dtype=str)
            data.pocketPos = np.array(self.data_dict['pocket_coordinates'], dtype=np.float32)
            data.ligandAtom = np.array(self.data_dict['lig_atoms_real'], dtype=str)
            data.ligandPos = np.array(self.data_dict['lig_coord_real'], dtype=np.float32)
            data.affinity =  np.array(self.data_dict['class'], dtype=int)
            data.idx = idx
            data.pocket_atomsnum = pocket_atomsnum
            return data
        else:
            # valid, test
            self.data_dict = self.lmdb_data[idx]
            data = Data()
            pocket_atomsnum = len(self.data_dict['pocket_atoms'])
            ligand_atomsnum = len(self.data_dict['lig_atoms_real'])
            num_atoms = pocket_atomsnum + ligand_atomsnum
            data.pocketAtom = np.array(self.data_dict['pocket_atoms'], dtype=str)
            data.pocketPos = np.array(self.data_dict['pocket_coordinates'], dtype=np.float32)
            data.ligandAtom = np.array(self.data_dict['lig_atoms_real'], dtype=str)
            data.ligandPos = np.array(self.data_dict['lig_coord_real'], dtype=np.float32)
            data.affinity =  np.array(self.data_dict['class'], dtype=int)
            data.idx = idx
            data.pocket_atomsnum = pocket_atomsnum
            return data


def make_DUDE_dataset(data, idx_path, max_num, split, split_ratio=0, fold=1, task_name=None, all_test=False):
    raw_dataset = DUDEDataset(data_path=data, split=split)

    train_idx = np.load(os.path.join(data, f'id_list_train{fold}.npy'))
    train_idx = np.array([int(x) for x in train_idx])
    np.random.shuffle(train_idx)
    valid_idx = np.load(os.path.join(data, f'id_list_valid{fold}.npy'))
    valid_idx = np.array([int(x) for x in valid_idx])
    if all_test > 0:
        test_idx = np.load(os.path.join(data, f'id_list_test{fold}_all.npy'))
    else:
        test_idx = np.load(os.path.join(data, f'id_list_test{fold}.npy'))
    test_idx = np.array([int(x) for x in test_idx])

    if split == 'train':
        data_idx = train_idx
    elif split == 'valid':
        data_idx = valid_idx
    elif split == 'test':
        data_idx = test_idx

    return Subset(raw_dataset, data_idx)



class ExtractDUDEDataset(BaseWrapperDataset):
    '''Return dataset to input in the model and target for training'''
    def __init__(self, dataset, smi, atoms, coordinates, task='complex_pretrain'):
        self.dataset = dataset
        self.smi = smi
        self.atoms = atoms
        self.coordinates = coordinates
        self.task = task
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=128)
    def __cached_item__(self, index: int, epoch: int):
        data = self.dataset[index]

        # Protein Part
        protein_atoms = data.pocketAtom
        protein_atoms_pos = data.pocketPos
        protein_num = protein_atoms.shape[0]

        # Ligand Part
        ligand_atoms = data.ligandAtom
        ligand_atom_pos = data.ligandPos
        ligand_num = ligand_atoms.shape[0]

        all_atoms_pos = torch.tensor(np.concatenate([protein_atoms_pos, ligand_atom_pos], axis=0))

        # Get affinity
        affinity = torch.tensor(data.affinity)

        # Get id
        idx = data.idx

        return {"smi": "", "atoms": protein_atoms, "protein_atoms_pos": protein_atoms_pos, 'atoms_lig': ligand_atoms, "ligand_atom_pos":ligand_atom_pos,
                "all_coordinate": all_atoms_pos, "prot_num": protein_num, "lig_num": ligand_num, 'affinity': affinity, 'idx': idx}


    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
