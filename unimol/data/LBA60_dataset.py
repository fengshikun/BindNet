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
# from .pair_lba import deserialize_array
import os
from torch.utils.data import Subset
import sys
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from torch_geometric.data import Batch
from torch_geometric.data import Data


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
                 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118}

atomic_number = {k.upper():v for k, v in atomic_number.items()}


class LBA60Dataset(InMemoryDataset):
    def __init__(self, data_path, split,  transform_noise=None, lp_sep=False):

        split_data_path = os.path.join(data_path, f"{split}")
        self._env = pk.load(open(os.path.join(split_data_path, 'data.pkl'), 'rb'))
        self.length = len(self._env.keys())

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        if not 0 <= idx < self.length:
            raise IndexError(idx)

        self.data_dict = self._env[idx]
        data = Data()
        pocket_atomsnum = len(self.data_dict['pocket_atoms'])
        ligand_atomsnum = len(self.data_dict['ligand_atoms'])
        num_atoms = pocket_atomsnum + ligand_atomsnum

        data.pocketAtom = np.array(self.data_dict['pocket_atoms'], dtype=str)
        data.pocketPos = np.array(self.data_dict['pocket_coordinates'], dtype=np.float32)
        data.ligandAtom = np.array(self.data_dict['ligand_atoms'], dtype=str)

        data.ligz = np.array(list(map(atomic_number.get, data.ligandAtom)), dtype=np.int64)

        data.ligandPos = np.array(self.data_dict['ligand_coordinates'], dtype=np.float32)
        data.finetune_target =  np.array(self.data_dict['affinity'], dtype=np.float32)

        data.pocket_atomsnum = pocket_atomsnum

        return data


def make_LBA60_dataset(data, idx_path, max_num, split, split_ratio=0):
    raw_dataset = LBA60Dataset(data_path=data, split=split)
    org_data_len = len(raw_dataset)
    org_idx = np.array([idx for idx in range(org_data_len)])
    filter_idx = org_idx

    data_idx = filter_idx

    return Subset(raw_dataset, data_idx)



class ExtractLBA60Dataset(BaseWrapperDataset):
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
        ligand_atoms_z = data.ligz
        ligand_atoms = data.ligandAtom
        ligand_atom_pos = data.ligandPos
        ligand_num = ligand_atoms.shape[0]

        all_atoms_pos = torch.tensor(np.concatenate([protein_atoms_pos, ligand_atom_pos], axis=0))

        # Get affinity
        finetune_target = torch.tensor(data.finetune_target)

        return {"smi": "", "atoms": protein_atoms, "protein_atoms_pos": protein_atoms_pos, 'atoms_lig': ligand_atoms, "ligand_atom_pos":ligand_atom_pos,
                "all_coordinate": all_atoms_pos, "prot_num": protein_num, "lig_num": ligand_num, 'affinity': finetune_target, "lig_atoms_z": ligand_atoms_z}


    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class FradDataset_LBA60(BaseWrapperDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        raw_data = self.dataset[idx]
        data = Data()
        data.z = torch.tensor(raw_data['lig_atoms_z'], dtype=torch.long)
        data.pos = torch.tensor(raw_data['ligand_atom_pos'], dtype=torch.float)
        return data

    def collater(self, samples):
        return Batch.from_data_list(samples)