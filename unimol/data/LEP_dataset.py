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



class LEPDataset(InMemoryDataset):
    def __init__(self, data_path, split,  transform_noise=None, lp_sep=False):

        split_data_path = os.path.join(data_path, f"{split}")

        self._env = lmdb.open(split_data_path, max_readers=100, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with self._env.begin(write=False) as txn:
            self.length = len(list(txn.cursor().iternext(values=False)))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        if not 0 <= idx < self.length:
            raise IndexError(idx)

        with self._env.begin(write=False) as txn:
            self.data_dict = pk.loads(txn.get(str(idx).encode()))
            data = Data()
            # get z, pos, and y
            pocket_atomsnum = len(self.data_dict['pocket_atoms'])
            ligand_atomsnum = len(self.data_dict['lig_atoms_real'])
            num_atoms = pocket_atomsnum + ligand_atomsnum


            data.pocketAtom = np.array(self.data_dict['pocket_atoms'], dtype=str)
            data.pocketPos = np.array(self.data_dict['pocket_coordinates'], dtype=np.float32)
            data.ligandAtom = np.array(self.data_dict['lig_atoms_real'], dtype=str)
            data.ligandPos = np.array(self.data_dict['lig_coord_real'], dtype=np.float32)

            data.inactive_pocketAtom = np.array(self.data_dict['inactive_pocket_atoms'], dtype=str)
            data.inactive_pocketPos = np.array(self.data_dict['inactive_pocket_coordinates'], dtype=np.float32)
            data.inactive_ligandAtom = np.array(self.data_dict['inactive_lig_atoms_real'], dtype=str)
            data.inactive_ligandPos = np.array(self.data_dict['inactive_lig_coord_real'], dtype=np.float32)

            if self.data_dict['label'] == 'A':
                data.finetune_target =  np.array(1, dtype=np.int64)
            elif self.data_dict['label'] == 'I':
                data.finetune_target =  np.array(0, dtype=np.int64)

            data.pocket_atomsnum = pocket_atomsnum

        return data


def make_LEP_dataset(data, idx_path, max_num, split, split_ratio=0):
    raw_dataset = LEPDataset(data_path=data, split=split)
    org_data_len = len(raw_dataset)
    org_idx = np.array([idx for idx in range(org_data_len)])
    filter_idx = org_idx

    data_idx = filter_idx

    return Subset(raw_dataset, data_idx)



class ExtractLEPDataset(BaseWrapperDataset):
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

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        data = self.dataset[index]

        # Protein Part
        protein_atoms = data.pocketAtom
        inactive_protein_atoms = data.inactive_pocketAtom
        protein_atoms_pos = data.pocketPos
        inactive_protein_atoms_pos = data.inactive_pocketPos
        protein_num = protein_atoms.shape[0]

        # Ligand Part
        ligand_atoms = data.ligandAtom
        inactive_ligand_atoms = data.inactive_ligandAtom
        ligand_atom_pos = data.ligandPos
        inactive_ligand_atom_pos = data.inactive_ligandPos
        ligand_num = ligand_atoms.shape[0]

        all_atoms_pos = torch.tensor(np.concatenate([protein_atoms_pos, ligand_atom_pos], axis=0))

        # Get affinity
        finetune_target = torch.tensor(data.finetune_target)

        return {"smi": "", "atoms": protein_atoms, "protein_atoms_pos": protein_atoms_pos, 'atoms_lig': ligand_atoms, "ligand_atom_pos":ligand_atom_pos,
                "all_coordinate": all_atoms_pos, "prot_num": protein_num, "lig_num": ligand_num, 'finetune_target': finetune_target,
                'inactive_protein_atoms_pos': inactive_protein_atoms_pos, 'inactive_ligand_atom_pos': inactive_ligand_atom_pos,
                'inactive_atoms': inactive_protein_atoms, 'inactive_atoms_lig': inactive_ligand_atoms}


    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
