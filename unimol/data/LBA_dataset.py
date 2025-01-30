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
import pickle
from Bio.PDB import PDBParser
from torch_geometric.data import Batch
from torch_geometric.data import Data

import copy
from rdkit import Chem

class LBADataset(InMemoryDataset):
    def __init__(self, data_path, split,  transform_noise=None, lp_sep=False):

        self.transform_noise = transform_noise
        npyfile = data_path + f"_{split}.npy"
        self.data_dict = np.load(npyfile, allow_pickle=True).item() # dict
        self.length = len(self.data_dict['index'])
        self.lp_sep = lp_sep
        self.pocket_atom_offset = 120

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):

        data = Data()
        num_atoms = self.data_dict['num_atoms'][idx]
        pocket_atomsnum = self.data_dict['pocket_atoms'][idx]
        ligand_atomsnum = self.data_dict['ligand_atoms'][idx]
        assert (pocket_atomsnum + ligand_atomsnum) == num_atoms

        data.z = torch.tensor(self.data_dict['charges'][idx][:num_atoms], dtype=torch.long)
        data.pos = torch.tensor(self.data_dict['positions'][idx][:num_atoms], dtype=torch.float32)
        if self.transform_noise is not None:
            data = self.transform_noise(data) # noisy node

        data.affinity =  torch.tensor(self.data_dict['neglog_aff'][idx], dtype=torch.float32)

        data.org_pos = torch.tensor(self.data_dict['positions'][idx][:num_atoms], dtype=torch.float32)

        data.pocket_atomsnum = pocket_atomsnum

        # type mask
        poc_lig_id = np.zeros(num_atoms)
        poc_lig_id[pocket_atomsnum: ] = 1 # lig 1
        data.type_mask = torch.tensor(poc_lig_id, dtype=torch.long)

        return data


def make_LBA_dataset(data, idx_path, max_num, split, split_ratio=0.95):
    raw_dataset = LBADataset(data_path=data, split=split)
    org_data_len = len(raw_dataset)
    org_idx = np.array([idx for idx in range(org_data_len)])
    filter_idx = org_idx
    data_idx = filter_idx

    return Subset(raw_dataset, data_idx)  # 根据data_idx做slice



class ExtractLBADataset(BaseWrapperDataset):
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
        all_atoms_z = data.z
        all_atoms_pos = data.pos
        type_mask = data.type_mask.to(torch.bool)  # type_mask: ligand 1, protein 0

        # Protein Part
        protein_atoms_z = all_atoms_z[~type_mask].numpy()
        protein_atoms_pos = all_atoms_pos[~type_mask].numpy()
        protein_num = (~type_mask).sum().item()

        # Ligand Part
        ligand_atoms_z = all_atoms_z[type_mask].numpy()
        ligand_atom_pos = all_atoms_pos[type_mask].numpy()
        ligand_num = type_mask.sum().item()

        # Get Atom from Atom_z index
        protein_atoms_sym = np.array([atomic_number_reverse[ele] for ele in protein_atoms_z])
        ligand_atoms_sym = np.array([atomic_number_reverse[ele] for ele in ligand_atoms_z])

        # Get affinity
        affinity = data.affinity

        return {"smi": "", "atoms": protein_atoms_sym, "protein_atoms_pos": protein_atoms_pos, 'atoms_lig': ligand_atoms_sym, "ligand_atom_pos":ligand_atom_pos,
                "all_coordinate": all_atoms_pos, "prot_num": protein_num, "lig_num": ligand_num, 'affinity': affinity, "lig_atoms_z": ligand_atoms_z}


    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class FradDataset_LBA30(BaseWrapperDataset):
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

class ExtractDockDataset(BaseWrapperDataset):
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
        data_item = self.dataset[index]

        data = Data()
        data.atoms = data_item['atoms']
        data.coordinates = data_item['coordinates']

        if data.coordinates.shape[1] == 5:
            print('debug')

        data.pocket_atoms = data_item['pocket_atoms']
        data.pocket_coordinates = data_item['pocket_coordinates']

        smi = data_item['smi']
        pocket_name = data_item['pocket_name']

        # Protein Part
        protein_atoms_sym = [ele[0] for ele in  data.pocket_atoms]
        if not isinstance(data.pocket_coordinates, np.ndarray):
            protein_atoms_pos = data.pocket_coordinates.numpy()
        else:
            protein_atoms_pos = data.pocket_coordinates
        protein_num = len(data.pocket_atoms)

        # Ligand Part
        ligand_atoms_sym = data.atoms
        ligand_atom_pos = data.coordinates
        ligand_num = len(data.atoms)

        if ligand_atom_pos.shape[1] == 5:
            print('debug')

        # Get Atom from Atom_z index
        # protein_atoms_sym = np.array([atomic_number_reverse[ele] for ele in protein_atoms_z])
        # ligand_atoms_sym = np.array([atomic_number_reverse[ele] for ele in ligand_atoms_z])

        # Get fake affinity
        affinity = 1.0

        return {"smi": smi, "atoms": protein_atoms_sym, "protein_atoms_pos": protein_atoms_pos, 'atoms_lig': ligand_atoms_sym, "ligand_atom_pos":ligand_atom_pos, "prot_num": protein_num, "lig_num": ligand_num, 'affinity': affinity, "pocket_name": pocket_name}


    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class CrossDockDataInfer(InMemoryDataset):
    def __init__(self, file_path='protein_ligand_df.pkl', max_num=512):

        with open(file_path,'rb') as f:
            self.protein_ligand_df  = pickle.load(f)
        self.max_num = max_num

    def __len__(self) -> int:        
        return len(self.protein_ligand_df)

    def __getitem__(self, idx):
        
        data = Data()


        # get z, pos, and y
        # read element
        data_item = self.protein_ligand_df.iloc[idx, :]
        complex_graph = data_item['complex_graph']
        print(complex_graph['name'])
        ligand_feats = complex_graph['ligand'].x
        atom_type_idx = ligand_feats[:,0] + 1
        # get atom type from atom type index
        ligand_atom_type = [Chem.GetPeriodicTable().GetElementSymbol(int(i)) for i in atom_type_idx]
        data.atoms = ligand_atom_type
        data.coordinates = complex_graph['ligand'].pos.numpy()

        pocket_atom_feats = complex_graph['atom'].x
        pocket_atom_feats = pocket_atom_feats[:,1] + 1
        pocket_atom_type = []
        for i in pocket_atom_feats:
            try:
                pocket_atom_type.append(Chem.GetPeriodicTable().GetElementSymbol(int(i)))
            except:
                pocket_atom_type.append("[UNK]")
        #pocket_atom_type = [Chem.GetPeriodicTable().GetElementSymbol(int(i)) for i in pocket_atom_feats]        
        data.pocket_atoms = pocket_atom_type
        data.pocket_coordinates = complex_graph['atom'].pos.numpy()
        data.smi = Chem.MolToSmiles(data_item['mol'])
        data.pocket_name = complex_graph['name']
        #breakpoint()
        if self.max_num is not None:
            if len(data.pocket_atoms) > self.max_num:
                org_len = len(data.pocket_atoms)
                random_idx = random.sample(range(org_len), self.max_num)
                data.pocket_atoms = np.array(pocket_atom_type)[random_idx]
                data.pocket_coordinates = complex_graph['atom'].pos.numpy()[random_idx]

        data.y = 1 # fake label
        return data
        # item = self.protein_ligand_df.iloc[idx, :]
        # complex_graph = item['complex_graph']
        # org_coordinates = copy.copy(complex_graph['ligand'].pos)
        # #noise_complex_graph = self.transform(item)
        
        # # extract the information for bindnet
        
        # bindnet_data = Data()
        
        
        # # extract the ligand information
        # ligand_coordinate = complex_graph['ligand'].pos.numpy() # already after applying noise, TODO for bindnet, change it to the rdkit coordinate
        # ligand_feats = complex_graph['ligand'].x
        # atom_type_idx = ligand_feats[:,0] + 1
        # # get atom type from atom type index
        # ligand_atom_type = [Chem.GetPeriodicTable().GetElementSymbol(int(i)) for i in atom_type_idx]
        # ligand_info = {'coordinates': ligand_coordinate, 'atom_type_symbol': ligand_atom_type, 'org_coordinates': org_coordinates}
        # bindnet_data.ligand = ligand_info
        # bindnet_data.smi = Chem.MolToSmiles(item['mol'])
        # # org_coordinates
        
        
        
        
        # # extract the pocket information
        # pocket_atom_feats = complex_graph['atom']['x']
        # pocket_atom_feats = pocket_atom_feats[:,1] + 1
        # pocket_atom_type = [Chem.GetPeriodicTable().GetElementSymbol(int(i)) for i in pocket_atom_feats]
        # pocket_info = {'pocket_coordinates': complex_graph['atom']['pos'].numpy(), 'pocket_atom_type': pocket_atom_type}
        # bindnet_data.pocket_name = complex_graph['name']
        # bindnet_data.pocket_info = pocket_info
        
        # bindnet_data.heterodata = complex_graph
        
        # # if bindnet_data.heterodata.name == '5c0m_protein_processed_fix.pdb___5c0m_ligand.sdf':
        # #     print('debug')
        
        # return bindnet_data


if __name__ == '__main__':
    raw_dataset = CrossDock()


    raw_data_len = len(raw_dataset)
    test_num = 10

    from tqdm import tqdm
    # for i in tqdm(range(raw_data_len)):
    for i in tqdm(range(test_num)):
        data_item = raw_dataset[torch.tensor(i)]
        print(data_item)
