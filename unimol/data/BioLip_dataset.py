from torch_geometric.data import Data, Dataset
import numpy as np
import torch
from torch_geometric.data import (InMemoryDataset, Data)
import mmap
import pickle as pk
from scipy.spatial import distance_matrix
import random
import lmdb
import os
from torch.utils.data import Subset
from tqdm import tqdm
import pandas as pd
import random

# import seaborn as sns
import matplotlib.pyplot as plt

atomic_number = {'H': 1, 'HE': 2, 'LI': 3, 'BE': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'NE': 10, 'NA': 11, 'MG': 12, 'AL': 13, 'SI': 14, 'P': 15, 'S': 16,
                 'CL': 17, 'AR': 18, 'K': 19, 'CA': 20, 'SC': 21, 'TI': 22, 'V': 23, 'CR': 24, 'MN': 25, 'FE': 26, 'CO': 27, 'NI': 28, 'CU': 29, 'ZN': 30, 'GA': 31,
                 'GE': 32, 'AS': 33, 'SE': 34, 'BR': 35, 'KR': 36, 'RB': 37, 'SR': 38, 'Y': 39, 'ZR': 40, 'NB': 41, 'MO': 42, 'TC': 43, 'RU': 44, 'RH': 45, 'PD': 46,
                 'AG': 47, 'CD': 48, 'IN': 49, 'SN': 50, 'SB': 51, 'TE': 52, 'I': 53, 'XE': 54, 'CS': 55, 'BA': 56, 'LA': 57, 'CE': 58, 'PR': 59, 'ND': 60, 'PM': 61,
                 'SM': 62, 'EU': 63, 'GD': 64, 'TB': 65, 'DY': 66, 'HO': 67, 'ER': 68, 'TM': 69, 'YB': 70, 'LU': 71, 'HF': 72, 'TA': 73, 'W': 74, 'RE': 75, 'OS': 76,
                 'IR': 77, 'PT': 78, 'AU': 79, 'HG': 80, 'TL': 81, 'PB': 82, 'BI': 83, 'PO': 84, 'AT': 85, 'RN': 86, 'FR': 87, 'RA': 88, 'AC': 89, 'TH': 90, 'PA': 91,
                   'U': 92, 'NP': 93, 'PU': 94, 'AM': 95, 'CM': 96, 'BK': 97, 'CF': 98, 'ES': 99, 'FM': 100, 'MD': 101, 'NO': 102, 'LR': 103, 'RF': 104, 'DB': 105,
                   'SG': 106, 'BH': 107, 'HS': 108, 'MT': 109, 'DS': 110, 'RG': 111, 'CN': 112, 'NH': 113, 'FL': 114, 'MC': 115, 'LV': 116, 'TS': 117, 'OG': 118,
                   'D': 119, 'X': 120}

atomic_number_reverse = {}
for k, v in atomic_number.items():
    atomic_number_reverse[v] = k

def deserialize_array(serialized):
    return np.frombuffer(serialized, dtype=np.float32)

class BioLip(InMemoryDataset):
    def __init__(self, pocket_data, lig_name, transform_noise=None, mask_ratio=0, mask_strategy='random', use_lig_feat=False, feat_path='.', split_num=4, except_lig_list=None):

        self.transform_noise = transform_noise
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy
        self.use_lig_feat = use_lig_feat
        self.split_num = split_num
        self.lig_name = lig_name # lig_pdb_lst
        self.except_lig_list = except_lig_list

        self.env = lmdb.open(pocket_data, lock=False, readahead=False, meminit=False)
        with self.env.begin() as txn:
            self.length = len(list(txn.cursor().iternext(values=False)))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx, drop_atom_lst=['H', 'D', 'X']):   # X is unidentifier atom

        data = Data()
        encode_idx = str(idx.item()).encode()
        org_data = pk.loads(self.env.begin().get(encode_idx))

        pocket_atoms = org_data['pocket_atoms']
        pocket_coordinates = org_data['pocket_coordinates']



        pocket_atoms = np.array(pocket_atoms)
        pocket_z = [atomic_number[ele] for ele in pocket_atoms]    # Get atom index
        pocket_z = np.array(pocket_z)

        if len(drop_atom_lst): # erase H
            mask_idx = np.in1d(pocket_atoms, drop_atom_lst)
            pocket_z = pocket_z[~mask_idx]
            pocket_coordinates = pocket_coordinates[~mask_idx]
            pocket_atoms = pocket_atoms[~mask_idx]

        lig_atoms_real = org_data['lig_atoms_real']
        lig_coord_real = org_data['lig_coord_real']
        lig_atoms_real = np.array(lig_atoms_real)
        lig_z = [atomic_number[ele] for ele in lig_atoms_real]
        lig_z = np.array(lig_z)


        if len(drop_atom_lst): # erase H and X
            # pocket_atoms = np.array(pocket_atoms)
            mask_idx = np.in1d(lig_atoms_real, drop_atom_lst)
            lig_z = lig_z[~mask_idx]
            lig_coord_real = lig_coord_real[~mask_idx]
            lig_atoms_real = lig_atoms_real[~mask_idx]


        num_atoms = len(pocket_atoms) + len(lig_atoms_real)
        poc_lig_id = np.zeros(num_atoms)
        poc_lig_id[len(pocket_atoms): ] = 1 # lig 1


        # concat z and pos
        all_z = np.concatenate([pocket_z, lig_z])
        all_pos = np.concatenate([pocket_coordinates, lig_coord_real])

        data.z = torch.tensor(all_z, dtype=torch.long)
        data.pos = torch.tensor(all_pos, dtype=torch.float32)
        data.type_mask = torch.tensor(poc_lig_id, dtype=torch.long)

        data.idx = idx.item()

        data.smi = 'S'
        data.pocket = 'P' # NOTE:  for debug


        if 'pocket_resnum' in org_data:
            residual = org_data['pocket_resnum']
            data.residual = residual

        # for rdkit conformer generation
        lig_pdb = self.lig_name[idx.item()]
        if lig_pdb in self.except_lig_list:
            data.rdkit = 0
        else:
            data.rdkit = 1
        data.lig_pdb = lig_pdb


        if self.transform_noise is not None:
            data = self.transform_noise(data) # noisy node

        return data

class CrossDockData(InMemoryDataset):
    def __init__(self, file_path='/data/protein/bowen19/train_retrieval.lmdb', max_num=512):
        # get file handler
        self.env = lmdb.open(
            file_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        self.txn = self.env.begin()
        keys = list(self.txn.cursor().iternext(values=False))
        self.length = len(keys)
        self.max_num = max_num
    
    def __len__(self) -> int:        
        return self.length
    
    def __getitem__(self, idx):

        data = Data()
        
        
        # get z, pos, and y
        # read element
        data.idx = idx
        ky = f'{idx}'.encode()
        datapoint_pickled = self.txn.get(ky)
        data_item = pk.loads(datapoint_pickled)
        # print(data_item["smi"])
        
        data.atoms = data_item['atoms']
        data.coordinates = data_item['lig_coord_real'] # rdkit pose
        data.pocket_atoms = data_item['pocket_atoms']
        data.pocket_coordinates = data_item['pocket_coordinates']
        data.smi = data_item['smi']
        data.pocket_name = data_item['protein_name']
        
        lig_coord_real = data_item['coordinates'] # real pose
        
        # random pick one 
        real_len = len(lig_coord_real)
        real_idx = random.randint(0, real_len - 1)
        data.lig_coord_real = lig_coord_real[real_idx]
        
        if len(data.pocket_atoms) > self.max_num:
            org_len = len(data.pocket_atoms)
            random_idx = random.sample(range(org_len), self.max_num)
            data.pocket_atoms = np.array(data_item['pocket_atoms'])[random_idx]
            data.pocket_coordinates = data_item['pocket_coordinates'][random_idx]
        
        # data.y = 1 # fake label
        return data




def make_BioLip_dataset(pocket_data, idx_path, max_num, split, split_ratio=0.95, remove_lba_casf=False):
    idx_path = './data/BioLip/remain_line.txt'

    head_columns = []
    with open('./data/BioLip/complex_info.txt', 'r') as cr:
        cr_lines = cr.readlines()
        for line in cr_lines:
            head_name = line.strip()
            head_columns.append(head_name)

    complex_info = pd.read_csv(idx_path, sep='\t', names=head_columns)


    ligand_name_compose = [head_columns[0], head_columns[4], head_columns[5], head_columns[6]]
    pdb_id = complex_info[ligand_name_compose[0]].values
    ligand_id = complex_info[ligand_name_compose[1]].values
    ligand_chain = complex_info[ligand_name_compose[2]].values
    ligand_s_num = complex_info[ligand_name_compose[3]].values

    lig_path_base = './data/BioLip/BioLiP_updated_set/ligand'
    lig_name = []
    for i, ele in enumerate(pdb_id):
        ele_name = f'{lig_path_base}/{ele}_{ligand_id[i]}_{ligand_chain[i]}_{ligand_s_num[i]}.pdb'
        lig_name.append(ele_name)


    # concant convert to rdkit mol lst
    except_lig_list = []
    fail_lig_file = ['./data/BioLip/cannot_converge_mol_lst.npy', './data/BioLip/failed_align_lst.npy', './data/BioLip/not_exits_lst.npy']

    for fail_file in fail_lig_file:
        except_lig_list.append(np.load(fail_file))

    except_lig_list = np.concatenate(except_lig_list)


    raw_dataset = BioLip(pocket_data, lig_name, except_lig_list=except_lig_list)
    atom_num_lst = np.load('./data/BioLip/BioLip_atom_num.npy')
    org_data_len = len(raw_dataset)
    idx_mask = (atom_num_lst < max_num)


    org_idx = np.array([idx for idx in range(org_data_len)])
    no_onlyH_ligand_idx_mask = np.load('./data/BioLip/not_only_H_ligand_array.npy')
    filter_idx = org_idx[idx_mask & no_onlyH_ligand_idx_mask]
    data_len = len(filter_idx)
    train_len = int(data_len * split_ratio)
    if split == 'train':
        data_idx = filter_idx[:train_len]
    else:
        data_idx = filter_idx[train_len:]

    return Subset(raw_dataset, data_idx)


if __name__ == '__main__':
    # pocket_data = '/data/protein/MHData/BioLiP/lmdb_atom3dPocket'
    # raw_dataset = BioLip(pocket_data, idx_path='')
    data_dir = '/drug/retrieval_gen/crossdock_train/valid_new.lmdb'
    test_dataset = CrossDockData(file_path=data_dir)
    for data_item in test_dataset:
        print(data_item)

