# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from .lmdb_dataset import atomic_number_reverse
from .data_utils import collate_lig_2d
import random
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import rdMolTransforms
import copy
from torch_geometric.data import Batch

from torch_geometric.data import Data
from torch_geometric.data import Batch

from torch_geometric.data import Data

def get_torsions(mol_list):
    atom_counter = 0
    torsionList = []
    dihedralList = []
    for m in mol_list:
        torsionSmarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
        torsionQuery = Chem.MolFromSmarts(torsionSmarts)
        matches = m.GetSubstructMatches(torsionQuery)
        conf = m.GetConformer()
        for match in matches:
            idx2 = match[0]
            idx3 = match[1]
            bond = m.GetBondBetweenAtoms(idx2, idx3)
            jAtom = m.GetAtomWithIdx(idx2)
            kAtom = m.GetAtomWithIdx(idx3)
            for b1 in jAtom.GetBonds():
                if (b1.GetIdx() == bond.GetIdx()):
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)
                for b2 in kAtom.GetBonds():
                    if ((b2.GetIdx() == bond.GetIdx())
                            or (b2.GetIdx() == b1.GetIdx())):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    # skip 3-membered rings
                    if (idx4 == idx1):
                        continue
                    # skip torsions that include hydrogens
                    #                     if ((m.GetAtomWithIdx(idx1).GetAtomicNum() == 1)
                    #                         or (m.GetAtomWithIdx(idx4).GetAtomicNum() == 1)):
                    #                         continue
                    if m.GetAtomWithIdx(idx4).IsInRing():
                        torsionList.append(
                            (idx4 + atom_counter, idx3 + atom_counter, idx2 + atom_counter, idx1 + atom_counter))
                        break
                    else:
                        torsionList.append(
                            (idx1 + atom_counter, idx2 + atom_counter, idx3 + atom_counter, idx4 + atom_counter))
                        break
                break

        atom_counter += m.GetNumAtoms()
    return torsionList

def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)


def GetDihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])

def apply_changes(mol, values, rotable_bonds):
    opt_mol = copy.deepcopy(mol)
    #     opt_mol = add_rdkit_conformer(opt_mol)

    # apply rotations
    [SetDihedral(opt_mol.GetConformer(), rotable_bonds[r], values[r]) for r in range(len(rotable_bonds))]

    #     # apply transformation matrix
    #     rdMolTransforms.TransformConformer(opt_mol.GetConformer(), GetTransformationMatrix(values[:6]))

    return opt_mol

class Add2DConformerDataset(BaseWrapperDataset):
    def __init__(self, dataset, smi, atoms, coordinates):
        self.dataset = dataset
        self.smi = smi
        self.atoms = atoms
        self.coordinates = coordinates
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array(self.dataset[index][self.atoms])
        assert len(atoms) > 0
        smi = self.dataset[index][self.smi]
        mol, coordinates_2d = smi2_2Dcoords(smi)
        coordinates = self.dataset[index][self.coordinates]
        coordinates.append(coordinates_2d)
        return {"smi": smi, "atoms": atoms, "coordinates": coordinates, "mol": mol}

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)



class ExtractCPConformerDataset(BaseWrapperDataset):
    def __init__(self, dataset, smi, atoms, coordinates, rdkit_random=False, rdkit_seed=100, rdkit_avar=10, rdkit_cvar=0.04, mask_feat=False, mask_ratio=0.8):
        self.dataset = dataset
        self.smi = smi
        self.atoms = atoms
        self.coordinates = coordinates
        self.set_epoch(None)


        self.rdkit_random = rdkit_random
        self.rdkit_seed = rdkit_seed # max seed: 10
        self.rdkit_avar = rdkit_avar
        self.rdkit_cvar = rdkit_cvar
        self.mask_feat = mask_feat
        self.mask_ratio = mask_ratio

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        data = self.dataset[index]
        all_atoms_z = data.z
        all_atoms_pos = data.pos
        type_mask = data.type_mask.to(torch.bool)
        idx = data.idx
        # lig_feat = torch.from_numpy(data.lig_feat)

        # only pick pocket part
        atoms_z = all_atoms_z[~type_mask].numpy()
        atoms_pos_prot = all_atoms_pos[~type_mask].numpy()

        # also contain the ligand part atom3d
        lig_atoms_z = all_atoms_z[type_mask].numpy()
        atoms_pos_lig = all_atoms_pos[type_mask].numpy()

        atoms_pos_lig_org = atoms_pos_lig # for org

        if self.rdkit_random:
            if data.rdkit:
                try:
                    mol = Chem.rdmolfiles.MolFromPDBFile(data.lig_pdb)
                    mol.RemoveAllConformers()
                    r_seed = random.randrange(self.rdkit_seed)
                    addh_mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(addh_mol, randomSeed=r_seed, maxAttempts=1000)
                    AllChem.MMFFOptimizeMolecule(addh_mol)
                    mol = Chem.RemoveHs(addh_mol)
                    rdkit_lig_atoms_pos = mol.GetConformer().GetPositions()
                    assert rdkit_lig_atoms_pos.shape[0] == atoms_pos_lig.shape[0]
                    atoms_pos_lig = rdkit_lig_atoms_pos
                except Exception as e:
                    print(f'exception {e}  at {data.lig_pdb}')
                    # add torsion + random noise
                    mol = Chem.rdmolfiles.MolFromPDBFile(data.lig_pdb)
                    rotable_bonds = get_torsions([mol])
                    # add random noise
                    if len(rotable_bonds):
                        org_angle = []
                        for rot_bond in rotable_bonds:
                            org_angle.append(GetDihedral(mol.GetConformer(), rot_bond))
                        org_angle = np.array(org_angle)
                        noise_angle = np.random.normal(loc=org_angle, scale=self.rdkit_avar)
                        new_mol = apply_changes(mol, noise_angle, rotable_bonds)
                        atoms_pos_lig = new_mol.GetConformer().GetPositions()
                    atoms_pos_lig = np.random.normal(loc=atoms_pos_lig, scale=self.rdkit_cvar)
            else:
                atoms_pos_lig = np.random.normal(loc=atoms_pos_lig, scale=self.rdkit_cvar)


        prot_num = (~type_mask).sum().item()
        lig_num = type_mask.sum().item()



        atoms_sym = np.array([atomic_number_reverse[ele] for ele in atoms_z])
        lig_atoms_sym = np.array([atomic_number_reverse[ele] for ele in lig_atoms_z])

        if len(atoms_sym) == 0 or len(lig_atoms_sym) == 0:
            print('empty')

        data_res = {"smi": "", "atoms": atoms_sym, "coordinates": atoms_pos_prot, 'atoms_lig': lig_atoms_sym, "atoms_pos_lig":atoms_pos_lig, "atoms_pos_lig_org": atoms_pos_lig_org, "all_coordinate": all_atoms_pos, "prot_num": prot_num, "lig_num": lig_num, "index": idx, "lig_atoms_z": lig_atoms_z}
        # residual for pocket data
        if 'residual' in data.keys:
            data_res['residue'] = np.array(data.residual)

        if self.mask_feat:
            sample_size = int(lig_num * self.mask_ratio + 1)
            masked_atom_indices = random.sample(range(lig_num), sample_size)
            mask_array = np.zeros(lig_num, dtype=np.float32)
            mask_array[masked_atom_indices] = 1
            data_res['mask_array'] = mask_array

        return data_res
        # "lig_feat": lig_feat,

        # atoms = np.array(self.dataset[index][self.atoms])
        # assert len(atoms) > 0
        # smi = self.dataset[index][self.smi]
        # mol, coordinates_2d = smi2_2Dcoords(smi)
        # coordinates = self.dataset[index][self.coordinates]
        # coordinates.append(coordinates_2d)
        # return {"smi": smi, "atoms": atoms, "coordinates": coordinates, "mol": mol}

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)



class ExtractCPConformerDataset2(BaseWrapperDataset):
    def __init__(self, dataset, smi, atoms, coordinates, rdkit_random=False, rdkit_seed=100, rdkit_avar=10, rdkit_cvar=0.04, mask_feat=False, mask_ratio=0.8):
        self.dataset = dataset
        self.smi = smi
        self.atoms = atoms
        self.coordinates = coordinates
        self.set_epoch(None)


        self.rdkit_random = rdkit_random
        self.rdkit_seed = rdkit_seed # max seed: 10
        self.rdkit_avar = rdkit_avar
        self.rdkit_cvar = rdkit_cvar
        self.mask_feat = mask_feat
        self.mask_ratio = mask_ratio

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        data = self.dataset[index]
            
            
        atoms_sym = data.pocket_atoms
        atoms_pos_prot = torch.tensor(data.pocket_coordinates)
        
        
        lig_atoms_sym = data.atoms
        atoms_pos_lig = data.coordinates
        atoms_pos_lig_org = torch.tensor(data.lig_coord_real)
        all_atoms_pos = torch.tensor(np.concatenate((atoms_pos_prot, atoms_pos_lig), axis=0))
        prot_num = len(atoms_sym)
        lig_num = len(lig_atoms_sym)
        idx = data.idx
        lig_atoms_z = None
        smi = data.smi

        data_res = {"smi": smi, "atoms": atoms_sym, "coordinates": atoms_pos_prot, 'atoms_lig': lig_atoms_sym, "atoms_pos_lig":atoms_pos_lig, "atoms_pos_lig_org": atoms_pos_lig_org, "all_coordinate": all_atoms_pos, "prot_num": prot_num, "lig_num": lig_num, "index": idx, "lig_atoms_z": lig_atoms_z}
        # residual for pocket data
        # if 'residual' in data.keys:
        #     data_res['residue'] = np.array(data.residual)

        if self.mask_feat:
            sample_size = int(lig_num * self.mask_ratio + 1)
            masked_atom_indices = random.sample(range(lig_num), sample_size)
            mask_array = np.zeros(lig_num, dtype=np.float32)
            mask_array[masked_atom_indices] = 1
            data_res['mask_array'] = torch.tensor(mask_array)

        return data_res
        # "lig_feat": lig_feat,

        # atoms = np.array(self.dataset[index][self.atoms])
        # assert len(atoms) > 0
        # smi = self.dataset[index][self.smi]
        # mol, coordinates_2d = smi2_2Dcoords(smi)
        # coordinates = self.dataset[index][self.coordinates]
        # coordinates.append(coordinates_2d)
        # return {"smi": smi, "atoms": atoms, "coordinates": coordinates, "mol": mol}

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)



class FradDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        raw_data = self.dataset[idx]
        data = Data()
        data.z = torch.tensor(raw_data['lig_atoms_z'], dtype=torch.long)
        data.pos = torch.tensor(raw_data['atoms_pos_lig'], dtype=torch.float)
        data.org_pos = torch.tensor(raw_data['atoms_pos_lig_org'], dtype=torch.float)
        return data

    def collater(self, samples):
        return Batch.from_data_list(samples)



class ChemBLConformerDataset(BaseWrapperDataset):
    def __init__(self, dataset, smi, atoms, coordinates):
        self.dataset = dataset
        self.smi = smi
        self.atoms = atoms
        self.coordinates = coordinates
        self.set_epoch(None)


    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch


    def _handle_data(self, data):
        all_atoms_z = data.z
        all_atoms_pos = data.pos
        type_mask = data.type_mask.to(torch.bool)
        # lig_feat = torch.from_numpy(data.lig_feat)

        # only pick pocket part
        atoms_z = all_atoms_z[~type_mask].numpy()
        atoms_pos_prot = all_atoms_pos[~type_mask].numpy()

        # also contain the ligand part atom3d
        lig_atoms_z = all_atoms_z[type_mask].numpy()
        atoms_pos_lig = all_atoms_pos[type_mask].numpy()

        atoms_pos_lig_org = atoms_pos_lig # for org

        prot_num = (~type_mask).sum().item()
        lig_num = type_mask.sum().item()



        atoms_sym = np.array([atomic_number_reverse[ele] for ele in atoms_z])
        lig_atoms_sym = np.array([atomic_number_reverse[ele] for ele in lig_atoms_z])

        if len(atoms_sym) == 0 or len(lig_atoms_sym) == 0:
            print('empty')

        data_res = {"smi": "", "atoms": atoms_sym, "coordinates": atoms_pos_prot, 'atoms_lig': lig_atoms_sym, "atoms_pos_lig":atoms_pos_lig, "atoms_pos_lig_org": atoms_pos_lig_org, "all_coordinate": all_atoms_pos, "prot_num": prot_num, "lig_num": lig_num}


        return data_res


    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        data = self.dataset[index]

        # NOTE pair A and pair B for running
        if isinstance(data, tuple): # train
            data_res = self._handle_data(data[0])
            data_res2 = self._handle_data(data[1])
            data_res['affinity_diff'] = data[0].diff
            for k in data_res2:
                data_res[f'{k}_2'] = data_res2[k]
        else:
            data_res = self._handle_data(data)
            data_res['affinity_value'] = data.y


        return data_res



        # all_atoms_z = data.z
        # all_atoms_pos = data.pos
        # type_mask = data.type_mask.to(torch.bool)
        # # lig_feat = torch.from_numpy(data.lig_feat)

        # # only pick pocket part
        # atoms_z = all_atoms_z[~type_mask].numpy()
        # atoms_pos_prot = all_atoms_pos[~type_mask].numpy()

        # # also contain the ligand part atom3d
        # lig_atoms_z = all_atoms_z[type_mask].numpy()
        # atoms_pos_lig = all_atoms_pos[type_mask].numpy()

        # atoms_pos_lig_org = atoms_pos_lig # for org
        # # if self.rdkit_random:
        #     if data.rdkit:
        #         try:
        #             mol = Chem.rdmolfiles.MolFromPDBFile(data.lig_pdb)
        #             mol.RemoveAllConformers()
        #             r_seed = random.randrange(self.rdkit_seed)
        #             addh_mol = Chem.AddHs(mol)
        #             AllChem.EmbedMolecule(addh_mol, randomSeed=r_seed, maxAttempts=1000)
        #             AllChem.MMFFOptimizeMolecule(addh_mol)
        #             mol = Chem.RemoveHs(addh_mol)
        #             rdkit_lig_atoms_pos = mol.GetConformer().GetPositions()
        #             assert rdkit_lig_atoms_pos.shape[0] == atoms_pos_lig.shape[0]
        #             atoms_pos_lig = rdkit_lig_atoms_pos
        #         except Exception as e:
        #             print(f'exception {e}  at {data.lig_pdb}')
        #             # add torsion + random noise
        #             mol = Chem.rdmolfiles.MolFromPDBFile(data.lig_pdb)
        #             rotable_bonds = get_torsions([mol])
        #             # add random noise
        #             if len(rotable_bonds):
        #                 org_angle = []
        #                 for rot_bond in rotable_bonds:
        #                     org_angle.append(GetDihedral(mol.GetConformer(), rot_bond))
        #                 org_angle = np.array(org_angle)
        #                 noise_angle = np.random.normal(loc=org_angle, scale=self.rdkit_avar)
        #                 new_mol = apply_changes(mol, noise_angle, rotable_bonds)
        #                 atoms_pos_lig = new_mol.GetConformer().GetPositions()
        #             atoms_pos_lig = np.random.normal(loc=atoms_pos_lig, scale=self.rdkit_cvar)
        #     else:
        #         atoms_pos_lig = np.random.normal(loc=atoms_pos_lig, scale=self.rdkit_cvar)


        # prot_num = (~type_mask).sum().item()
        # lig_num = type_mask.sum().item()



        # atoms_sym = np.array([atomic_number_reverse[ele] for ele in atoms_z])
        # lig_atoms_sym = np.array([atomic_number_reverse[ele] for ele in lig_atoms_z])

        # if len(atoms_sym) == 0 or len(lig_atoms_sym) == 0:
        #     print('empty')

        # data_res = {"smi": "", "atoms": atoms_sym, "coordinates": atoms_pos_prot, 'atoms_lig': lig_atoms_sym, "atoms_pos_lig":atoms_pos_lig, "atoms_pos_lig_org": atoms_pos_lig_org, "all_coordinate": all_atoms_pos, "prot_num": prot_num, "lig_num": lig_num}


        # return data_res
        # "lig_feat": lig_feat,

        # atoms = np.array(self.dataset[index][self.atoms])
        # assert len(atoms) > 0
        # smi = self.dataset[index][self.smi]
        # mol, coordinates_2d = smi2_2Dcoords(smi)
        # coordinates = self.dataset[index][self.coordinates]
        # coordinates.append(coordinates_2d)
        # return {"smi": smi, "atoms": atoms, "coordinates": coordinates, "mol": mol}

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)



def smi2_2Dcoords(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    len(mol.GetAtoms()) == len(
        coordinates
    ), "2D coordinates shape is not align with {}".format(smi)
    return mol, coordinates


class RightPadLigDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)
    def collater(self, samples):
        return collate_lig_2d(samples)
