# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset

from .LBA_dataset import LBADataset
import torch

class TTADataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, atoms, coordinates, conf_size=10):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.conf_size = conf_size
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset) * self.conf_size

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        smi_idx = index // self.conf_size
        coord_idx = index % self.conf_size
        atoms = np.array(self.dataset[smi_idx][self.atoms])
        coordinates = np.array(self.dataset[smi_idx][self.coordinates][coord_idx])
        smi = self.dataset[smi_idx]["smi"]
        target = self.dataset[smi_idx]["target"]
        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "smi": smi,
            "target": target,
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)

from .lmdb_dataset import atomic_number_reverse



class TTADockingPoseDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        atoms,
        coordinates,
        pocket_atoms,
        pocket_coordinates,
        holo_coordinates,
        holo_pocket_coordinates,
        is_train=True,
        conf_size=10,
    ):
        self.dataset = dataset
        self.atoms = atoms
        self.coordinates = coordinates
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.holo_coordinates = holo_coordinates
        self.holo_pocket_coordinates = holo_pocket_coordinates
        self.is_train = is_train
        if isinstance(self.dataset, LBADataset):
            conf_size = 1
        self.conf_size = conf_size
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset) * self.conf_size

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        if isinstance(self.dataset, LBADataset):
            data_item = self.dataset[index]
            type_mask = data_item.type_mask.to(torch.bool)
            
            atoms_z = data_item.z[type_mask].numpy()
            pocket_atoms_z = data_item.z[~type_mask].numpy()
            atoms = np.array([atomic_number_reverse[ele] for ele in atoms_z])
            pocket_atoms = np.array([atomic_number_reverse[ele] for ele in pocket_atoms_z])
            
            coordinates = data_item.pos[type_mask].numpy()
            pocket_coordinates = data_item.pos[~type_mask].numpy()
            holo_coordinates = coordinates
            holo_pocket_coordinates = pocket_coordinates
            smi = ""
            pocket = ""
        else:
        
            smi_idx = index // self.conf_size
            coord_idx = index % self.conf_size
            atoms = np.array(self.dataset[smi_idx][self.atoms])
            coordinates = np.array(self.dataset[smi_idx][self.coordinates][coord_idx])
            pocket_atoms = np.array(
                [item[0] for item in self.dataset[smi_idx][self.pocket_atoms]]
            )
            pocket_coordinates = np.array(self.dataset[smi_idx][self.pocket_coordinates][0])
            if self.is_train:
                holo_coordinates = np.array(self.dataset[smi_idx][self.holo_coordinates][0])
                holo_pocket_coordinates = np.array(
                    self.dataset[smi_idx][self.holo_pocket_coordinates][0]
                )
            else:
                holo_coordinates = coordinates
                holo_pocket_coordinates = pocket_coordinates

            smi = self.dataset[smi_idx]["smi"]
            pocket = self.dataset[smi_idx]["pocket"]

        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_coordinates": holo_coordinates.astype(np.float32),
            "holo_pocket_coordinates": holo_pocket_coordinates.astype(np.float32),
            "smi": smi,
            "pocket": pocket,
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
