# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from collections.abc import Iterable

import numpy as np
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    TokenizeDataset,
    RightPadDataset2D,
    RawArrayDataset,
    FromNumpyDataset,
    EpochShuffleDataset,
)
from unimol.data import (
    KeyDataset,
    ConformerSampleDockingPoseDataset,
    DistanceDataset,
    EdgeTypeDataset,
    NormalizeDataset,
    RightPadDatasetCoord,
    LMDBDataset,
    CrossDistanceDataset,
    NormalizeDockingPoseDataset,
    TTADockingPoseDataset,
    RightPadDatasetCross2D,
    CroppingPocketDockingPoseDataset,
    PrependAndAppend2DDataset,
    RemoveHydrogenPocketDataset,
    make_BioLip_dataset,
    ConformerSampleDockingPoseDataset_BioDebug,
    LBADataset
)
from unicore import checkpoint_utils
from unicore.tasks import UnicoreTask, register_task



logger = logging.getLogger(__name__)


@register_task("docking_pose")
class DockingPose(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="downstream data path",
        )
        parser.add_argument(
            "--finetune-mol-model",
            default=None,
            type=str,
            help="pretrained molecular model path",
        )
        parser.add_argument(
            "--finetune-pocket-model",
            default=None,
            type=str,
            help="pretrained pocket model path",
        )

        parser.add_argument(
            "--finetune-complex-model",
            default=None,
            type=str,
            help="pretrained complex model path",
        )

        parser.add_argument(
            "--conf-size",
            default=10,
            type=int,
            help="number of conformers generated with each molecule",
        )
        parser.add_argument(
            "--dist-threshold",
            type=float,
            default=8.0,
            help="threshold for the distance between the molecule and the pocket",
        )
        parser.add_argument(
            "--max-pocket-atoms",
            type=int,
            default=512,
            help="selected maximum number of atoms in a pocket",
        )

        parser.add_argument(
            "--freeze-encoder",
            type=int,
            default=0,
            help="freeze pocket and molecule",
        )

        parser.add_argument(
            "--lba_data",
            type=int,
            default=0,
            help="lba dataset for vis",
        )

    def __init__(self, args, dictionary, pocket_dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.pocket_dictionary = pocket_dictionary
        self.seed = args.seed

        self.freeze_encoder = args.freeze_encoder
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        self.pocket_mask_idx = pocket_dictionary.add_symbol("[MASK]", is_special=True)

    @classmethod
    def setup_task(cls, args, **kwargs):
        mol_dictionary = Dictionary.load(os.path.join(args.data, "dict_ligand.txt"))
        pocket_dictionary = Dictionary.load(os.path.join(args.data, "dict_protein.txt"))
        logger.info("ligand dictionary: {} types".format(len(mol_dictionary)))
        logger.info("pocket dictionary: {} types".format(len(pocket_dictionary)))
        return cls(args, mol_dictionary, pocket_dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        'smi','pocket','atoms','coordinates','pocket_atoms','pocket_coordinates','holo_coordinates','holo_pocket_coordinates','scaffold'
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """
        data_path = os.path.join(self.args.data, split + ".lmdb")

        if self.args.lba_data:
            dataset = LBADataset(self.args.data + '/lba', split='test')
        else:
            dataset = LMDBDataset(data_path)
        if split.startswith("train"):
            smi_dataset = KeyDataset(dataset, "smi")
            poc_dataset = KeyDataset(dataset, "pocket")
            dataset = ConformerSampleDockingPoseDataset(
                dataset,
                self.args.seed,
                "atoms",
                "coordinates",
                "pocket_atoms",
                "pocket_coordinates",
                "holo_coordinates",
                "holo_pocket_coordinates",
                True,
            )

        else:
            dataset = TTADockingPoseDataset(
                dataset,
                "atoms",
                "coordinates",
                "pocket_atoms",
                "pocket_coordinates",
                "holo_coordinates",
                "holo_pocket_coordinates",
                True,
                self.args.conf_size,
            )
            smi_dataset = KeyDataset(dataset, "smi")
            poc_dataset = KeyDataset(dataset, "pocket")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket_atoms",
            "pocket_coordinates",
            "holo_pocket_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDockingPoseDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            "holo_pocket_coordinates",
            self.args.max_pocket_atoms,
        )
        dataset = RemoveHydrogenPocketDataset(
            dataset, "atoms", "coordinates", "holo_coordinates", True, True
        )

        apo_dataset = NormalizeDataset(dataset, "coordinates")
        apo_dataset = NormalizeDataset(apo_dataset, "pocket_coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)

        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
        src_pocket_dataset = TokenizeDataset(
            src_pocket_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")
        src_pocket_dataset = PrependAndAppend(
            src_pocket_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket_edge_type = EdgeTypeDataset(
            src_pocket_dataset, len(self.pocket_dictionary)
        )
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(
            distance_pocket_dataset, 0.0
        )

        holo_dataset = NormalizeDockingPoseDataset(
            dataset,
            "holo_coordinates",
            "holo_pocket_coordinates",
            "holo_center_coordinates",
        )
        holo_coord_dataset = KeyDataset(holo_dataset, "holo_coordinates")
        holo_coord_dataset = FromNumpyDataset(holo_coord_dataset)
        holo_coord_pocket_dataset = KeyDataset(holo_dataset, "holo_pocket_coordinates")
        holo_coord_pocket_dataset = FromNumpyDataset(holo_coord_pocket_dataset)

        holo_cross_distance_dataset = CrossDistanceDataset(
            holo_coord_dataset, holo_coord_pocket_dataset
        )

        holo_distance_dataset = DistanceDataset(holo_coord_dataset)
        holo_coord_dataset = PrependAndAppend(holo_coord_dataset, 0.0, 0.0)
        holo_distance_dataset = PrependAndAppend2DDataset(holo_distance_dataset, 0.0)
        holo_coord_pocket_dataset = PrependAndAppend(
            holo_coord_pocket_dataset, 0.0, 0.0
        )
        holo_cross_distance_dataset = PrependAndAppend2DDataset(
            holo_cross_distance_dataset, 0.0
        )

        holo_center_coordinates = KeyDataset(holo_dataset, "holo_center_coordinates")
        holo_center_coordinates = FromNumpyDataset(holo_center_coordinates)

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_tokens": RightPadDataset(
                        src_pocket_dataset,
                        pad_idx=self.pocket_dictionary.pad(),
                    ),
                    "pocket_src_distance": RightPadDataset2D(
                        distance_pocket_dataset,
                        pad_idx=0,
                    ),
                    "pocket_src_edge_type": RightPadDataset2D(
                        pocket_edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_coord": RightPadDatasetCoord(
                        coord_pocket_dataset,
                        pad_idx=0,
                    ),
                },
                "target": {
                    "distance_target": RightPadDatasetCross2D(
                        holo_cross_distance_dataset, pad_idx=0
                    ),
                    "holo_coord": RightPadDatasetCoord(holo_coord_dataset, pad_idx=0),
                    "holo_distance_target": RightPadDataset2D(
                        holo_distance_dataset, pad_idx=0
                    ),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "pocket_name": RawArrayDataset(poc_dataset),
                "holo_center_coordinates": RightPadDataset(
                    holo_center_coordinates,
                    pad_idx=0,
                ),
            },
        )
        if split.startswith("train"):
            nest_dataset = EpochShuffleDataset(
                nest_dataset, len(nest_dataset), self.args.seed
            )
        self.datasets[split] = nest_dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)

        if args.finetune_complex_model is not None:
            print('load complex model weight from...', args.finetune_complex_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_complex_model,
            )
            # load complex layer
            missing_keys, not_matched_keys = model.load_state_dict(state["model"], strict=False)
            print(f'load complex model, missing keys: {missing_keys}')


            # pickout the mol and pocket part
            # load pocket
            missing_keys, not_matched_keys = model.pocket_model.load_state_dict(state["model"], strict=False)
            print(f'load complex model pocket part, missing keys: {missing_keys}')
            new_state = {}
            for k in state["model"]:
                if k.startswith('lig'):
                    new_k = k.replace('lig_', '')
                    new_state[new_k] = state['model'][k]
            # load ligand
            missing_keys, not_matched_keys = model.mol_model.load_state_dict(new_state, strict=False)
            print(f'load complex model mol part, missing keys: {missing_keys}')

            print(f'load complex complete')
        else:
            if args.finetune_mol_model is not None:
                print("load pretrain model weight from...", args.finetune_mol_model)
                state = checkpoint_utils.load_checkpoint_to_cpu(
                    args.finetune_mol_model,
                )
                model.mol_model.load_state_dict(state["model"], strict=False)
            if args.finetune_pocket_model is not None:
                print("load pretrain model weight from...", args.finetune_pocket_model)
                state = checkpoint_utils.load_checkpoint_to_cpu(
                    args.finetune_pocket_model,
                )
                model.pocket_model.load_state_dict(state["model"], strict=False)

        if self.freeze_encoder:
            model.mol_model.eval()
            model.pocket_model.eval()

        return model
