# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    EpochShuffleDataset,
    TokenizeDataset,
    RightPadDataset2D,
    FromNumpyDataset,
    RawArrayDataset,
)
from unimol.data import (
    KeyDataset,
    ConformerSampleDataset,
    DistanceDataset,
    ProtLigDistanceDataset,
    EdgeTypeDataset,
    MaskPointsDataset,
    RemoveHydrogenDataset,
    AtomTypeDataset,
    NormalizeDataset,
    CroppingDataset,
    RightPadDatasetCoord,
    Add2DConformerDataset,
    ExtractCPConformerDataset,
    ExtractCPConformerDataset2,
    RightPadLigDataset,
    LMDBDataset,
    Pocket,
    make_pocket_dataset,
    make_BioLip_dataset,
    FradDataset,
    CrossDockData
)
from unicore.tasks import UnicoreTask, register_task


from unimol.data.graphormer_data import (
    BatchedDataDataset_pretrain,
    Convert2DDataset
)

logger = logging.getLogger(__name__)


@register_task("unimol")
class UniMolTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )
        parser.add_argument(
            "--mask-prob",
            default=0.15,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0.05,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0.05,
            type=float,
            help="probability of replacing a token with a random token",
        )
        parser.add_argument(
            "--noise-type",
            default="uniform",
            choices=["trunc_normal", "uniform", "normal", "none"],
            help="noise type in coordinate noise",
        )
        parser.add_argument(
            "--noise",
            default=1.0,
            type=float,
            help="coordinate noise for masked atoms",
        )
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--remove-polar-hydrogen",
            action="store_true",
            help="remove polar hydrogen atoms",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--dict-name",
            default="dict_protein.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--ligdict-name",
            default="dict_ligand.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--only-polar",
            default=1,
            type=int,
            help="1: only polar hydrogen ; -1: all hydrogen ; 0: remove all hydrogen ",
        )
        parser.add_argument(
            "--run-name",
            default='cdmol',
            type=str,
            help="wandb run name",
        )


        parser.add_argument(
            "--remove-lba-casf",
            default=0,
            type=int,
            help="remove the overlap with lba and casf",
        )
        # parser.add_argument(
        #     "--ctl-2d",
        #     default=1,
        #     type=int,
        #     help="1: 2d contrastive ; 0: remove all hydrogen ",
        # )

    def __init__(self, args, dictionary, lig_dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.lig_dictionary = lig_dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        lig_dictionary.add_symbol("[MASK]", is_special=True)

        self.ctl_2d = self.args.ctl_2d

        self.complex_pretrain = self.args.complex_pretrain
        self.pocket_data = self.args.pocket_data
        self.idx_path = self.args.idx_path
        self.max_comnum = self.args.max_comnum
        self.online_ligfeat = self.args.online_ligfeat
        self.rdkit_random = self.args.rdkit_random
        self.mask_feature = self.args.mask_feature


        if self.args.only_polar > 0:
            self.args.remove_polar_hydrogen = True
        elif args.only_polar < 0:
            self.args.remove_polar_hydrogen = False
        else:
            self.args.remove_hydrogen = True

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        lig_dictionary = Dictionary.load(os.path.join(args.data, args.ligdict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary, lig_dictionary)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if self.complex_pretrain:
            raw_dataset = CrossDockData(f'{self.pocket_data}/{split}_new.lmdb', max_num=self.max_comnum)
            # raw_dataset = make_pocket_dataset(self.pocket_data, self.idx_path, self.max_comnum, split)
            # raw_dataset = make_BioLip_dataset(self.pocket_data, self.idx_path, self.max_comnum, split, remove_lba_casf=self.args.remove_lba_casf)
            # raw_dataset = Pocket(self.pocket_data, self.idx_path)
        else:
            split_path = os.path.join(self.args.data, split + ".lmdb")
            raw_dataset = LMDBDataset(split_path)

        def one_dataset(raw_dataset, coord_seed, mask_seed, load2d=False, complex_pretrain=False, online_ligfeat=False, rdkit_random=False, mask_feature=False):
            if self.args.mode =='train':
                if self.complex_pretrain:
                    # NOTE: rdkit_random is the setting for the ligand conformer generation
                    dataset = ExtractCPConformerDataset2(raw_dataset, "smi", "atoms", "coordinates", rdkit_random=rdkit_random, mask_feat=mask_feature)
                    smi_dataset = KeyDataset(dataset, "smi")
                    # lig_feat_dataset = KeyDataset(dataset, "lig_feat")
                    # frad_dataset = FradDataset(dataset)
                else:
                    raw_dataset = Add2DConformerDataset(
                        raw_dataset, "smi", "atoms", "coordinates"
                    )
                    smi_dataset = KeyDataset(raw_dataset, "smi")


            if not self.complex_pretrain: # for normal pretraining
                dataset = ConformerSampleDataset(
                    raw_dataset, coord_seed, "atoms", "coordinates"
                )
                dataset = AtomTypeDataset(raw_dataset, dataset)
                dataset = RemoveHydrogenDataset(
                    dataset,
                    "atoms",
                    "coordinates",
                    self.args.remove_hydrogen,
                    self.args.remove_polar_hydrogen,
                )

            # crop mask dataset: cut max_atoms, if exceeds, random select
            # first, we don't allow the max_atoms work
            if self.complex_pretrain:
                dataset = CroppingDataset(
                    dataset, self.seed, "atoms", "coordinates", self.max_comnum
                )
            else:
                dataset = CroppingDataset(
                    dataset, self.seed, "atoms", "coordinates", self.args.max_atoms
                )
            dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
            dataset = NormalizeDataset(dataset, "atoms_pos_lig", normalize_coord=True)
            if mask_feature:
                dataset = NormalizeDataset(dataset, "atoms_pos_lig_org", normalize_coord=True)

            token_dataset = KeyDataset(dataset, "atoms")
            token_dataset = TokenizeDataset(
                token_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
            )
            coord_dataset = KeyDataset(dataset, "coordinates")
            if not self.complex_pretrain:
                expand_dataset = MaskPointsDataset(
                    token_dataset,
                    coord_dataset,
                    self.dictionary,
                    pad_idx=self.dictionary.pad(),
                    mask_idx=self.mask_idx,
                    noise_type=self.args.noise_type,
                    noise=self.args.noise,
                    seed=mask_seed,
                    mask_prob=self.args.mask_prob,
                    leave_unmasked_prob=self.args.leave_unmasked_prob,
                    random_token_prob=self.args.random_token_prob,
                )
            else:
                expand_dataset = MaskPointsDataset(
                    token_dataset,
                    coord_dataset,
                    self.dictionary,
                    pad_idx=self.dictionary.pad(),
                    mask_idx=self.mask_idx,
                    noise_type=self.args.noise_type,
                    noise=self.args.noise,
                    seed=mask_seed,
                    mask_prob=self.args.mask_prob,
                    leave_unmasked_prob=self.args.leave_unmasked_prob,
                    random_token_prob=self.args.random_token_prob,
                    mask_forbidden=True
                )

            def PrependAndAppend(dataset, pre_token, app_token):
                dataset = PrependTokenDataset(dataset, pre_token)
                return AppendTokenDataset(dataset, app_token)

            encoder_token_dataset = KeyDataset(expand_dataset, "atoms")
            encoder_coord_dataset = KeyDataset(expand_dataset, "coordinates")

            if not self.complex_pretrain:
                encoder_target_dataset = KeyDataset(expand_dataset, "targets")
                tgt_dataset = PrependAndAppend(
                encoder_target_dataset, self.dictionary.pad(), self.dictionary.pad()
                )
            else:
                # prepare label for the prot and ligand
                prot_pos_dataset = KeyDataset(dataset, "all_coordinate")
                prot_num_dataset = KeyDataset(dataset, "prot_num")
                lig_num_dataset = KeyDataset(dataset, "lig_num")
                # prot_lig_dataset = ProtLigDistanceDataset(prot_pos_dataset, type_mask_dataset)

            src_dataset = PrependAndAppend(
                encoder_token_dataset, self.dictionary.bos(), self.dictionary.eos()
            )

            encoder_coord_dataset = PrependAndAppend(encoder_coord_dataset, 0.0, 0.0)
            encoder_distance_dataset = DistanceDataset(encoder_coord_dataset)

            edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
            coord_dataset = FromNumpyDataset(coord_dataset)
            coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
            distance_dataset = DistanceDataset(coord_dataset)

            net_input = {
                "src_tokens": RightPadDataset(
                    src_dataset,
                    pad_idx=self.dictionary.pad(),
                ),
                "src_coord": RightPadDatasetCoord(
                    encoder_coord_dataset,
                    pad_idx=0,
                ),
                "src_distance": RightPadDataset2D(
                    encoder_distance_dataset,
                    pad_idx=0,
                ),
                "src_edge_type": RightPadDataset2D(
                    edge_type,
                    pad_idx=0,
                ),
                # "frad_dataset": frad_dataset,
            }
            if online_ligfeat:
                lig_encoder_token_dataset = KeyDataset(dataset, "atoms_lig")
                lig_encoder_token_dataset = TokenizeDataset(
                    lig_encoder_token_dataset, self.lig_dictionary, max_seq_len=self.args.max_seq_len
                )
                lig_src_dataset = PrependAndAppend(
                    lig_encoder_token_dataset, self.lig_dictionary.bos(), self.lig_dictionary.eos()
                )
                net_input['lig_tokens'] = RightPadDataset(
                    lig_src_dataset,
                    pad_idx=self.lig_dictionary.pad(),
                )

                lig_encoder_coord_dataset = KeyDataset(dataset, "atoms_pos_lig")
                lig_encoder_coord_dataset = PrependAndAppend(lig_encoder_coord_dataset, 0.0, 0.0)
                net_input['lig_coord'] = RightPadDatasetCoord(
                    lig_encoder_coord_dataset,
                    pad_idx=0,
                )

                lig_encoder_distance_dataset = DistanceDataset(lig_encoder_coord_dataset)
                lig_edge_type = EdgeTypeDataset(lig_src_dataset, len(self.lig_dictionary))

                net_input['lig_distance'] = RightPadDataset2D(
                    lig_encoder_distance_dataset,
                    pad_idx=0,
                )
                net_input['lig_edge_type'] = RightPadDataset2D(
                    lig_edge_type,
                    pad_idx=0,
                )
                if mask_feature:
                    lig_encoder_coord_dataset_org = KeyDataset(dataset, "atoms_pos_lig_org")
                    lig_encoder_coord_dataset_org = PrependAndAppend(lig_encoder_coord_dataset_org, 0.0, 0.0)
                    net_input['lig_org_coord_org'] = RightPadDatasetCoord(
                        lig_encoder_coord_dataset_org,
                        pad_idx=0,
                    )

                    lig_encoder_distance_dataset_org = DistanceDataset(lig_encoder_coord_dataset_org)

                    net_input['lig_org_distance'] = RightPadDataset2D(
                        lig_encoder_distance_dataset_org,
                        pad_idx=0,
                    )

                    feat_masking_idx = KeyDataset(dataset, "mask_array")
                    feat_masking_idx = PrependAndAppend(
                        feat_masking_idx, 0, 0
                    )
                    net_input['feat_masking_idx'] = RightPadDataset(
                        feat_masking_idx,
                        pad_idx=0,
                    )



            if complex_pretrain:
                # net_input["lig_feat_input"] = RightPadLigDataset(lig_feat_dataset)
                net_input['prot_num_lst'] = prot_num_dataset
                net_input['lig_num_lst'] = lig_num_dataset
                return net_input, {
                    "all_pos": RightPadDatasetCoord(prot_pos_dataset, pad_idx=0),
                    "prot_num": prot_num_dataset,
                    "lig_num": lig_num_dataset
                }

            if load2d:
                mol_dataset = KeyDataset(raw_dataset, "mol") # for 2D
                molg_dataset = Convert2DDataset(mol_dataset)
                net_input["mol_graph"] = BatchedDataDataset_pretrain(
                    molg_dataset
                )
            return net_input, {
                "tokens_target": RightPadDataset(
                    tgt_dataset, pad_idx=self.dictionary.pad()
                ),
                "distance_target": RightPadDataset2D(distance_dataset, pad_idx=0),
                "coord_target": RightPadDatasetCoord(coord_dataset, pad_idx=0),
                "smi_name": RawArrayDataset(smi_dataset),
            },

        net_input, target = one_dataset(raw_dataset, self.args.seed, self.args.seed, load2d=self.ctl_2d, complex_pretrain=self.complex_pretrain, online_ligfeat=self.online_ligfeat, rdkit_random=self.rdkit_random, mask_feature=self.mask_feature)
        dataset = {"net_input": net_input, "target": target}
        dataset = NestedDictionaryDataset(dataset)
        if split in ["train", "train.small"]:
            dataset = EpochShuffleDataset(dataset, len(dataset), self.args.seed)
        self.datasets[split] = dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        return model
