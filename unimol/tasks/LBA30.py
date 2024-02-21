import os
import logging
import torch
from unicore.tasks import UnicoreTask, register_task
logger = logging.getLogger(__name__)
import unicore
import lmdb

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
    make_LBA_dataset,
    ExtractLBADataset,
    NormalizeDataset,
    MaskPointsDataset,
    EdgeTypeDataset,
    DistanceDataset,
    RightPadDatasetCoord,
    FradDataset_LBA30,
    # pre_transform,
    # make_LBA_aaSequence_dataset,
    # Extract_LBA_aaSequence_Dataset
    ExtractDockDataset,
    CrossDockDataInfer
)

task_metainfo = {
    'LBA_affinity' : {
        'mean': 6.52,
        'std': 2.00,
        'target_name': 'LBA_affinity'
    }
}



@register_task("affinity_regres")
class AffinityRegres(UnicoreTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("--task-name", type=str, help="downstream task name")
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
            default="dict_coarse.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--ligdict-name",
            default="dict.txt",
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
            "--test-model",
            default='test_model',
            type=str,
            help="wandb run name",
        )
        
        parser.add_argument(
            "--extract-feat",
            default=0,
            type=int,
            help="wandb run name",
        )
        
        
        parser.add_argument(
            "--save-path",
            default='/data/protein/bowenfeat',
            type=str,
            help="save path of extract",
        )

    def __init__(self, args, dictionary, lig_dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.lig_dictionary = lig_dictionary
        self.seed = args.seed

        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        lig_dictionary.add_symbol("[MASK]", is_special=True)

        self.LBA_data = self.args.LBA_data
        self.idx_path = self.args.idx_path
        self.max_comnum = self.args.max_comnum
        if self.args.task_name in task_metainfo:
            self.mean = task_metainfo[self.args.task_name]["mean"]
            self.std = task_metainfo[self.args.task_name]["std"]


    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        lig_dictionary = Dictionary.load(os.path.join(args.data, args.ligdict_name))
        logger.info("Protein Dictionary: {} types".format(len(dictionary)))
        logger.info("Ligand Dictionary: {} types".format(len(lig_dictionary)))
        return cls(args, dictionary, lig_dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if hasattr(self.args, 'use_esm') and self.args.use_esm:
            raw_dataset = make_LBA_aaSequence_dataset(data=self.LBA_data, idx_path=self.idx_path, max_num=self.max_comnum, split=split)
        elif self.args.extract_feat:
            raw_dataset = CrossDockDataInfer(file_path=self.LBA_data, max_num=self.max_comnum) # no difference on the split
        else:
            raw_dataset = make_LBA_dataset(data=self.LBA_data, idx_path=self.idx_path, max_num=self.max_comnum, split=split)

        def one_dataset(raw_dataset, coord_seed, mask_seed):
            '''Get the input and target of one raw_dataset'''
            if self.args.mode =='train':
                if hasattr(self.args, 'use_esm') and self.args.use_esm:
                    dataset = Extract_LBA_aaSequence_Dataset(raw_dataset, "smi", "atoms", "coordinates", task='aff_finetune')
                elif self.args.extract_feat:
                    dataset = ExtractDockDataset(raw_dataset, "smi", "atoms", "coordinates", task='aff_finetune')
                    smi_dataset = KeyDataset(dataset, "smi")
                    pocket_dataset = KeyDataset(dataset, "pocket_name")
                else:
                    dataset = ExtractLBADataset(raw_dataset, "smi", "atoms", "coordinates", task='aff_finetune')
                    smi_dataset = KeyDataset(dataset, "smi")
                if (hasattr(self.args, 'use_frad') and self.args.use_frad) or (hasattr(self.args, 'use_egnn') and self.args.use_egnn):
                    if hasattr(self.args, 'use_egnn') and self.args.use_egnn:
                        transformer = pre_transform
                    else:
                        transformer = None
                    frad_dataset = FradDataset_LBA30(dataset, transformer)

            # Normalize the coordinates information for protein and ligand
            if not (hasattr(self.args, 'use_esm') and self.args.use_esm):
                dataset = NormalizeDataset(dataset, "protein_atoms_pos", normalize_coord=True)
            dataset = NormalizeDataset(dataset, "ligand_atom_pos", normalize_coord=True)

            # Tokenize the protein and ligand atoms
            if hasattr(self.args, 'use_esm') and self.args.use_esm:
                protein_token_dataset = KeyDataset(dataset, "aa_sequence")
                protein_aa_num_eachLigand_dataset = KeyDataset(dataset, "protein_aa_num_eachLigand")
            else:
                protein_token_dataset = KeyDataset(dataset, "atoms")     # The element in protein_token_dataset is Atom name
                protein_token_dataset = TokenizeDataset(protein_token_dataset, self.dictionary, max_seq_len=self.args.max_seq_len)    # TokenizeDataset change the atom name to atom index from dictionary
            ligand_token_dataset = KeyDataset(dataset, "atoms_lig")
            ligand_token_dataset = TokenizeDataset(ligand_token_dataset, self.lig_dictionary, max_seq_len=self.args.max_seq_len)  # lig_dictionary has token in dictionary in order and other atoms

            # Get the coordinate information for protein and ligand
            if not (hasattr(self.args, 'use_esm') and self.args.use_esm):
                protein_coord_dataset = KeyDataset(dataset, "protein_atoms_pos")
            ligand_coord_dataset = KeyDataset(dataset, "ligand_atom_pos")

            # Maybe used. But not mask now. Protein
            if not (hasattr(self.args, 'use_esm') and self.args.use_esm):
                expand_dataset = MaskPointsDataset(
                    protein_token_dataset,
                    protein_coord_dataset,
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
                protein_token_dataset = KeyDataset(expand_dataset, "atoms")
                protein_coord_dataset = KeyDataset(expand_dataset, "coordinates")
            # Ligand
            ligand_expand_dataset = MaskPointsDataset(
                ligand_token_dataset,
                ligand_coord_dataset,
                self.lig_dictionary,
                pad_idx=self.lig_dictionary.pad(),
                mask_idx=self.mask_idx,
                noise_type=self.args.noise_type,
                noise=self.args.noise,
                seed=mask_seed,
                mask_prob=self.args.mask_prob,
                leave_unmasked_prob=self.args.leave_unmasked_prob,
                random_token_prob=self.args.random_token_prob,
                mask_forbidden=True
            )
            if not (hasattr(self.args, 'use_esm') and self.args.use_esm):
                ligand_token_dataset = KeyDataset(ligand_expand_dataset, "atoms")
            ligand_coord_dataset = KeyDataset(ligand_expand_dataset, "coordinates")

            # Add CLS and SEP token for protein&ligand token and coordinate embedding
            def PrependAndAppend(dataset, pre_token, app_token):
                dataset = PrependTokenDataset(dataset, pre_token)   # Add CLS Token
                return AppendTokenDataset(dataset, app_token)       # Add SEP Token
            if not (hasattr(self.args, 'use_esm') and self.args.use_esm):
                protein_token_dataset = PrependAndAppend(protein_token_dataset, self.dictionary.bos(), self.dictionary.eos())
                protein_coord_dataset = PrependAndAppend(protein_coord_dataset, 0.0, 0.0)
            ligand_token_dataset = PrependAndAppend(ligand_token_dataset, self.lig_dictionary.bos(), self.lig_dictionary.eos())
            ligand_coord_dataset = PrependAndAppend(ligand_coord_dataset, 0.0, 0.0)

            # Get the distance, edgeTypen number information for protein and ligand
            if not (hasattr(self.args, 'use_esm') and self.args.use_esm):
                protein_distance_dataset = DistanceDataset(protein_coord_dataset)
                protein_edge_type = EdgeTypeDataset(protein_token_dataset, len(self.dictionary))
                prot_num_dataset = KeyDataset(dataset, "prot_num")
            ligand_distance_dataset = DistanceDataset(ligand_coord_dataset)
            ligand_edge_type = EdgeTypeDataset(ligand_token_dataset, len(self.lig_dictionary))
            lig_num_dataset = KeyDataset(dataset, "lig_num")

            # Define the net_input
            # protein part input
            if hasattr(self.args, 'use_esm') and self.args.use_esm:
                net_input = {
                    "aa_sequence": protein_token_dataset,
                    "protein_aa_num_eachLigand": protein_aa_num_eachLigand_dataset,
                }
            else:
                net_input = {
                    "src_tokens": RightPadDataset(
                        protein_token_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "src_coord": RightPadDatasetCoord(
                        protein_coord_dataset,
                        pad_idx=0,
                    ),
                    "src_distance": RightPadDataset2D(
                        protein_distance_dataset,
                        pad_idx=0,
                    ),
                    "src_edge_type": RightPadDataset2D(
                        protein_edge_type,
                        pad_idx=0,
                    ),
                    "prot_num_lst" :  prot_num_dataset
                }
            # ligand part input
            net_input["lig_tokens"] =  RightPadDataset(
                ligand_token_dataset,
                pad_idx=self.lig_dictionary.pad(),
            )
            net_input["lig_coord"] =  RightPadDatasetCoord(
                ligand_coord_dataset,
                pad_idx=0,
            )
            net_input["lig_distance"] =  RightPadDataset2D(
                ligand_distance_dataset,
                pad_idx=0,
            )
            net_input["lig_edge_type"] =  RightPadDataset2D(
                ligand_edge_type,
                pad_idx=0,
            )
            net_input["lig_num_lst"] =  lig_num_dataset
            if (hasattr(self.args, 'use_frad') and self.args.use_frad) or (hasattr(self.args, 'use_egnn') and self.args.use_egnn):
                net_input["frad_dataset"] = frad_dataset

            
            if self.args.extract_feat:
                net_input['smi'] = smi_dataset
                net_input['pocket_name'] = pocket_dataset
            # Define the target
            # protein_ligand_pos_dataset = KeyDataset(dataset, "all_coordinate")
            complex_affinity_dataset = KeyDataset(dataset, "affinity")

            return net_input, {
                # "all_pos": RightPadDatasetCoord(protein_ligand_pos_dataset, pad_idx=0),
                # "prot_num": prot_num_dataset,
                # "lig_num": lig_num_dataset,
                "affinity": complex_affinity_dataset,
            }

        net_input, target = one_dataset(raw_dataset=raw_dataset, coord_seed=self.args.seed, mask_seed=self.args.seed)
        dataset = {"net_input": net_input, "target": target}
        dataset = NestedDictionaryDataset(dataset)
        # if self.args.extract_feat:
        #     return dataset
        self.datasets[split] = dataset

    def build_model(self, args):
        from unicore import models
        model = models.build_model(args=args, task=self)
        return model
    
    
    def extract_feat(self, model):
        # self.load_dataset("train", extrat_feat=True)
        extract_dataset = torch.utils.data.DataLoader(self.datasets['train'], batch_size=32, collate_fn=self.datasets['train'].collater)
        from tqdm import tqdm
        import pickle
        
        env = lmdb.open(self.args.save_path, map_size=109951162777)
        txn = env.begin(write=True)
        
        write_count = 0
        for _, sample in enumerate(tqdm(extract_dataset)):
            sample = unicore.utils.move_to_cuda(sample)
            feats = model(**sample['net_input'])
            nfeats = feats.detach().cpu().numpy()
            
            bz = feats.shape[0]
            for i in range(bz):
                idx = write_count + i
                key = str(idx)
                smi = sample['net_input']['smi'][i]
                pocket_name = sample['net_input']['pocket_name'][i]
                write_data = {'feat': nfeats[i], 'smi': smi, 'pocket_name': pocket_name}
                # serialized_data = pickle.dumps(nfeats[i])
                serialized_data = pickle.dumps(write_data)
                txn.put(key.encode(), serialized_data)
                if idx % 1000 == 0:
                    txn.commit()
                    txn = env.begin(write=True)
            write_count += bz
            # dist = sample["net_input"]["mol_src_distance"]
            # et = sample["net_input"]["mol_src_edge_type"]
            # st = sample["net_input"]["mol_src_tokens"]
            # mol_padding_mask = st.eq(model.mol_model.padding_idx)
            # mol_x = model.mol_model.embed_tokens(st)

        txn.commit()
        env.close()
        exit(0)

    def train_step(
        self, sample, model, loss, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *loss*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~unicore.data.UnicoreDataset`.
            model (~unicore.models.BaseUnicoreModel): the model
            loss (~unicore.losses.UnicoreLoss): the loss
            optimizer (~unicore.optim.UnicoreOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        if self.args.extract_feat:
            print("???")
            model.eval()
            self.extract_feat(model)



        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = loss(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output
    