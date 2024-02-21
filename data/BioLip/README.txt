BioLip.py: used to build the lmdb from BioLip raw data. The lmdb raw data should be download manually from: https://zhanggroup.org/BioLiP/index.cgi. And the downloaded folder which is named "BioLiP_updated_set" on 2023.11.20 should be moved in this folder.

All the other file in this folder is used in pretrain. In principle all these data could be ignored.

cannot_converge_mol_lst.npy, failed_align_lst.npy, not_exits_lst.npy: BioLip item whose ligand can't generate pdb from smile.
BioLip_atom_num.npy: record the atom num for each BioLip item. Used in pretrain.

complex_info.txt: the column information about BioLip data. Used in pretrain.

remain_line.txt: recorded the BioLip item which are used in BindNet. (more filter information is shown in paper)
