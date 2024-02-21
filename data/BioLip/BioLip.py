import os
import pandas as pd
import lmdb
import pickle as pkl
import numpy as np
from tqdm import tqdm
import re


# Record BioLip data with lmdb
env1 = lmdb.open('./lmdb', map_size=10099511627776)
txn1 = env1.begin(write=True)
complex_idx = 0

file = "./BioLiP_nomalMolecular.txt"
atom3d_pocket_PDB_folder = "./BioLiP_atom3d_pocket_PDB"
biolip_pocket_PDB_folder = "./BioLiP_pocket_PDB"

no_atom3d_line = open("./no_atom3d_line.txt", 'w')

protein_atom_set = set()
ligand_atom_set = set()

with open(file, 'r') as infile:
	for line_1 in infile:
		try:
			protein_name = line_1.split('\t')[0]
			protein_chain = line_1.split('\t')[1]
			ligand_name = line_1.split('\t')[4]
			ligand_chain = line_1.split('\t')[5]
			Ligand_serial_number = line_1.split('\t')[6]
			protein_file_path = os.path.join("./BioLiP_updated_set/receptor", f"{protein_name}{protein_chain}.pdb")
			ligand_file_path = os.path.join("./BioLiP_updated_set/ligand", f"{protein_name}_{ligand_name}_{ligand_chain}_{Ligand_serial_number}.pdb")

			# Get ligand information
			ligand_coordinates_list = list()
			ligand_atoms_list = list()
			with open(ligand_file_path, 'r') as infile:
				for line in infile:
					if line.startswith('TER'):
						continue
					coordinates_x = float(line[30:38].strip())
					coordinates_y = float(line[38:46].strip())
					coordinates_z = float(line[46:54].strip())
					element_symbol = str(line[76:78].strip())
					ligand_coordinates_list.append([coordinates_x, coordinates_y, coordinates_z])
					ligand_atoms_list.append(element_symbol)
			ligand_coordinates_array = np.array(ligand_coordinates_list).astype(np.float32)
			ligand_atom_set = ligand_atom_set | set(ligand_atoms_list)

			# ligand center coordinate, used to get protein pocket residue
			ligand_center_coor = ligand_coordinates_array.mean(axis=0)
			ligand_sphere_radius = np.sqrt(np.sum(np.square(ligand_coordinates_array - ligand_center_coor), axis=1)).mean()

			# Get pocket information
			protein_coordinates_list = list()
			protein_atoms_list = list()
			residue_num_list = list()
			retain_residue_list = list()
			with open(protein_file_path, 'r') as infile:
				for line in infile:
					if line.startswith('TER'):
						continue
					residue_num = int(line[22:26].strip())
					coordinates_x = float(line[30:38].strip())
					coordinates_y = float(line[38:46].strip())
					coordinates_z = float(line[46:54].strip())
					element_symbol = str(line[76:78].strip())
					# See if distance to ligand < 8. Record residue num and distance
					if str(line[12:16].strip()) == 'CA':
						CA_coor = np.array([coordinates_x, coordinates_y, coordinates_z]).astype(np.float32)
						dist = np.sqrt(np.sum(np.square(CA_coor - ligand_center_coor)))
						if dist <= (8 + ligand_sphere_radius):
							retain_residue_list.append( (residue_num, dist) )
					residue_num_list.append(residue_num)
					protein_coordinates_list.append([coordinates_x, coordinates_y, coordinates_z])
					protein_atoms_list.append(element_symbol)
			protein_coordinates_array = np.array(protein_coordinates_list).astype(np.float32)
			protein_atom_set = protein_atom_set | set(protein_atoms_list)
			# use while to filter farthest residue if Atom num > 600
			residue_num_df = pd.DataFrame(residue_num_list)
			pocket_atom_num = residue_num_df[residue_num_df[0].isin(set([x[0] for x in retain_residue_list]))==True].index
			while (len(pocket_atom_num) + ligand_coordinates_array.shape[0]) > 600:
				retain_residue_list = sorted(retain_residue_list, key=lambda x:x[1])[:-1]
				pocket_atom_num = pd.DataFrame(residue_num_list)[pd.Series(residue_num_list).isin(set([x[0] for x in retain_residue_list]))==True].index
			# pocket_coordinates_array Record the Coordinate in pocket
			pocket_coordinates_array = protein_coordinates_array[pocket_atom_num]
			pocket_atoms_list = list(np.array(protein_atoms_list)[pocket_atom_num])
			if len(pocket_atoms_list) == 0:
				print(line_1.strip(), file=no_atom3d_line)
			else:
				data = dict()
				data['pocket_atoms'] = pocket_atoms_list
				data['pocket_coordinates'] = pocket_coordinates_array
				data['lig_atoms_real'] = ligand_atoms_list
				data['lig_coord_real'] = ligand_coordinates_list
				key = str(complex_idx)
				serialized_data = pkl.dumps(data)
				txn1.put(key.encode(), serialized_data)
		except:
			print('ERROR: ', line_1.strip(), file=no_atom3d_line)


no_atom3d_line.close()
txn1.commit()
env1.close()