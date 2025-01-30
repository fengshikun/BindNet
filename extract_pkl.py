import pickle


file_path = '/mnt/nfs-ssd/data/moyuanhuan/DiffDockData/diffDock_dataset_matching/valid_cache/protein_ligand_df.pkl'
# file_path = '/mnt/nfs-ssd/data/moyuanhuan/DiffDockData/diffDock_dataset_matching/train_cache/protein_ligand_df.pkl'
with open(file_path,'rb') as f:
    protein_ligand_df  = pickle.load(f)
    
pkl_len = len(protein_ligand_df)

# iterate over the pkl file

start_write = False
write_cnt = 0
write_lst = []
for i in range(pkl_len):
    data_item = protein_ligand_df.iloc[i]
    # data_item = self.protein_ligand_df.iloc[idx, :]
    complex_graph = data_item['complex_graph']
    # print(complex_graph['name'])
    if complex_graph['name'] == '5j9y_protein_processed_fix.pdb___5j9y_ligand.sdf':
        start_write = True
        print(complex_graph)
        # print(f"Error with {complex_graph['name']}")
    # do something with the data
    
    if start_write:
        write_lst.append(data_item)
        write_cnt += 1
    
    if write_cnt > 10:
        break
        
    print(complex_graph['name'])

# convert the data to a pandas dataframe
import pandas as pd
write_df = pd.DataFrame(write_lst)
# save write_df to a new pkl file
with open('5j9y_data_df.pkl', 'wb') as f:
    pickle.dump(write_df, f)