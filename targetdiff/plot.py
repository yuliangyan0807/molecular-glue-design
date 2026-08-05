import os
import json
import numpy as np

results_dir = 'outputs_mgd_test/dataset_summary.json'

ligand_rmsd = []
seq_acc = []
with open(results_dir, 'r') as f:
    data = json.load(f)
    for item in data['items']:
        ligand_rmsd.append(item['eval_summary']['ligand_rmsd_mean'])
        seq_acc.append(item['eval_summary']['seq_acc_mean'])

print(f"Ligand RMSD: {np.mean(ligand_rmsd)}, {np.std(ligand_rmsd)}")
print(f"Seq Acc: {np.mean(seq_acc)}, {np.std(seq_acc)}")