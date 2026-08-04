# Molecular Glue Design

Ternary complex generation with a flow-matching model (protein–protein docking + ligand atom types/coords), optional SeFMol ligand refinement, and DockQ / Vina evaluation.

**Code:** https://github.com/yuliangyan0807/molecular-glue-design

## Environment

Python 3.10, CUDA 12.4. GPU recommended for train/eval. Multi-GPU via `torchrun`.

```
torch==2.6.0
torch-geometric==2.6.1
torch_scatter==2.1.2+pt26cu124
torch_sparse==0.6.18+pt26cu124
torch_cluster==1.6.3+pt26cu124
lightning==2.6.1
pytorch-lightning==2.6.0
rdkit==2023.9.3
numpy==1.24.0
scipy==1.15.3
pandas==1.5.2
biopython==1.83
biopandas==0.5.1
datasets==4.8.4
einops==0.8.2
e3nn==0.6.0
easydict==1.9
lmdb==1.7.5
networkx==3.4.2
openbabel-wheel==3.1.1.11
PyYAML==6.0.3
tqdm==4.67.3
scikit-learn==1.7.2
vina==1.2.7
meeko==0.1.dev3
pillow==12.2.0
pymol-open-source==3.2.0a0
huggingface_hub==1.10.1
```

## Checkpoints

| Role | Path | Notes |
|------|------|--------|
| **Flow model (eval/sample)** | `checkpoints_0407/latest.pt` | **Epoch 800** final weights (used by all eval/sample scripts) |
| Flow model (val-best, unused in current eval) | `checkpoints_0407/best.pt` | Epoch 421, lowest val RMSD during training |
| Interface encoder (train init) | `checkpoints/interface-model/latest.pt` | Frozen during flow training; also referenced in config |
| SeFMol refine (optional) | `SeFMol/ckpt/checkpoint/sefmol.pt` | Used by `sample.sh` / `run_eval_ligand.sh` |

Weights and datasets are **not** shipped in this repository. Download from Google Drive:

https://drive.google.com/drive/folders/1L0Yxdl5cAOQAxO1cbaHtflLueYH9pcRu

| Drive file | Local path after download / extract |
|------------|-------------------------------------|
| `flow_latest.pt` | `checkpoints_0407/latest.pt` |
| `flow_best.pt` | `checkpoints_0407/best.pt` |
| `interface_latest.pt` | `checkpoints/interface-model/latest.pt` |
| `sefmol.pt` | `SeFMol/ckpt/checkpoint/sefmol.pt` |
| `TernaryDataset_test.tar.gz` | `data/Moloctite/TernaryDataset_test/` |
| `MGD_test.tar.gz` | `data/TernaryDB/MGD_test/` |
| `TernaryDataset_filtered.tar.gz` | `data/Moloctite/TernaryDataset_filtered/` |
| `interface_modeling_dataset_1208.tar.gz` | `data/Moloctite/interface_modeling_dataset_1208/` |

```bash
mkdir -p SeFMol/ckpt/checkpoint checkpoints_0407 checkpoints/interface-model data/Moloctite data/TernaryDB

mv flow_latest.pt checkpoints_0407/latest.pt
mv flow_best.pt checkpoints_0407/best.pt
mv interface_latest.pt checkpoints/interface-model/latest.pt
mv sefmol.pt SeFMol/ckpt/checkpoint/sefmol.pt

tar -xzf TernaryDataset_test.tar.gz -C data/Moloctite
tar -xzf MGD_test.tar.gz -C data/TernaryDB
tar -xzf TernaryDataset_filtered.tar.gz -C data/Moloctite
tar -xzf interface_modeling_dataset_1208.tar.gz -C data/Moloctite
```

Config: `configs/flow_matching_config.yaml`  
(`model.interface_model.path` → `./checkpoints/interface-model/latest.pt`, `trainable: false`)

SeFMol source lives under `SeFMol/` (code only; see `SeFMol/README.md` for the upstream project).

## Data layout

```
data/Moloctite/TernaryDataset_filtered   # train (~20.7k), HuggingFace datasets on disk
data/Moloctite/TernaryDataset_test       # test (~92)
data/Moloctite/interface_modeling_dataset_1208  # interface pretrain (~19.1k)
data/TernaryDB/MGD_test                  # PDB files for DockQ / Vina
```

Train/val split is done in code from `TernaryDataset_filtered` (`train_split: 0.95`, `val_split: 0.05`, seed `42`).

## 1. Train interface model (optional if you already have the ckpt)

```bash
# defaults in train_interface.sh
NPROC=8 \
DATASET_DIR=./data/Moloctite/interface_modeling_dataset_1208 \
CHECKPOINT_DIR=./checkpoints/interface-model \
bash train_interface.sh
```

Then point `configs/flow_matching_config.yaml` → `model.interface_model.path` at that `latest.pt`.

## 2. Train ternary flow model

Hyperparameters (from config):

- `max_epochs: 800`, `batch_size: 4`, `lr: 5e-4` → `end_lr: 4e-4` (linear)
- Loss weights: `trans=0.008`, `rot=1.0`, `seqs=1.0`, `coords=10.0`
- Sampling steps: `num_timesteps: 50`
- Checkpoint every 50 epochs + `latest.pt`; `best.pt` when val RMSD improves

```bash
# 8-GPU example; write into a fixed dir for reproducibility
bash train_flow.sh --gpus 0,1,2,3,4,5,6,7 \
  --config configs/flow_matching_config.yaml \
  --checkpoint_dir checkpoints_0407 \
  --dataset ./data/Moloctite/TernaryDataset_filtered
```

For a paper-style run matching our eval, keep / download **`checkpoints_0407/latest.pt`** (epoch 800).

## 3. Evaluate (protein metrics: RMSD / DockQ / …)

```bash
bash run_evaluation.sh
```

Equivalent settings:

- Checkpoint: `checkpoints_0407/latest.pt`
- Dataset: `data/Moloctite/TernaryDataset_test`
- PDB dir: `data/TernaryDB/MGD_test`
- `num_trajectories_per_sample=100`, `trajectory_batch_size=2`, `seed=42`
- Multi-GPU: `GPU_IDS=0,1,2,3,4,5,6,7`

Outputs go to `evaluation_results/` (`summary_metrics.json`, `detailed_results.json`, …).

## 4. Ligand evaluation (+ SeFMol refine, Vina)

Depends on step 3’s `detailed_results.json`:

```bash
DETAILED_JSON=evaluation_results/detailed_results.json \
OUTPUT_JSON=evaluation_results/ligand_eval_results.json \
SEFMOL_REFINE=1 \
SEFMOL_CKPT=SeFMol/ckpt/checkpoint/sefmol.pt \
bash run_eval_ligand.sh
```

Optional merge of best DockQ + ligand metrics:

```bash
python export_per_complex_best_metrics.py \
  --detailed_json evaluation_results/detailed_results.json \
  --ligand_json evaluation_results/ligand_eval_results.json \
  --output_json evaluation_results/per_complex_best_metrics.json
```

## 5. End-to-end sampling (flow + SeFMol)

```bash
# all test complexes; OOM → lower TRAJ_BATCH_SIZE
SAMPLE_ALL=1 DEVICE=cuda:0 NUM_TRAJ=100 TRAJ_BATCH_SIZE=8 bash sample.sh
```

Hard-coded in `sample.sh`:

- `--checkpoint checkpoints_0407/latest.pt`
- `--sefmol_ckpt SeFMol/ckpt/checkpoint/sefmol.pt`
- `--sefmol_refine`

Single complex:

```bash
SAMPLE_ALL=0 COMPLEX_NAME=5MN0_A_B_A8S DEVICE=cuda:0 bash sample.sh
```

## Reproduce our reported eval numbers

1. Download weights and data from the Google Drive folder above and place/extract as listed.
2. `bash run_evaluation.sh` → protein/complex metrics.
3. `bash run_eval_ligand.sh` → ligand / Vina (with SeFMol).

Do **not** swap in `best.pt` unless you intentionally want the val-RMSD checkpoint; current scripts and results use **`latest.pt` @ epoch 800**.

## Project entrypoints

| Script | Purpose |
|--------|---------|
| `train_interface.sh` / `interface_training.py` | Interface pretraining |
| `train_flow.sh` / `flow_training.py` | Ternary flow training |
| `run_evaluation.sh` / `evaluation.py` | Batch test eval (DockQ etc.) |
| `run_eval_ligand.sh` / `eval_ligand.py` | Ligand + SeFMol + Vina |
| `sample.sh` / `sample.py` | Sampling + optional refine/render |

## Citation

If you find this work useful, please cite:

```bibtex
@article{yan2026triglue,
  title   = {TriGlue: a Biology-Inspired Generative Model for Generating Molecular Glue-Induced Ternary Complex},
  author  = {Yan, Yuliang and Yan, Shuo and Tang, Haochun and Sun, Yiqin and Dai, Enyan},
  journal = {arXiv preprint arXiv:2607.22143},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.22143}
}

@article{zhang2026sefmol,
  title   = {Steering Semi-Flexible Molecular Diffusion Model for Structure-Based Drug Design with Reinforcement Learning},
  author  = {Zhang, Xudong and Qu, Sanqing and Lu, Fan and Wang, Jianmin and Tian, Zhixin and Gu, Shangding and Zhang, Yanping and Knoll, Alois and Gao, Shaorong and Chen, Guang and Jiang, Changjun},
  journal = {Science Advances},
  volume  = {12},
  number  = {16},
  year    = {2026},
  doi     = {10.1126/sciadv.ady9955},
  url     = {https://www.science.org/doi/10.1126/sciadv.ady9955}
}
```
