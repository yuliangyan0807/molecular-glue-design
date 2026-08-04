# Molecular Glue Design

Ternary complex generation with a flow-matching model (protein–protein docking + ligand atom types/coords), optional SeFMol ligand refinement, and DockQ / Vina evaluation.

## Environment

Recommended: conda env `mgd`.

```bash
conda activate mgd
# Key packages used in our runs (approx.):
# torch 2.12 + CUDA, rdkit, datasets, biopandas, biopython, scipy, wandb, pyyaml, tqdm
```

GPU recommended for train/eval. Multi-GPU via `torchrun`.

## Checkpoints

| Role | Path | Notes |
|------|------|--------|
| **Flow model (eval/sample)** | `checkpoints_0407/latest.pt` | **Epoch 800** final weights (used by all eval/sample scripts) |
| Flow model (val-best, unused in current eval) | `checkpoints_0407/best.pt` | Epoch 421, lowest val RMSD during training |
| Interface encoder (train init) | `checkpoints/interface-model/latest.pt` | Frozen during flow training; also referenced in config |
| SeFMol refine (optional) | `SeFMol/ckpt/checkpoint/sefmol.pt` | Used by `sample.sh` / `run_eval_ligand.sh` |

Weights are **not** shipped in this repository. Download them and place files at the paths above:

- Flow + interface checkpoints: *[TODO: add download link]*
- SeFMol checkpoint (`sefmol.pt`): *[TODO: add download link]*

```bash
mkdir -p SeFMol/ckpt/checkpoint checkpoints_0407 checkpoints/interface-model
# after download:
# mv /path/to/sefmol.pt SeFMol/ckpt/checkpoint/sefmol.pt
# mv /path/to/latest.pt checkpoints_0407/latest.pt
# mv /path/to/interface_latest.pt checkpoints/interface-model/latest.pt
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

This is the main evaluation path used for `evaluation_results_0422`:

```bash
bash run_evaluation.sh
```

Equivalent settings:

- Checkpoint: `checkpoints_0407/latest.pt`
- Dataset: `data/Moloctite/TernaryDataset_test`
- PDB dir: `data/TernaryDB/MGD_test`
- `num_trajectories_per_sample=100`, `trajectory_batch_size=2`, `seed=42`
- Multi-GPU: `GPU_IDS=0,1,2,3,4,5,6,7`

Outputs go to `evaluation_results_0422/` (`summary_metrics.json`, `detailed_results.json`, …).

Quick / ablation-style eval (fewer trajs):

```bash
bash ablation_study.sh   # also uses checkpoints_0407/latest.pt, 20 trajs
```

## 4. Ligand evaluation (+ SeFMol refine, Vina)

Depends on step 3’s `detailed_results.json`:

```bash
DETAILED_JSON=evaluation_results_0422/detailed_results.json \
OUTPUT_JSON=evaluation_results_0422/ligand_eval_results_0428.json \
SEFMOL_REFINE=1 \
SEFMOL_CKPT=SeFMol/ckpt/checkpoint/sefmol.pt \
bash run_eval_ligand.sh
```

Optional merge of best DockQ + ligand metrics:

```bash
python export_per_complex_best_metrics.py \
  --detailed_json evaluation_results_0422/detailed_results.json \
  --ligand_json evaluation_results_0422/ligand_eval_results_0428.json \
  --output_json evaluation_results_0422/per_complex_best_metrics.json
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

1. Place weights at `checkpoints_0407/latest.pt` (and interface ckpt if re-training).
2. Place test data under the paths above.
3. `bash run_evaluation.sh` → protein/complex metrics.
4. `bash run_eval_ligand.sh` → ligand / Vina (with SeFMol).

Do **not** swap in `best.pt` unless you intentionally want the val-RMSD checkpoint; current scripts and results use **`latest.pt` @ epoch 800**.

## Project entrypoints

| Script | Purpose |
|--------|---------|
| `train_interface.sh` / `interface_training.py` | Interface pretraining |
| `train_flow.sh` / `flow_training.py` | Ternary flow training |
| `run_evaluation.sh` / `evaluation.py` | Batch test eval (DockQ etc.) |
| `run_eval_ligand.sh` / `eval_ligand.py` | Ligand + SeFMol + Vina |
| `sample.sh` / `sample.py` | Sampling + optional refine/render |
