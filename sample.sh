#!/usr/bin/env bash
set -euo pipefail

# Single-complex or batch sampling with OOM-friendly knobs.
# Usage examples:
#   COMPLEX_NAME=3SML_A_P_FW1 DEVICE=cuda:0 NUM_TRAJ=100 TRAJ_BATCH_SIZE=8 ./sample.sh
#   BATCH_NAMES_FILE=complex_list.txt RENDER_COMPARE=1 ./sample.sh
#   SAMPLE_ALL=1 DEVICE=cuda:0 ./sample.sh   # every complex in --dataset_path (no --name / --names_file)
DIR='./data/TernaryDB/MGD_test'

COMPLEX_NAME="${COMPLEX_NAME:-5MN0_A_B_A8S}"
DEVICE="${DEVICE:-cuda:0}"
NUM_TRAJ="${NUM_TRAJ:-100}"
# Critical for OOM: lower this (e.g. 16 -> 8 -> 4 -> 2 -> 1)
TRAJ_BATCH_SIZE="${TRAJ_BATCH_SIZE:-8}"

SAMPLE_ALL="${SAMPLE_ALL:-1}"
BATCH_NAMES_FILE="${BATCH_NAMES_FILE:-}"
RENDER_COMPARE="${RENDER_COMPARE:-1}"
COMPARE_SPLIT="${COMPARE_SPLIT:-0}"

PY_ARGS=(
  --config configs/flow_matching_config.yaml
  --checkpoint checkpoints_0407/latest.pt
  --dataset_path data/Moloctite/TernaryDataset_test
  --pdb_base_dir data/TernaryDB/MGD_test
  --output_dir sample_outputs
  --device "${DEVICE}"
  --num_trajectories "${NUM_TRAJ}"
  --trajectory_batch_size "${TRAJ_BATCH_SIZE}"
  --ligand_candidate_topk 3
  --sefmol_refine
  --sefmol_ckpt SeFMol/ckpt/checkpoint/sefmol.pt
  --sefmol_device "${DEVICE}"
)

if [[ "${SAMPLE_ALL}" == "1" ]]; then
  if [[ -n "${BATCH_NAMES_FILE}" ]]; then
    echo "sample.sh: unset BATCH_NAMES_FILE when SAMPLE_ALL=1" >&2
    exit 1
  fi
  # sample.py: omit --name and --names_file → iterate full dataset
elif [[ -n "${BATCH_NAMES_FILE}" ]]; then
  PY_ARGS+=( --names_file "${BATCH_NAMES_FILE}" )
else
  PY_ARGS+=( --name "${COMPLEX_NAME}" )
fi

if [[ "${RENDER_COMPARE}" == "1" ]]; then
  PY_ARGS+=( --render_compare )
fi
if [[ "${COMPARE_SPLIT}" == "1" ]]; then
  PY_ARGS+=( --compare_save_split_pngs )
fi

python sample.py "${PY_ARGS[@]}"
