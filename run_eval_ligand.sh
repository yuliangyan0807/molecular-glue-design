#!/bin/bash

# Ligand evaluation launcher for eval_ligand.py
# Usage:
#   bash run_eval_ligand.sh
#   DETAILED_JSON=path/to/detailed_results.json bash run_eval_ligand.sh

set -euo pipefail

# Paths
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="${SCRIPT_PATH:-eval_ligand.py}"
DETAILED_JSON="${DETAILED_JSON:-evaluation_results/detailed_results.json}"
PDB_BASE_DIR="${PDB_BASE_DIR:-data/TernaryDB/MGD_test}"
DATASET_PATH="${DATASET_PATH:-data/Moloctite/TernaryDataset_test}"
OUTPUT_JSON="${OUTPUT_JSON:-evaluation_results/ligand_eval_results.json}"

# SeFMol refine parameters (set SEFMOL_REFINE=1 to enable)
SEFMOL_REFINE="${SEFMOL_REFINE:-1}"
SEFMOL_CKPT="${SEFMOL_CKPT:-SeFMol/ckpt/checkpoint/sefmol.pt}"
SEFMOL_DEVICE="${SEFMOL_DEVICE:-cuda:0}"
SEFMOL_TIMESTEPS="${SEFMOL_TIMESTEPS:-50}"
SEFMOL_SAMPLE_MODE="${SEFMOL_SAMPLE_MODE:-sefmol_sample}"  # sefmol_sample | rigid_sample
SEFMOL_CENTER_POS_MODE="${SEFMOL_CENTER_POS_MODE:-protein}" # none | protein | value_func_protein
# Optional space-separated list, e.g. "1 1 1 50 3.0 2.0 0.5 2"
SEFMOL_CONDITION_PROPERTIES="${SEFMOL_CONDITION_PROPERTIES:-}"

# Vina parameters
HIGH_AFFINITY_THRESHOLD="${HIGH_AFFINITY_THRESHOLD:--7.0}"
DOCK_EXHAUSTIVENESS="${DOCK_EXHAUSTIVENESS:-8}"
DOCK_N_POSES="${DOCK_N_POSES:-20}"
VINA_DEBUG="${VINA_DEBUG:-0}"  # 1 to enable --vina_debug
VINA_TMP_DIR="${VINA_TMP_DIR:-/mnt/data/tmp}"

# Put Vina temp files on large disk and clean them after evaluation.
export VINA_TMP_DIR
mkdir -p "${VINA_TMP_DIR}"

cleanup_vina_tmp() {
  rm -rf "${VINA_TMP_DIR}"/vina_* "${VINA_TMP_DIR}"/p1proc_*
}
trap cleanup_vina_tmp EXIT

CMD="${PYTHON_BIN} \"${SCRIPT_PATH}\" \
  --detailed_json \"${DETAILED_JSON}\" \
  --high_affinity_threshold \"${HIGH_AFFINITY_THRESHOLD}\" \
  --dock_exhaustiveness \"${DOCK_EXHAUSTIVENESS}\" \
  --dock_n_poses \"${DOCK_N_POSES}\""

# Enable Vina pipeline only when PDB_BASE_DIR is non-empty.
if [ -n "${PDB_BASE_DIR}" ]; then
  CMD="${CMD} --pdb_base_dir \"${PDB_BASE_DIR}\" --dataset_path \"${DATASET_PATH}\""
fi

if [ "${VINA_DEBUG}" = "1" ]; then
  CMD="${CMD} --vina_debug"
fi

if [ -n "${OUTPUT_JSON}" ]; then
  CMD="${CMD} --output_json \"${OUTPUT_JSON}\""
fi

if [ "${SEFMOL_REFINE}" = "1" ]; then
  if [ -z "${SEFMOL_CKPT}" ]; then
    echo "ERROR: SEFMOL_REFINE=1 requires SEFMOL_CKPT to a checkpoint file." >&2
    exit 1
  fi
  CMD="${CMD} --sefmol_refine --sefmol_ckpt \"${SEFMOL_CKPT}\" --sefmol_device \"${SEFMOL_DEVICE}\" --sefmol_timesteps \"${SEFMOL_TIMESTEPS}\" --sefmol_sample_mode \"${SEFMOL_SAMPLE_MODE}\" --sefmol_center_pos_mode \"${SEFMOL_CENTER_POS_MODE}\""
  if [ -n "${SEFMOL_CONDITION_PROPERTIES}" ]; then
    CMD="${CMD} --sefmol_condition_properties ${SEFMOL_CONDITION_PROPERTIES}"
  fi
fi

echo "Running ligand evaluation with:"
echo "  Python: ${PYTHON_BIN}"
echo "  Script: ${SCRIPT_PATH}"
echo "  Detailed JSON: ${DETAILED_JSON}"
echo "  PDB Base Dir: ${PDB_BASE_DIR:-<disabled>}"
echo "  Dataset Path: ${DATASET_PATH}"
echo "  High Affinity Threshold: ${HIGH_AFFINITY_THRESHOLD}"
echo "  Dock Exhaustiveness: ${DOCK_EXHAUSTIVENESS}"
echo "  Dock N Poses: ${DOCK_N_POSES}"
echo "  Vina Debug: ${VINA_DEBUG}"
echo "  Vina Tmp Dir: ${VINA_TMP_DIR}"
echo "  Output JSON: ${OUTPUT_JSON:-<auto by eval_ligand.py>}"
echo "  SeFMol Refine: ${SEFMOL_REFINE}"
if [ "${SEFMOL_REFINE}" = "1" ]; then
  echo "  SeFMol Ckpt: ${SEFMOL_CKPT}"
  echo "  SeFMol Device: ${SEFMOL_DEVICE}"
  echo "  SeFMol Timesteps: ${SEFMOL_TIMESTEPS}"
  echo "  SeFMol Sample Mode: ${SEFMOL_SAMPLE_MODE}"
  echo "  SeFMol Center Pos Mode: ${SEFMOL_CENTER_POS_MODE}"
  echo "  SeFMol Condition Properties: ${SEFMOL_CONDITION_PROPERTIES:-<default in eval_ligand.py>}"
fi
echo ""

eval "${CMD}"

echo ""
echo "Ligand evaluation finished."
