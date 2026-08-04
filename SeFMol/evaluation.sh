#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ===== Sampling stage (run before evaluation) =====
# If 1: auto-run sampling when RESULT_DIR has no result_*.pt
# If 0: skip sampling stage
RUN_SAMPLE_FIRST="${RUN_SAMPLE_FIRST:-1}"

# sample.py args
SAMPLE_CONFIG="${SAMPLE_CONFIG:-${SCRIPT_DIR}/configs/sampling.yml}"
LOAD_CKPT_PATH="${LOAD_CKPT_PATH:-${SCRIPT_DIR}/ckpt/checkpoint/sefmol.pt}"
SAMPLE_DEVICE="${SAMPLE_DEVICE:-cuda:0}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-100}"
SAMPLE_MODE="${SAMPLE_MODE:-sefmol_sample}"    # rigid_sample | sefmol_sample
SAMPLE_TIMESTEPS="${SAMPLE_TIMESTEPS:-50}"
START_ID="${START_ID:-0}"
END_ID="${END_ID:-89}"

# ===== Required paths =====
# Directory that contains sampling outputs: result_*.pt (recursive).
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results}"
# Output directory for evaluation summaries/jsonl.
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/eval_results}"

# Keep protein source consistent with targetdiff/pocketxmol evaluation.
MGD_TEST_DIR="${MGD_TEST_DIR:-/home/yuliangyan/Code/Trust-App-AI-Lab/molecular_glue_design/data/TernaryDB/MGD_test}"
# Fallback root for resolving relative data.protein_filename when MGD path not found.
DATASET_ROOT="${DATASET_ROOT:-${MGD_TEST_DIR}}"

# ===== Evaluation options =====
ATOM_ENC_MODE="${ATOM_ENC_MODE:-add_aromatic}"   # basic | add_aromatic | full
MAX_SAMPLES_PER_COMPLEX="${MAX_SAMPLES_PER_COMPLEX:-50}"  # -1 means use all samples

HIGH_AFFINITY_THRESHOLD="${HIGH_AFFINITY_THRESHOLD:--7.0}"
DOCK_EXHAUSTIVENESS="${DOCK_EXHAUSTIVENESS:-8}"
DOCK_N_POSES="${DOCK_N_POSES:-20}"
VINA_DEBUG="${VINA_DEBUG:-0}"                    # 1 to enable verbose vina logs

# Optional Vina tmp dir (recommended on large runs to avoid filling root disk)
VINA_TMP_DIR="${VINA_TMP_DIR:-/mnt/data/tmp}"
mkdir -p "${VINA_TMP_DIR}"
export VINA_TMP_DIR

mkdir -p "${RESULT_DIR}"

maybe_run_sampling() {
  local existing_count
  existing_count="$(ls "${RESULT_DIR}"/result_*.pt 2>/dev/null | wc -l || true)"

  if [ "${RUN_SAMPLE_FIRST}" != "1" ]; then
    echo "[Sampling] RUN_SAMPLE_FIRST=${RUN_SAMPLE_FIRST}, skip sampling stage."
    return
  fi

  if [ "${existing_count}" -gt 0 ]; then
    echo "[Sampling] Found ${existing_count} existing result_*.pt in ${RESULT_DIR}, skip sampling."
    return
  fi

  echo "[Sampling] No result_*.pt found: sample and evaluate in one pass (TargetDiff-style)..."
  INLINE_CMD=(
    python evaluation.py
    --inline_sampling
    --sampling_config "${SAMPLE_CONFIG}"
    --load_ckpt_path "${LOAD_CKPT_PATH}"
    --start_id "${START_ID}"
    --end_id "${END_ID}"
    --device "${SAMPLE_DEVICE}"
    --batch_size "${SAMPLE_BATCH_SIZE}"
    --sample_mode "${SAMPLE_MODE}"
    --timesteps "${SAMPLE_TIMESTEPS}"
    --result_dir "${RESULT_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --dataset_root "${DATASET_ROOT}"
    --mgd_test_dir "${MGD_TEST_DIR}"
    --atom_enc_mode "${ATOM_ENC_MODE}"
    --max_samples_per_complex "${MAX_SAMPLES_PER_COMPLEX}"
    --high_affinity_threshold "${HIGH_AFFINITY_THRESHOLD}"
    --dock_exhaustiveness "${DOCK_EXHAUSTIVENESS}"
    --dock_n_poses "${DOCK_N_POSES}"
  )
  if [ "${VINA_DEBUG}" = "1" ]; then
    INLINE_CMD+=(--vina_debug)
  fi
  "${INLINE_CMD[@]}"

  echo "[Sampling] Inline sample+eval finished; skipping second evaluation pass."
  exit 0
}

CMD=(
  python evaluation.py
  --result_dir "${RESULT_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --dataset_root "${DATASET_ROOT}"
  --mgd_test_dir "${MGD_TEST_DIR}"
  --atom_enc_mode "${ATOM_ENC_MODE}"
  --max_samples_per_complex "${MAX_SAMPLES_PER_COMPLEX}"
  --high_affinity_threshold "${HIGH_AFFINITY_THRESHOLD}"
  --dock_exhaustiveness "${DOCK_EXHAUSTIVENESS}"
  --dock_n_poses "${DOCK_N_POSES}"
)

if [ "${VINA_DEBUG}" = "1" ]; then
  CMD+=(--vina_debug)
fi

echo "Sampling + SeFMol evaluation pipeline:"
echo "  RUN_SAMPLE_FIRST=${RUN_SAMPLE_FIRST}"
echo "  SAMPLE_CONFIG=${SAMPLE_CONFIG}"
echo "  LOAD_CKPT_PATH=${LOAD_CKPT_PATH}"
echo "  SAMPLE_DEVICE=${SAMPLE_DEVICE}"
echo "  SAMPLE_BATCH_SIZE=${SAMPLE_BATCH_SIZE}"
echo "  SAMPLE_MODE=${SAMPLE_MODE}"
echo "  SAMPLE_TIMESTEPS=${SAMPLE_TIMESTEPS}"
echo "  START_ID=${START_ID}"
echo "  END_ID=${END_ID}"
echo

maybe_run_sampling

echo "Running SeFMol evaluation:"
echo "  RESULT_DIR=${RESULT_DIR}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  MGD_TEST_DIR=${MGD_TEST_DIR}"
echo "  DATASET_ROOT=${DATASET_ROOT}"
echo "  ATOM_ENC_MODE=${ATOM_ENC_MODE}"
echo "  MAX_SAMPLES_PER_COMPLEX=${MAX_SAMPLES_PER_COMPLEX}"
echo "  HIGH_AFFINITY_THRESHOLD=${HIGH_AFFINITY_THRESHOLD}"
echo "  DOCK_EXHAUSTIVENESS=${DOCK_EXHAUSTIVENESS}"
echo "  DOCK_N_POSES=${DOCK_N_POSES}"
echo "  VINA_DEBUG=${VINA_DEBUG}"
echo "  VINA_TMP_DIR=${VINA_TMP_DIR}"
echo

"${CMD[@]}"

echo
echo "Done. Outputs:"
echo "  ${OUTPUT_DIR}/dataset_summary.json"
echo "  ${OUTPUT_DIR}/per_complex_summary.json"
echo "  ${OUTPUT_DIR}/global_summary.json"
