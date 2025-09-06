#!/usr/bin/env bash
set -euo pipefail

# Configuration
NPROC=${NPROC:-1}
DATASET_DIR=${DATASET_DIR:-"./interface_modeling_dataset"}
LOG_DIR=${LOG_DIR:-"./logs"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"./checkpoints"}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-1e-4}
EPOCHS=${EPOCHS:-500}
PROJECT=${PROJECT:-"interface_model"}
NAME=${NAME:-"interface_model_$(date +%Y%m%d-%H%M%S)"}

# Print config
echo "Launching training with:"
echo "  NPROC         = ${NPROC}"
echo "  DATASET_DIR   = ${DATASET_DIR}"
echo "  LOG_DIR       = ${LOG_DIR}"
echo "  CHECKPOINT_DIR= ${CHECKPOINT_DIR}"
echo "  BATCH_SIZE    = ${BATCH_SIZE}"
echo "  LR            = ${LR}"
echo "  EPOCHS        = ${EPOCHS}"
echo "  PROJECT       = ${PROJECT}"
echo "  NAME          = ${NAME}"

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

# Launch
if [[ "${NPROC}" -le 1 ]]; then
  python interface_training.py \
    --dataset_dir "${DATASET_DIR}" \
    --log_dir "${LOG_DIR}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --epochs "${EPOCHS}" \
    --project "${PROJECT}" \
    --name "${NAME}"
else
  torchrun --standalone --nproc_per_node="${NPROC}" interface_training.py \
    --dataset_dir "${DATASET_DIR}" \
    --log_dir "${LOG_DIR}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --epochs "${EPOCHS}" \
    --project "${PROJECT}" \
    --name "${NAME}"
fi
