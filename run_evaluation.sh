#!/bin/bash

# Evaluation script for Ternary Flow Matching Model
# Usage: bash run_evaluation.sh

# Set paths (modify these according to your setup)
CONFIG_PATH="configs/flow_matching_config.yaml"  # Path to config YAML file
# CHECKPOINT_PATH="checkpoints/flow-model/0317latest.pt"  # Path to model checkpoint
CHECKPOINT_PATH="checkpoints_0407/latest.pt"  # Path to model checkpoint
DATASET_PATH="data/Moloctite/TernaryDataset_test"  # Path to dataset directory
PDB_BASE_DIR="data/TernaryDB/MGD_test"  # Base directory containing PDB files
OUTPUT_DIR="./evaluation_results"  # Output directory for results

# Evaluation parameters
DEVICE="cuda"  # Device to use: cuda or cpu
GPU_IDS="0,1,2,3,4,5,6,7"  # Comma-separated GPU IDs (e.g., "0,1,2,3"). If empty, uses all available GPUs.
NUM_SAMPLES=""  # Number of samples to evaluate (empty = all, or set to a number like "50")
NUM_TRAJECTORIES_PER_SAMPLE=100  # Total trajectories per input sample
TRAJECTORY_BATCH_SIZE=2        # Max trajectories per model.sample() (GPU micro-batch); empty = one batch of NUM_TRAJECTORIES_PER_SAMPLE
SEED=42  # Random seed

# Optional: Save individual predictions
# SAVE_PREDICTIONS="--save_predictions"  # Uncomment to save predictions
SAVE_PREDICTIONS=""

# Build command
CMD="python evaluation.py \
    --config \"${CONFIG_PATH}\" \
    --checkpoint \"${CHECKPOINT_PATH}\" \
    --dataset_path \"${DATASET_PATH}\" \
    --pdb_base_dir \"${PDB_BASE_DIR}\" \
    --output_dir \"${OUTPUT_DIR}\" \
    --device \"${DEVICE}\" \
    --num_trajectories_per_sample \"${NUM_TRAJECTORIES_PER_SAMPLE}\" \
    --seed \"${SEED}\""

if [ -n "${TRAJECTORY_BATCH_SIZE}" ]; then
    CMD="${CMD} --trajectory_batch_size \"${TRAJECTORY_BATCH_SIZE}\""
fi

# Add GPU IDs if specified
if [ -n "${GPU_IDS}" ]; then
    CMD="${CMD} --gpu_ids \"${GPU_IDS}\""
fi

# Add num_samples if specified
if [ -n "${NUM_SAMPLES}" ]; then
    CMD="${CMD} --num_samples \"${NUM_SAMPLES}\""
fi

# Add save_predictions if specified
if [ -n "${SAVE_PREDICTIONS}" ]; then
    CMD="${CMD} ${SAVE_PREDICTIONS}"
fi

# Run evaluation
echo "Running evaluation with the following parameters:"
echo "  Config: ${CONFIG_PATH}"
echo "  Checkpoint: ${CHECKPOINT_PATH}"
echo "  Dataset: ${DATASET_PATH}"
echo "  PDB Base Dir: ${PDB_BASE_DIR}"
echo "  Output Dir: ${OUTPUT_DIR}"
echo "  Device: ${DEVICE}"
echo "  GPU IDs: ${GPU_IDS:-auto (all available)}"
echo "  Num Samples: ${NUM_SAMPLES:-all}"
echo "  Num Trajectories per Sample: ${NUM_TRAJECTORIES_PER_SAMPLE}"
echo "  Trajectory Batch Size: ${TRAJECTORY_BATCH_SIZE:-all at once}"
echo "  Seed: ${SEED}"
echo ""

eval ${CMD}

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Evaluation completed! Results saved to ${OUTPUT_DIR}"
else
    echo ""
    echo "✗ Evaluation failed!"
    exit 1
fi

