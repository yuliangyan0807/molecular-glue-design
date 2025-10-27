#!/bin/bash
# Distributed training script for flow_training.py using torchrun

# Default parameters
NUM_NODES=1
NUM_GPUS=8
CUDA_VISIBLE="0,1,2,3,4,5,6,7"
MASTER_ADDR="localhost"
MASTER_PORT="29500"

# Parse script arguments
PYTHON_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --gpus)
            CUDA_VISIBLE="$2"
            NUM_GPUS=$(echo $CUDA_VISIBLE | tr ',' '\n' | wc -l)
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS] [PYTHON ARGS]"
            echo ""
            echo "Script Options:"
            echo "  --num_gpus N      Number of GPUs to use (default: 8)"
            echo "  --gpus LIST       Comma-separated list of GPU IDs (e.g., 0,1,2,3)"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Python Script Args:"
            echo "  --batch_size N    Batch size per GPU"
            echo "  --lr FLOAT        Learning rate"
            echo "  --dataset PATH    Dataset path"
            echo "  --resume PATH     Resume from checkpoint"
            echo "  --no_wandb        Disable wandb logging"
            echo "  See 'python flow_training.py --help' for more options"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Use 8 GPUs with default config"
            echo "  $0 --num_gpus 4                      # Use 4 GPUs"
            echo "  $0 --gpus 0,1,2,3 --batch_size 16    # Use specific GPUs with custom batch size"
            echo "  $0 --num_gpus 2 --lr 1e-4            # Use 2 GPUs with custom learning rate"
            exit 0
            ;;
        *)
            PYTHON_ARGS="$PYTHON_ARGS $1"
            shift
            ;;
    esac
done

# Set up environment
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# Check if GPUs are available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found!"
    exit 1
fi

nvidia-smi --list-gpus > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Error: No NVIDIA GPUs detected!"
    exit 1
fi

# Print configuration
echo "================================================"
echo "Distributed Training Configuration"
echo "================================================"
echo "GPU IDs:       $CUDA_VISIBLE"
echo "Num GPUs:      $NUM_GPUS"
echo "Num Nodes:     $NUM_NODES"
echo "Master:        $MASTER_ADDR:$MASTER_PORT"
echo "================================================"
echo ""

# Run distributed training
torchrun \
    --nnodes=${NUM_NODES} \
    --nproc_per_node=${NUM_GPUS} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    flow_training.py ${PYTHON_ARGS}

EXIT_CODE=$?

echo ""
echo "================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code $EXIT_CODE"
fi
echo "================================================"

exit $EXIT_CODE