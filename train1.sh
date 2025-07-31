#!/bin/bash
#SBATCH --job-name=medreason
#SBATCH -p gpu
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:2
#SBATCH --mail-type=ALL

# Load modules
module purge
module load miniconda
module load GCC/12.2.0
module load CUDA/12.4

conda activate med1

echo "=== ENVIRONMENT SETUP ==="
# Set GPU visibility
export TOKENIZERS_PARALLELISM=false

# PyTorch optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export TORCH_CUDNN_BENCHMARK=true
export TORCH_ALLOW_TF32_CUDNN_OVERRIDE=1

# DeepSpeed optimizations
export DEEPSPEED_NVCC_FLAGS="-gencode arch=compute_80,code=sm_80"
export DEEPSPEED_CUDA_ARCH="8.0"

# Communication backend
export NCCL_DEBUG=ERROR
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_TREE_THRESHOLD=0

# Control the parallelism of CPU-bound computations
export OMP_NUM_THREADS=16

# Fix Triton cache warning
export TRITON_CACHE_DIR="/tmp/triton_cache_$$"
mkdir -p "$TRITON_CACHE_DIR"

# W&B Setup
export WANDB_API_KEY="dcb0e216ebbdf52149865275d6cff550b91f3ca1"
export WANDB_PROJECT="medreason"
export WANDB_RUN_NAME="medreason-$(date +%Y%m%d)"

# Find script
if [ -f "src/medreason/grpo1.py" ]; then
    SCRIPT_PATH="src/medreason/grpo1.py"
else
    echo "❌ GRPO script not found"
    exit 1
fi

# Verify config exists
if [ ! -f "config1.yaml" ]; then
    echo "❌ Config not found"
    exit 1
fi

# Check accelerate config
if [ ! -f "accelerate_config.yaml" ]; then
    echo "❌ accelerate_config.yaml not found in current directory"
    exit 1
fi

# Final status
echo "🔍 Final Launch Status:"
echo "  Script: $SCRIPT_PATH"
echo "  Config: config1.yaml"
echo "  GPUs: 2"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Starting training..."

# Set additional distributed training variables
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# Launch training
accelerate launch \
    --config_file accelerate_config.yaml \
    --main_process_port 12355 \
    "$SCRIPT_PATH" config1.yaml

exit_code=$?

echo "=================================================================="
echo "Training finished at $(date)"
echo "Exit code: $exit_code"
echo "=================================================================="

if [ $exit_code -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed"
fi

# Cleanup
rm -rf "$TRITON_CACHE_DIR" 2>/dev/null || true

exit $exit_code