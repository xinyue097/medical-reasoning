#!/bin/bash
#SBATCH --job-name=medreason
#SBATCH -p gpu
#SBATCH --output=../../logs/train_%j.out
#SBATCH --error=../../logs/train_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:4
#SBATCH --mail-type=ALL

# Load modules
module purge
module load miniconda
conda activate med1

echo "=== ENVIRONMENT SETUP ==="
# Environment variables
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export TORCH_CUDNN_BENCHMARK=true
export TORCH_ALLOW_TF32_CUDNN_OVERRIDE=1
export DEEPSPEED_NVCC_FLAGS="-gencode arch=compute_80,code=sm_80"
export DEEPSPEED_CUDA_ARCH="8.0"
export NCCL_DEBUG=ERROR
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_TREE_THRESHOLD=0
export OMP_NUM_THREADS=8
export TRITON_CACHE_DIR="/tmp/triton_cache_$$"
export HF_DATASETS_CACHE="/tmp/hf_datasets_cache_$$"
export HF_HOME="/tmp/hf_home_$$"
export TRANSFORMERS_CACHE="/tmp/transformers_cache_$$"
export WANDB_API_KEY="dcb0e216ebbdf52149865275d6cff550b91f3ca1"
export WANDB_PROJECT="med_qwen3_8B"
export WANDB_RUN_NAME="med_qwen3_8B-$(date +%Y%m%d)"
export MASTER_ADDR=localhost
export MASTER_PORT=12355

mkdir -p "$TRITON_CACHE_DIR" "$HF_DATASETS_CACHE" "$HF_HOME" "$TRANSFORMERS_CACHE"
echo "=================================================================="
echo "Start Training"
echo "=================================================================="

# Launch training
accelerate launch \
    --config_file ../accelerate_configs/zero2.yaml \
    --main_process_port 12355 \
    ../../src/medreason/grpo.py configs.yaml

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
rm -rf "$TRITON_CACHE_DIR" "$HF_DATASETS_CACHE" "$HF_HOME" "$TRANSFORMERS_CACHE" 2>/dev/null || true

exit $exit_code