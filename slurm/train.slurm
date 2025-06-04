#!/bin/bash
#SBATCH --job-name=open_r1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --partition=p      # Adjust
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=3-00:00:00


if [[ "$*" == *"--help"* ]]; then
  echo "Usage: sbatch slurm/train.slurm [options]"
  echo "Options:"
  echo "  --model MODEL            Model name"
  echo "  --task TASK              Task name (e.g. sft, grpo)"
  echo "  --config SUFFIX          Configuration suffix (e.g. demo, v00.00)"
  echo "  --accelerator CONFIG     Accelerator configuration name (e.g. zero3)"
  echo "  --dp N                   Data parallelism for vLLM server (default: 1)"
  echo "  --tp N                   Tensor parallelism for vLLM server (default: 1)"
  echo "  --args \"ARGS\"          Optional arguments to pass to the training script"
  exit 0
fi

# Specific configuration optimized for the Hugging Face Compute Cluster
module load cuda/12.4
set -x -e


# Default values
MODEL=""
TASK=""
CONFIG_SUFFIX=""
ACCELERATOR=""
DP=1
TP=1
OPTIONAL_ARGS=""

done
