#!/bin/bash
#SBATCH --job-name=deepseek-r1-generation
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --time=04-00:00:00

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --hf-dataset)
            HF_DATASET="$2"
            shift 2
            ;;
        --hf-dataset-config)
            HF_DATASET_CONFIG="$2"
            shift 2
            ;;
        --hf-dataset-split)
            HF_DATASET_SPLIT="$2"
            shift 2
            ;;
        --prompt-column)
            PROMPT_COLUMN="$2"
            shift 2
            ;;
        --prompt-template)
            PROMPT_TEMPLATE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top-p)
            TOP_P="$2"
            shift 2
            ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --num-generations)
            NUM_GENERATIONS="$2"
            shift 2
            ;;
        --input-batch-size)
            INPUT_BATCH_SIZE="$2"
            shift 2
            ;;
        --client-replicas)
            CLIENT_REPLICAS="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --retries)
            RETRIES="$2"
            shift 2
            ;;
        --hf-output-dataset)
            HF_OUTPUT_DATASET="$2"
            shift 2
            ;;
        --private)
            PRIVATE="true"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

if [ -z "$MODEL" ] || [ -z "$HF_DATASET" ]; then
    echo "Error: --model and --hf-dataset are required parameters"
    exit 1
fi

# Set default values for optional parameters
HF_DATASET_SPLIT=${HF_DATASET_SPLIT:-"train"}
PROMPT_COLUMN=${PROMPT_COLUMN:-"prompt"}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-"{{ instruction }}"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
NUM_GENERATIONS=${NUM_GENERATIONS:-1}
INPUT_BATCH_SIZE=${INPUT_BATCH_SIZE:-64}
CLIENT_REPLICAS=${CLIENT_REPLICAS:-1}
TIMEOUT=${TIMEOUT:-900}
RETRIES=${RETRIES:-0}
PRIVATE=${PRIVATE:-"false"}


# TODO: Saving options
