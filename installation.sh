#!/bin/bash

module purge
module load miniconda
module load GCC/12.2.0
module load CUDA/12.4

conda config --set solver libmamba
conda create --name medenv python=3.11

# conda install -c conda-forge -y \
#     "pandas>=2.2.3" \
#     "packaging>=23.0" \
#     "pytest" \
#     "einops>=0.8.0" \
#     "datasets>=3.2.0" \
#     "safetensors>=0.3.3" \
#     "sentencepiece>=0.1.99" \
#     "wandb>=0.19.1" \
#     "langdetect" \
#     "async-lru>=2.0.5"

conda install -c conda-forge -y \
    "pandas>=2.2.3" \
    "packaging>=23.0" \
    "pytest" \
    "einops>=0.8.0" \
    "datasets>=3.2.0" \
    "safetensors>=0.3.3" \
    "sentencepiece>=0.1.99" \
    "wandb>=0.19.1" \
    "isort>=5.12.0" \
    "flake8>=6.0.0" \
    "parameterized>=0.9.0" \
    "jieba" \
    "langdetect" \
    "async-lru>=2.0.5"


conda install -y pytorch==2.6.0 torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

pip install accelerate==1.4.0
pip install transformers==4.52.3
pip install deepspeed==0.16.8
pip install "bitsandbytes>=0.43.0"
pip install "peft>=0.14.0"
pip install "trl[vllm]==0.18.0"
pip install "distilabel[vllm,ray,openai]>=1.5.2"
pip install "hf_transfer>=0.1.4"
pip install "huggingface-hub[cli,hf_xet]>=0.30.2,<1.0"
pip install "latex2sympy2_extended>=1.0.6"
pip install "liger-kernel>=0.5.10"
pip install "math-verify==0.5.2"
pip install "ruff>=0.9.0"
pip install "git+https://github.com/huggingface/lighteval.git@d3da6b9bbf38104c8b5e1acc86f83541f9a502d1"