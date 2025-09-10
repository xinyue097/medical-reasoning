# ==== 环境准备 ====
source ~/.bashrc         # conda
module load CUDA/12.4.1 
conda activate med


# cd ~/scratch/medmcq/LLaMA-Factory   # 
cd ~/scratch/medmcq/

# export VLLM_WORKER_MULTIPROC_METHOD=spawn



# llamafactory-cli train llama3_lora_sft.yaml
# llamafactory-cli train qwen3_14B_lora_sft.yaml 
# llamafactory-cli train qwen3_8B_lora_sft.yaml 

# llamafactory-cli export merge.yaml
#  llamafactory-cli train qwen2_5vl_lora_sft.yaml 
# llamafactory-cli train medgemma_4b.yaml
# llamafactory-cli train medgemma_4b_finding.yaml
# python test.py
# python eval.py
# python data.py data/medmcqa/processed/train.jsonl data/medmcqa/processed/train_alpaca.json
# ./eval.sh

# python medreason_eval.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datasets medmcqa,medbullets_op5 --batch_size 256 --max_new_tokens 1024
python medreason_eval.py --model_name LLaMA-Factory/saves/llama3-8b/lora/sft/model --datasets medmcqa,medbullets_op5,medqa,medxpert --batch_size 512 --max_new_tokens 1024 --temperature 0.1 
# python medreason_eval.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datasets medmcqa,medbullets_op5,medqa,medxpert --batch_size 512 --max_new_tokens 1024 --temperature 0.1 