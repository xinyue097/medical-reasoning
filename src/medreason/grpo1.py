# Licensed under the Apache License, Version 2.0 (the "License");

import logging
import os
import sys
# import json
# import shutil

import datasets
import transformers
import torch
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

# Add the src directory to Python path to import modules
current_file = os.path.abspath(__file__) 
medreason_dir = os.path.dirname(current_file) 
src_dir = os.path.dirname(medreason_dir)
project_root = os.path.dirname(src_dir)

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from medreason.configs import GRPOConfig, GRPOScriptArguments
from medreason.rewards import get_reward_funcs
from medreason.utils import get_dataset, get_model, get_tokenizer
from medreason.utils.wandb_logging import init_wandb_training
from medreason.utils.optimization import (
    setup_torch_optimizations, 
    setup_deepspeed_optimizations, 
    configure_model_attention,
    SDPA_AVAILABLE,
    SDPBackend,
    sdpa_kernel
)
from medreason.utils.vllm import setup_vllm_engine, VLLM_AVAILABLE
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config

logger = logging.getLogger(__name__)


def main():
    config_file = sys.argv[1]
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_yaml_file(config_file)
    
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Setup all optimizations
    setup_torch_optimizations()
    setup_deepspeed_optimizations()

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log optimization status
    logger.info("🔥 A100 OPTIMIZATION STATUS:")
    logger.info(f"  SDPA: {'✅' if SDPA_AVAILABLE else '❌'}")
    logger.info(f"  Math SDP: {'✅' if torch.backends.cuda.math_sdp_enabled() else '❌'}")
    logger.info(f"  Flash SDP: {'✅' if torch.backends.cuda.flash_sdp_enabled() else '❌'}")
    logger.info(f"  Memory Efficient SDP: {'✅' if torch.backends.cuda.mem_efficient_sdp_enabled() else '❌'}")
    logger.info(f"  vLLM: {'✅' if VLLM_AVAILABLE else '❌'}")
    logger.info(f"  TF32: {'✅' if torch.backends.cuda.matmul.allow_tf32 else '❌'}")
    logger.info(f"  GPU Count: {torch.cuda.device_count()}")

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Detect any last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    dataset = get_dataset(script_args)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    ##############
    # Load model #
    ##############
    logger.info("*** Loading model with SDPA optimizations ***")
    model = get_model(model_args, training_args)
    model = configure_model_attention(model, training_args)

     # Apply LoRA
    if model_args.use_peft:
        from peft import get_peft_model
        
        peft_config = get_peft_config(model_args)
        model = get_peft_model(model, peft_config)
    else:
        logger.warning("⚠️ PEFT disabled in config")
    
    vllm_engine = setup_vllm_engine(model_args.model_name_or_path, training_args)
    reward_funcs = get_reward_funcs(script_args)

    def make_conversation(example, prompt_column: str = script_args.dataset_prompt_column):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})
            if not hasattr(make_conversation, '_logged_prompt'):
                print(f"DEBUG: Using system prompt: {training_args.system_prompt[:100]}...")
                make_conversation._logged_prompt = True
        else:
            print("WARNING: No system prompt found!")

        # Check if this is a multiple choice question format
        if all(key in example for key in ["question", "opa", "opb", "opc", "opd", "cop"]):
            cop_value = example.get("cop", -1)
            question_text = example["question"]
            option_a = example["opa"]
            option_b = example["opb"] 
            option_c = example["opc"]
            option_d = example["opd"]
            # Create formatted prompt with question and options
            formatted_prompt = f"{question_text}\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}"

            prompt.append({"role": "user", "content": formatted_prompt})
        else:
            print(f"Unexpected data format - missing required keys")
            print(f"Available keys: {list(example.keys())}")
            return None
        
        return {"prompt": prompt, "cop": int(cop_value)}

    # Filter function to remove None entries
    def filter_valid_examples(example):
        if example is None:
            return False
        if "cop" not in example:
            return False
        if example["cop"] not in [0, 1, 2, 3]:
            return False
        return True
    
    #############################
    # Initialize the GRPO trainer
    #############################
    # Update your dataset mapping with better error handling:
    logger.info("Processing training dataset...")
    train_dataset = dataset[script_args.dataset_train_split].map(
        make_conversation,
        remove_columns=dataset[script_args.dataset_train_split].column_names,
        desc="Processing training examples"
    ).filter(
        filter_valid_examples,
        desc="Filtering valid examples"
    )

    logger.info(f"Training dataset size after filtering: {len(train_dataset)}")

    logger.info("Processing evaluation dataset...")
    eval_dataset = (
        dataset[script_args.dataset_test_split].map(
            make_conversation,
            remove_columns=dataset[script_args.dataset_test_split].column_names,
            desc="Processing eval examples"
        ).filter(
            filter_valid_examples,
            desc="Filtering valid eval examples"
        ).rename_column("cop", "labels")
        if training_args.do_eval
        else None
    )

    if eval_dataset:
        logger.info(f"Eval dataset size after filtering: {len(eval_dataset)}")

    # Limit eval dataset for faster evaluation
    if eval_dataset and len(eval_dataset) > 1000:
        logger.info(f"Limiting eval dataset from {len(eval_dataset)} to 1000 examples for speed")
        eval_dataset = eval_dataset.select(range(1000))
    
    # Enhanced trainer with SDPA and vLLM support
    trainer_kwargs = {
        "model": model,
        "reward_funcs": reward_funcs,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        # "peft_config": get_peft_config(model_args),
        "peft_config": None,
        "processing_class": tokenizer,
    }
    
    # Add vLLM engine if available
    if vllm_engine is not None:
        trainer_kwargs["vllm_engine"] = vllm_engine
    
    trainer = GRPOTrainer(**trainer_kwargs)

    # Memory diagnostics
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            allocated = torch.cuda.memory_allocated(i) / 1e9
            logger.info(f"GPU {i}: {gpu_mem:.1f}GB total, {allocated:.1f}GB allocated")

    ###############
    # Training loop
    ###############
    logger.info("*** Start Training ***")
    
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    
    # Use SDPA context for training
    if SDPA_AVAILABLE:
        with sdpa_kernel([SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
    else:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["medreason", "a100-optimized", "deepspeed", "sdpa", "med-env"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        # Use SDPA context
        if SDPA_AVAILABLE:
            with sdpa_kernel([SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                metrics = trainer.evaluate()
        else:
            metrics = trainer.evaluate()
        
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Cleanup
    if vllm_engine is not None:
        del vllm_engine
    torch.cuda.empty_cache()
    
    logger.info("🎉 Training completed!")


if __name__ == "__main__":
    main()