# Licensed under the Apache License, Version 2.0 (the "License");

import logging
import os
import sys

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

# Add the src directory to Python path to import modules
current_file = os.path.abspath(__file__) 
medreason_dir = os.path.dirname(current_file) 
src_dir = os.path.dirname(medreason_dir)
project_root = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from medreason.configs import ScriptArguments, SFTConfig
from medreason.utils import get_dataset, get_model, get_tokenizer
from medreason.utils.wandb_logging import init_wandb_training
from trl import ModelConfig, SFTTrainer, TrlParser, get_peft_config, setup_chat_format

logger = logging.getLogger(__name__)


def main():
    config_file = sys.argv[1]
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_yaml_file(config_file)
    
    set_seed(training_args.seed)

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

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ######################################
    # Load dataset, tokenizer, and model #
    ######################################
    dataset = get_dataset(script_args)
    tokenizer = get_tokenizer(model_args, training_args)

    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    if tokenizer.chat_template is None:
        logger.info("No chat template provided, defaulting to ChatML.")
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

    def make_conversation(example, prompt_column: str = getattr(script_args, 'dataset_prompt_column', 'question')):
        """
        Convert medical dataset examples to conversation format.
        """
        # Check if this is a multiple choice question format (for medmcqa dataset)
        if all(key in example for key in ["question", "opa", "opb", "opc", "opd", "cop"]):
            cop_value = example.get("cop", -1)
            if cop_value not in [0, 1, 2, 3]:
                return None
                
            question_text = example["question"]
            option_a = example["opa"]
            option_b = example["opb"] 
            option_c = example["opc"]
            option_d = example["opd"]
            
            # Format question same as evaluation
            formatted_prompt = f"{question_text}\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}"
            
            correct_answer = ["A", "B", "C", "D"][cop_value]
            correct_option_text = [option_a, option_b, option_c, option_d][cop_value]
            
            # Simple direct answer for SFT training
            target_response = correct_answer
            
            messages = [
                {"role": "user", "content": formatted_prompt},
                {"role": "assistant", "content": target_response}
            ]
            
            return {"messages": messages, "cop": int(cop_value), "correct_option": correct_option_text}
        
        return None

    # Filter function to remove None entries
    def filter_valid_examples(example):
        return example is not None

    # Process datasets with medical conversation format
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
        )
        if training_args.eval_strategy != "no"
        else None
    )

    if eval_dataset:
        logger.info(f"Eval dataset size after filtering: {len(eval_dataset)}")

    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args) if model_args.use_peft else None,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
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
        "tags": ["medreason"],
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
        metrics = trainer.evaluate()
        if eval_dataset:
            metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    main()
