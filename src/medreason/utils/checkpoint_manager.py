"""
Checkpoint management utilities for training.
"""
import logging
import os
from datetime import datetime
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)


def validate_training_config(training_args, script_args):
    """Validate training configuration and checkpoint settings"""
    logger.info("🔍 Validating training configuration...")
    
    # Check for conflicting settings
    issues = []
    warnings = []
    
    # Checkpoint validation
    if training_args.resume_from_checkpoint and training_args.overwrite_output_dir:
        issues.append("Cannot use both --resume_from_checkpoint and --overwrite_output_dir")
    
    # Memory and batch size validation
    if training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps > 32:
        warnings.append(f"Large effective batch size detected: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    
    # Save strategy validation
    if training_args.save_steps and training_args.max_steps:
        if training_args.save_steps > training_args.max_steps:
            warnings.append(f"save_steps ({training_args.save_steps}) > max_steps ({training_args.max_steps}) - no intermediate saves will occur")
    
    # Eval strategy validation  
    if training_args.eval_steps and training_args.max_steps:
        if training_args.eval_steps > training_args.max_steps:
            warnings.append(f"eval_steps ({training_args.eval_steps}) > max_steps ({training_args.max_steps}) - no evaluations will occur")
    
    # Log results
    if issues:
        for issue in issues:
            logger.error(f"❌ CONFIG ERROR: {issue}")
        raise ValueError("Configuration validation failed. Please fix the issues above.")
    
    if warnings:
        for warning in warnings:
            logger.warning(f"⚠️  CONFIG WARNING: {warning}")
    
    logger.info("✅ Configuration validation passed")

    if hasattr(training_args, 'resume_from_checkpoint') and training_args.resume_from_checkpoint:
        checkpoint_path = training_args.resume_from_checkpoint
        adapter_config = os.path.join(checkpoint_path, 'adapter_config.json')
        
        if os.path.exists(adapter_config):
            logger.info("📁 PEFT checkpoint detected for resume")
            # Could add more PEFT-specific validation here
        else:
            logger.warning("⚠️  Resuming from non-PEFT checkpoint - may cause structure mismatch")


def detect_and_handle_checkpoints(training_args):
    """
    Detect and handle checkpoints with enhanced logic.
    
    Returns:
        tuple: (last_checkpoint, resume_checkpoint)
    """
    last_checkpoint = None
    resume_checkpoint = None
    
    # Check if output directory exists and handle accordingly
    if os.path.isdir(training_args.output_dir):
        logger.info(f"Output directory exists: {training_args.output_dir}")
        
        # Get list of existing files/folders
        existing_items = os.listdir(training_args.output_dir)
        checkpoint_dirs = [item for item in existing_items if item.startswith('checkpoint-')]
        
        if checkpoint_dirs:
            logger.info(f"Found {len(checkpoint_dirs)} checkpoint(s): {sorted(checkpoint_dirs)}")
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint:
                logger.info(f"Latest checkpoint detected: {last_checkpoint}")
        
        # Handle different scenarios
        if training_args.overwrite_output_dir:
            logger.warning("🗑️  Overwrite mode enabled - will clear output directory")
            if last_checkpoint and not training_args.resume_from_checkpoint:
                logger.warning(f"⚠️  Existing checkpoint will be overwritten: {last_checkpoint}")
        elif existing_items and not checkpoint_dirs and training_args.do_train:
            # Directory has files but no checkpoints
            raise ValueError(
                f"❌ Output directory ({training_args.output_dir}) contains files but no valid checkpoints. "
                f"Found: {existing_items[:5]}{'...' if len(existing_items) > 5 else ''}. "
                "Use --overwrite_output_dir to clear it, or choose a different output directory."
            )
        elif last_checkpoint and not training_args.resume_from_checkpoint and training_args.do_train:
            logger.info("📂 Valid checkpoint found - will resume training automatically")
    else:
        logger.info(f"Creating new output directory: {training_args.output_dir}")
        os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Determine final checkpoint to use
    if training_args.resume_from_checkpoint:
        resume_checkpoint = training_args.resume_from_checkpoint
        if os.path.isfile(resume_checkpoint):
            logger.info(f"📁 Resuming from specified checkpoint file: {resume_checkpoint}")
        elif os.path.isdir(resume_checkpoint):
            logger.info(f"📁 Resuming from specified checkpoint directory: {resume_checkpoint}")
        else:
            raise ValueError(f"❌ Specified checkpoint not found: {resume_checkpoint}")
    elif last_checkpoint:
        resume_checkpoint = last_checkpoint
        logger.info(f"📁 Resuming from auto-detected checkpoint: {resume_checkpoint}")
    else:
        logger.info("🆕 Starting training from scratch - no checkpoint to resume from")
    
    # Log checkpoint strategy
    if resume_checkpoint:
        logger.warning(f"🔄 CHECKPOINT RESUME: {resume_checkpoint}")
        # Validate checkpoint directory structure
        _validate_checkpoint_files(resume_checkpoint)
    else:
        logger.info("🆕 FRESH START: Training from scratch")
    
    return last_checkpoint, resume_checkpoint


# Replace the _validate_checkpoint_files function (lines 124-150)

def _validate_checkpoint_files(checkpoint_path):
    """Validate checkpoint directory structure with PEFT support"""
    checkpoint_files = ['config.json', 'training_args.bin']
    model_files = ['model.safetensors', 'pytorch_model.bin']
    peft_files = ['adapter_config.json', 'adapter_model.safetensors', 'adapter_model.bin']
    
    missing_files = []
    
    # Check required files
    for file in checkpoint_files:
        file_path = os.path.join(checkpoint_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    # Check for model files (either full model or PEFT adapter)
    model_file_found = False
    peft_file_found = False
    
    # Check for full model files
    for model_file in model_files:
        file_path = os.path.join(checkpoint_path, model_file)
        if os.path.exists(file_path):
            model_file_found = True
            break
    
    # Check for PEFT adapter files
    peft_config_exists = os.path.exists(os.path.join(checkpoint_path, 'adapter_config.json'))
    for peft_file in ['adapter_model.safetensors', 'adapter_model.bin']:
        file_path = os.path.join(checkpoint_path, peft_file)
        if os.path.exists(file_path):
            peft_file_found = True
            break
    
    # Determine checkpoint type
    if peft_config_exists and peft_file_found:
        logger.info("✅ PEFT checkpoint detected with adapter files")
    elif model_file_found:
        logger.info("✅ Full model checkpoint detected")
    else:
        missing_files.extend(['model files OR adapter files'])
    
    if missing_files:
        logger.warning(f"⚠️  Some checkpoint files missing: {missing_files}")
        logger.warning("Training will attempt to continue but may encounter issues")
    else:
        logger.info("✅ Checkpoint validation passed - all required files present")


# Add new function for PEFT-specific checkpoint handling
def handle_peft_checkpoint_mismatch(trainer, resume_checkpoint):
    """
    Handle PEFT checkpoint loading with better error recovery
    """
    try:
        # First, try normal checkpoint loading
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        return train_result, True
    except Exception as e:
        error_msg = str(e)
        
        if "base_model.model.model" in error_msg or "adapter" in error_msg.lower():
            logger.error(f"❌ PEFT checkpoint structure mismatch: {error_msg}")
            logger.warning("🔧 Attempting PEFT-specific recovery...")
            
            # Try to load only the adapter weights if possible
            try:
                import torch
                checkpoint_path = resume_checkpoint
                
                # Check if this is a PEFT checkpoint
                adapter_config_path = os.path.join(checkpoint_path, 'adapter_config.json')
                if os.path.exists(adapter_config_path):
                    logger.info("📁 PEFT adapter config found, attempting adapter-only loading")
                    # This is a more complex recovery that would need custom implementation
                    logger.warning("⚠️  PEFT adapter recovery not implemented, starting fresh")
                else:
                    logger.warning("⚠️  No adapter config found, checkpoint may be incompatible")
                
            except Exception as recovery_error:
                logger.error(f"❌ PEFT recovery failed: {recovery_error}")
        
        # If all recovery attempts fail, start fresh
        logger.warning("🆕 All checkpoint recovery attempts failed, starting fresh training")
        train_result = trainer.train()
        return train_result, False


def execute_training_with_checkpoint_handling(trainer, resume_checkpoint, sdpa_available, sdpa_kernel, sdp_backend):
    """
    Execute training with proper checkpoint handling and PEFT error recovery.
    """
    logger.info("*** Start Training ***")
    
    if resume_checkpoint:
        logger.info(f"🔄 Resuming training from: {resume_checkpoint}")
        
        # ✅ ENHANCED: Use PEFT-aware checkpoint handling
        train_result, success = handle_peft_checkpoint_mismatch(trainer, resume_checkpoint)
        
        if success:
            logger.info("✅ Successfully resumed training from checkpoint")
        else:
            logger.warning("⚠️  Started fresh training due to checkpoint issues")
        
        return train_result
    else:
        logger.info("🆕 Starting fresh training")
        # Use SDPA context for training
        if sdpa_available and sdpa_kernel and sdp_backend:
            with sdpa_kernel([sdp_backend.MATH, sdp_backend.FLASH_ATTENTION, sdp_backend.EFFICIENT_ATTENTION]):
                train_result = trainer.train()
        else:
            train_result = trainer.train()
        return train_result

def save_model_with_timestamp(trainer, training_args, tokenizer, train_result):
    """
    Save model with enhanced organization and completion markers.
    
    Args:
        trainer: The trainer instance
        training_args: Training arguments
        tokenizer: Tokenizer instance
        train_result: Training results
    """
    logger.info("*** Save model ***")
    
    # Create final checkpoint directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_dir = os.path.join(training_args.output_dir, f"final_model_{timestamp}")
    
    try:
        # Align the model's generation config with the tokenizer's eos token
        trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
        
        # Save the final model
        trainer.save_model(final_model_dir)
        logger.info(f"✅ Final model saved to {final_model_dir}")
        
        # Also save to the main output directory for convenience
        trainer.save_model(training_args.output_dir)
        logger.info(f"✅ Model also saved to {training_args.output_dir}")
        
        # Create a success marker file
        success_file = os.path.join(training_args.output_dir, "TRAINING_COMPLETED_SUCCESSFULLY.txt")
        with open(success_file, 'w') as f:
            f.write(f"Training completed successfully at {datetime.now()}\n")
            f.write(f"Total steps: {train_result.global_step}\n")
            f.write(f"Final loss: {train_result.training_loss:.4f}\n")
            f.write(f"Final model location: {final_model_dir}\n")
            f.write(f"Training args: {training_args.output_dir}/training_args.bin\n")
        
        logger.info(f"✅ Training completion marker created: {success_file}")
        
    except Exception as e:
        logger.error(f"❌ Error saving model: {e}")
        raise


def execute_evaluation_with_sdpa(trainer, training_args, dataset, script_args, sdpa_available, sdpa_kernel, sdp_backend):
    """
    Execute evaluation with SDPA context if available.
    
    Args:
        trainer: The trainer instance
        training_args: Training arguments
        dataset: Dataset for evaluation
        script_args: Script arguments
        sdpa_available: Whether SDPA is available
        sdpa_kernel: SDPA kernel function
        sdp_backend: SDP backend enum
    """
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        # Use SDPA context
        if sdpa_available and sdpa_kernel and sdp_backend:
            with sdpa_kernel([sdp_backend.MATH, sdp_backend.FLASH_ATTENTION, sdp_backend.EFFICIENT_ATTENTION]):
                metrics = trainer.evaluate()
        else:
            metrics = trainer.evaluate()
        
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
        logger.info("✅ Evaluation completed")
        return metrics
    return None