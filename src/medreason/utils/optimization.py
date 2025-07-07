"""
Optimization utilities for PyTorch, DeepSpeed, and SDPA.
"""
import logging
import os
import torch

logger = logging.getLogger(__name__)

# SDPA availability check
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    SDPA_AVAILABLE = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    logger.info(f"SDPA available: {SDPA_AVAILABLE}")
except ImportError:
    SDPA_AVAILABLE = False
    logger.warning("SDPA not available, using standard attention")


def setup_sdpa_optimization():
    """Configure SDPA"""
    if SDPA_AVAILABLE and torch.cuda.is_available():
        torch.backends.cuda.enable_math_sdp(True) 
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        # Set SDPA context
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        logger.info(f"  Math SDP: {torch.backends.cuda.math_sdp_enabled()}")
        logger.info(f"  Flash SDP: {torch.backends.cuda.flash_sdp_enabled()}")
        logger.info(f"  Memory Efficient SDP: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    else:
        logger.warning("SDPA not available, using standard attention")


def setup_deepspeed_optimizations():
    """Configure DeepSpeed optimizations for A100"""
    os.environ["DEEPSPEED_NVCC_FLAGS"] = "-gencode arch=compute_80,code=sm_80" 
    os.environ["DEEPSPEED_CUDA_ARCH"] = "8.0"
    # Memory optimizations
    os.environ["DEEPSPEED_CPU_OFFLOAD"] = "false"
    os.environ["DEEPSPEED_ALLREDUCE_POST_ACCUMULATION"] = "true"
    logger.info("⚡ DeepSpeed optimizations")


def setup_torch_optimizations():
    """Configure PyTorch optimizations for A100 with SDPA"""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    setup_sdpa_optimization()
    # Memory optimizations
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True 
    # Enable JIT compilation for speed
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    # Set memory allocation strategy
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
    logger.info("🔧 PyTorch optimizations enabled for A100 with SDPA")


def configure_model_attention(model, training_args):
    """Configure model to use SDPA attention implementation"""
    if hasattr(model.config, 'attn_implementation'):
        if SDPA_AVAILABLE:
            model.config.attn_implementation = "sdpa"
            logger.info("Model configured to use SDPA attention")
        else:
            model.config.attn_implementation = "eager"
            logger.warning("SDPA not available, using eager attention")
    # Enable gradient checkpointing if specified
    if getattr(training_args, 'gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    return model