"""
vLLM utilities for accelerated generation.
"""
import logging
import torch

logger = logging.getLogger(__name__)

# vLLM imports
try:
    from vllm import LLM
    # from vllm import SamplingParams
    VLLM_AVAILABLE = True
    logger.info("vLLM available for accelerated generation")
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available, falling back to HuggingFace generation")


def setup_vllm_engine(model_path: str, training_args):
    """Initialize vLLM engine for fast generation with SDPA"""
    if not VLLM_AVAILABLE or not getattr(training_args, 'use_vllm', False):
        return None
    try:
        vllm_engine = LLM(
            model=model_path,
            tensor_parallel_size=torch.cuda.device_count(),  # Use all available GPUs
            gpu_memory_utilization=getattr(training_args, 'vllm_gpu_memory_utilization', 0.8),
            max_model_len=getattr(training_args, 'max_completion_length', 1024),
            dtype="bfloat16",
            trust_remote_code=True,
            enforce_eager=False,
            disable_log_stats=True,
            use_v2_block_manager=True,
        )
        logger.info("vLLM engine initialized successfully with SDPA")
        return vllm_engine
    except Exception as e:
        logger.warning(f"Failed to initialize vLLM: {e}")
        return None