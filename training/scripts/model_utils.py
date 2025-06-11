"""
Model utilities for efficient LLaMA training with DeepSpeed and flash attention.
Optimized for A100 GPUs and large language models.
"""

import os
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

import torch
import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# Optional imports for advanced features
try:
    from transformers.deepspeed import HfDeepSpeedConfig
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    HfDeepSpeedConfig = None
    deepspeed = None
    DEEPSPEED_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model loading and training."""
    model_name_or_path: str
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = None
    use_flash_attention: bool = True
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    low_cpu_mem_usage: bool = True

def setup_distributed_training(local_rank: int) -> None:
    """Initialize distributed training environment."""
    if local_rank != -1:
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

def get_model_and_tokenizer(
    config: ModelConfig,
    local_rank: int = -1,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer with optimized settings for A100s."""
    
    # Convert torch_dtype from string to torch type
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)
    
    # Load tokenizer first (needed for embedding resizing)
    tokenizer_name = config.tokenizer_name or config.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=config.cache_dir,
        padding_side="right",
        use_fast=True,
    )
    
    # Ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load model with optimized settings
    model_kwargs = {
        "cache_dir": config.cache_dir,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": config.low_cpu_mem_usage,
    }
    
    # Add flash attention if requested and available
    if config.use_flash_attention:
        try:
            from transformers.models.llama.modeling_llama import LlamaAttention
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2.0")
        except ImportError:
            logger.warning("Flash Attention not available, falling back to standard attention")
    
    # Load model
    if local_rank == -1:
        # Single GPU or CPU training
        model_kwargs["device_map"] = config.device_map
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            **model_kwargs
        )
    else:
        # Distributed training - let DeepSpeed handle device placement
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            **model_kwargs
        )
    
    # Resize embeddings if needed
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        
    return model, tokenizer

def create_deepspeed_config(
    train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_ratio: float = 0.03,
    zero_stage: int = 3,
) -> Dict[str, Any]:
    """Create DeepSpeed ZeRO config optimized for A100 GPUs."""
    
    return {
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
            }
        },
        
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": learning_rate,
                "warmup_num_steps": "auto",
                "warmup_type": "linear",
                "warmup_ratio": warmup_ratio,
            }
        },
        
        "fp16": {
            "enabled": False,
        },
        
        "bf16": {
            "enabled": True,
        },
        
        "zero_optimization": {
            "stage": zero_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e7,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5,
        },
        
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        
        "wall_clock_breakdown": False,
        "steps_per_print": 100,
    }

def init_deepspeed(
    model: PreTrainedModel,
    ds_config: Dict[str, Any],
    local_rank: int,
) -> Any:
    """Initialize DeepSpeed engine."""
    
    if not DEEPSPEED_AVAILABLE:
        raise ImportError("DeepSpeed is not available. Please install with: pip install deepspeed")
    
    # Set HF DeepSpeed config
    _ = HfDeepSpeedConfig(ds_config)  # Keep reference to avoid GC
    
    # Initialize DeepSpeed
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters(),
    )
    
    return model_engine 