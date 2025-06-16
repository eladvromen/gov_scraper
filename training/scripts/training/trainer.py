"""
Main trainer module for LLaMA continued pre-training on legal text.
Supports temporal analysis and efficient multi-GPU training.
"""

import os
import sys
from pathlib import Path
import logging
import math
import random
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass

import torch
import wandb
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import get_scheduler, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, set_seed
from torch.optim import AdamW
from datasets import load_dataset
import yaml

# Add the scripts directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_utils import DataConfig, load_and_process_datasets
from utils.model_utils import (
    ModelConfig,
    get_model_and_tokenizer,
    setup_distributed_training,
    create_deepspeed_config,
    init_deepspeed,
)

logger = logging.getLogger(__name__)

def set_all_seeds(seed: int):
    """
    Set all random seeds for complete reproducibility.
    Critical for exact reproduction across pre/post Brexit training.
    """
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Transformers library seed
    set_seed(seed)
    
    # Ensure deterministic behavior (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for complete reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Set all seeds to {seed} for reproducible training")

@dataclass
class TrainingConfig:
    """Configuration for training process."""
    output_dir: str
    run_name: str
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    
    # Optimization
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Checkpointing
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Evaluation
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    max_eval_samples: Optional[int] = 1000
    
    # Logging
    logging_steps: int = 50
    log_level: str = "info"
    use_wandb: bool = False  # New flag to control wandb usage
    
    # Distributed training
    local_rank: int = -1
    deepspeed_stage: int = 3
    use_deepspeed: bool = False

def create_dataloaders(
    data_config: DataConfig,
    tokenizer: PreTrainedTokenizer,
    training_config: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    """Create training and evaluation dataloaders."""
    
    train_dataset, eval_dataset = load_and_process_datasets(
        data_config,
        tokenizer,
    )
    
    # Create dataloaders with deterministic behavior
    def worker_init_fn(worker_id):
        """Initialize worker with deterministic seed"""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.per_device_train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,  # Ensure deterministic workers
        generator=torch.Generator().manual_seed(data_config.seed),  # Deterministic shuffling
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_config.per_device_eval_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,  # Ensure deterministic workers
    )
    
    return train_dataloader, eval_dataloader

def train(
    model_config: ModelConfig,
    data_config: DataConfig,
    training_config: TrainingConfig,
):
    """Main training function."""
    
    # CRITICAL: Set all seeds for reproducibility FIRST
    set_all_seeds(data_config.seed)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=training_config.log_level.upper(),
    )
    
    # Initialize distributed training if needed
    setup_distributed_training(training_config.local_rank)
    
    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(
        model_config,
        training_config.local_rank,
    )
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing for memory efficiency")
    
    # Create dataloaders
    train_dataloader, eval_dataloader = create_dataloaders(
        data_config,
        tokenizer,
        training_config,
    )
    
    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / training_config.gradient_accumulation_steps
    )
    max_train_steps = training_config.num_train_epochs * num_update_steps_per_epoch
    
    # Initialize optimizer and scheduler - using memory-efficient optimizer
    try:
        # Try to use 8-bit AdamW for memory efficiency
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        logger.info("Using 8-bit AdamW optimizer for memory efficiency")
    except ImportError:
        # Fallback to regular AdamW with reduced precision
        optimizer = AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            eps=1e-6,  # Slightly larger epsilon for stability
        )
        logger.info("Using regular AdamW optimizer")

    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=int(max_train_steps * training_config.warmup_ratio),
        num_training_steps=max_train_steps,
    )
    
    # Initialize DeepSpeed if enabled
    if training_config.use_deepspeed:
        try:
            ds_config = create_deepspeed_config(
                train_batch_size=training_config.per_device_train_batch_size * training_config.gradient_accumulation_steps,
                gradient_accumulation_steps=training_config.gradient_accumulation_steps,
                learning_rate=training_config.learning_rate,
                warmup_ratio=training_config.warmup_ratio,
                zero_stage=training_config.deepspeed_stage,
            )
            model = init_deepspeed(model, ds_config, training_config.local_rank)
            logger.info("Using DeepSpeed for training")
        except ImportError:
            logger.warning("DeepSpeed not available, falling back to standard training")
            training_config.use_deepspeed = False
    
    # Initialize wandb if enabled
    if training_config.use_wandb and training_config.local_rank in [-1, 0]:
        try:
            wandb.init(
                project="llama-legal-pretraining",
                name=training_config.run_name,
                config={
                    "model_config": vars(model_config),
                    "data_config": vars(data_config),
                    "training_config": vars(training_config),
                }
            )
            logger.info("Using Weights & Biases for logging")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            training_config.use_wandb = False
    
    # Training loop
    logger.info("***** Starting training *****")
    progress_bar = tqdm(
        range(max_train_steps),
        disable=training_config.local_rank not in [-1, 0],
    )
    
    for epoch in range(training_config.num_train_epochs):
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )
            
            loss = outputs.loss / training_config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights
            if (step + 1) % training_config.gradient_accumulation_steps == 0:
                if training_config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                progress_bar.update(1)
                
                # Log metrics
                if training_config.local_rank in [-1, 0] and step % training_config.logging_steps == 0:
                    metrics = {
                        "train/loss": loss.item() * training_config.gradient_accumulation_steps,
                        "train/epoch": epoch,
                        "train/step": step,
                        "train/lr": scheduler.get_last_lr()[0],
                    }
                    if training_config.use_wandb:
                        wandb.log(metrics)
                    logger.info(f"Step {step}: {metrics}")
            
            # Evaluation
            if (step + 1) % (training_config.eval_steps * training_config.gradient_accumulation_steps) == 0:
                model.eval()
                eval_loss = 0
                eval_steps = 0
                
                for eval_batch in eval_dataloader:
                    eval_batch = {k: v.to(model.device) for k, v in eval_batch.items()}
                    with torch.no_grad():
                        outputs = model(**eval_batch)
                        eval_loss += outputs.loss.item()
                        eval_steps += 1
                
                eval_loss = eval_loss / eval_steps
                
                if training_config.local_rank in [-1, 0]:
                    metrics = {
                        "eval/loss": eval_loss,
                        "eval/epoch": epoch,
                        "eval/step": step,
                    }
                    if training_config.use_wandb:
                        wandb.log(metrics)
                    logger.info(f"Evaluation: {metrics}")
                
                model.train()
            
            # Save checkpoint
            if (step + 1) % (training_config.save_steps * training_config.gradient_accumulation_steps) == 0:
                if training_config.local_rank in [-1, 0]:
                    checkpoint_dir = os.path.join(
                        training_config.output_dir,
                        f"checkpoint-{step + 1}",
                    )
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
    
    # Save final model
    if training_config.local_rank in [-1, 0]:
        model.save_pretrained(training_config.output_dir)
        tokenizer.save_pretrained(training_config.output_dir)
        if training_config.use_wandb:
            wandb.finish()
    
if __name__ == "__main__":
    # This will be replaced by a proper CLI using click
    # For now, this is just a placeholder
    pass 