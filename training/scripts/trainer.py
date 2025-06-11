"""
Main trainer module for LLaMA continued pre-training on legal text.
Supports temporal analysis and efficient multi-GPU training.
"""

import os
import logging
import math
from typing import Optional, Dict, Any
from dataclasses import dataclass

import torch
import wandb
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import get_scheduler, PreTrainedTokenizer

from .data_utils import DataConfig, load_and_process_datasets
from .model_utils import (
    ModelConfig,
    get_model_and_tokenizer,
    setup_distributed_training,
    create_deepspeed_config,
    init_deepspeed,
)

logger = logging.getLogger(__name__)

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
    
    # Distributed training
    local_rank: int = -1
    deepspeed_stage: int = 3

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
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.per_device_train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_config.per_device_eval_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    return train_dataloader, eval_dataloader

def train(
    model_config: ModelConfig,
    data_config: DataConfig,
    training_config: TrainingConfig,
):
    """Main training function."""
    
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
    
    # Create DeepSpeed config and initialize
    ds_config = create_deepspeed_config(
        train_batch_size=training_config.per_device_train_batch_size * training_config.gradient_accumulation_steps,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        warmup_ratio=training_config.warmup_ratio,
        zero_stage=training_config.deepspeed_stage,
    )
    
    model_engine = init_deepspeed(
        model,
        ds_config,
        training_config.local_rank,
    )
    
    # Initialize wandb if main process
    if training_config.local_rank in [-1, 0]:
        wandb.init(
            project="llama-legal-pretraining",
            name=training_config.run_name,
            config={
                "model_config": vars(model_config),
                "data_config": vars(data_config),
                "training_config": vars(training_config),
            }
        )
    
    # Training loop
    logger.info("***** Starting training *****")
    progress_bar = tqdm(
        range(max_train_steps),
        disable=training_config.local_rank not in [-1, 0],
    )
    
    for epoch in range(training_config.num_train_epochs):
        model_engine.train()
        
        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model_engine(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )
            
            loss = outputs.loss
            
            # Backward pass
            model_engine.backward(loss)
            
            # Update weights
            if (step + 1) % training_config.gradient_accumulation_steps == 0:
                model_engine.step()
                progress_bar.update(1)
                
                # Log metrics
                if training_config.local_rank in [-1, 0] and step % training_config.logging_steps == 0:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/epoch": epoch,
                        "train/step": step,
                    })
            
            # Evaluation
            if (step + 1) % (training_config.eval_steps * training_config.gradient_accumulation_steps) == 0:
                model_engine.eval()
                eval_loss = 0
                eval_steps = 0
                
                for eval_batch in eval_dataloader:
                    eval_batch = {k: v.to(model_engine.device) for k, v in eval_batch.items()}
                    
                    with torch.no_grad():
                        outputs = model_engine(
                            input_ids=eval_batch["input_ids"],
                            attention_mask=eval_batch["attention_mask"],
                            labels=eval_batch["labels"],
                        )
                        
                    eval_loss += outputs.loss.item()
                    eval_steps += 1
                
                eval_loss = eval_loss / eval_steps
                perplexity = math.exp(eval_loss)
                
                if training_config.local_rank in [-1, 0]:
                    wandb.log({
                        "eval/loss": eval_loss,
                        "eval/perplexity": perplexity,
                        "eval/step": step,
                    })
                    
                    # Save checkpoint
                    if training_config.local_rank == -1:
                        model_engine.save_checkpoint(
                            os.path.join(training_config.output_dir, f"checkpoint-{step}")
                        )
                
                model_engine.train()
    
    # Save final model
    if training_config.local_rank in [-1, 0]:
        model_engine.save_checkpoint(
            os.path.join(training_config.output_dir, "final_checkpoint")
        )
        
    wandb.finish()
    
if __name__ == "__main__":
    # This will be replaced by a proper CLI using click
    # For now, this is just a placeholder
    pass 