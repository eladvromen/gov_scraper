#!/usr/bin/env python3
"""
Main training script for LLaMA continued pre-training on legal text.
Supports training on any temporal subset of the data.
"""

import os
import click
import yaml
import logging
from typing import Optional, List

from .trainer import train, ModelConfig, DataConfig, TrainingConfig

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.option('--dataset-patterns', '-d', multiple=True, help='Dataset file patterns to train on (e.g. "*_2013_2016_*.jsonl")')
@click.option('--output-dir', '-o', help='Override output directory from config')
@click.option('--run-name', '-n', help='Override run name from config')
@click.option('--local_rank', type=int, default=-1, help='Local rank for distributed training')
def main(
    config_path: str,
    dataset_patterns: Optional[List[str]],
    output_dir: Optional[str],
    run_name: Optional[str],
    local_rank: int,
):
    """Train LLaMA model on legal text data.
    
    CONFIG_PATH: Path to YAML config file
    """
    
    # Load base configuration
    config = load_config(config_path)
    
    # Create model config
    model_config = ModelConfig(
        model_name_or_path=config['model']['name'],
        tokenizer_name=config['model'].get('tokenizer_name'),
        cache_dir=config['model'].get('cache_dir'),
        use_flash_attention=config['model'].get('use_flash_attention', True),
        torch_dtype=config['model'].get('torch_dtype', 'bfloat16'),
        device_map=config['model'].get('device_map', 'auto'),
        low_cpu_mem_usage=config['model'].get('low_cpu_mem_usage', True),
    )
    
    # Create data config
    data_config = DataConfig(
        data_dir=config['data']['data_dir'],
        dataset_patterns=dataset_patterns or config['data']['dataset_patterns'],
        train_split=config['data'].get('train_split', 0.95),
        eval_split=config['data'].get('eval_split', 0.05),
        seed=config['data'].get('seed', 42),
        text_column=config['data'].get('text_column', 'text'),
        add_eos_token=config['data'].get('add_eos_token', True),
        chunk_size=config['data'].get('chunk_size', 2048),
        max_eval_samples=config['data'].get('max_eval_samples', None),
    )
    
    # Create training config
    training_config = TrainingConfig(
        output_dir=output_dir or config['training']['output_dir'],
        run_name=run_name or config['training']['run_name'],
        learning_rate=config['training'].get('learning_rate', 2e-5),
        num_train_epochs=config['training'].get('num_train_epochs', 3),
        per_device_train_batch_size=config['training'].get('per_device_train_batch_size', 4),
        per_device_eval_batch_size=config['training'].get('per_device_eval_batch_size', 8),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 4),
        warmup_ratio=config['training'].get('warmup_ratio', 0.03),
        weight_decay=config['training'].get('weight_decay', 0.01),
        max_grad_norm=config['training'].get('max_grad_norm', 1.0),
        save_strategy=config['training'].get('save_strategy', 'steps'),
        save_steps=config['training'].get('save_steps', 500),
        save_total_limit=config['training'].get('save_total_limit', 3),
        evaluation_strategy=config['training'].get('evaluation_strategy', 'steps'),
        eval_steps=config['training'].get('eval_steps', 500),
        max_eval_samples=config['training'].get('max_eval_samples', 1000),
        logging_steps=config['training'].get('logging_steps', 50),
        log_level=config['training'].get('log_level', 'info'),
        local_rank=local_rank,
        deepspeed_stage=config['training'].get('deepspeed_stage', 3),
    )
    
    # Start training
    train(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
    )

if __name__ == "__main__":
    main() 