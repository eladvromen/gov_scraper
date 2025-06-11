"""
Data loading and processing utilities for LLaMA continued pre-training.
Handles any temporal chunk-based legal text dataset in our format.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
import numpy as np
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for dataset loading and processing."""
    data_dir: str
    dataset_patterns: Union[str, List[str]]  # File patterns or list of patterns
    train_split: float = 0.95
    eval_split: float = 0.05
    seed: int = 42
    text_column: str = "text"
    add_eos_token: bool = True
    chunk_size: int = 2048
    max_eval_samples: Optional[int] = None

class LegalChunkDataset(Dataset):
    """Dataset for legal text chunks supporting both JSONL and txt formats."""
    
    def __init__(
        self,
        file_paths: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DataConfig,
        split: str = "train"
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        
        # Load dataset using HuggingFace datasets
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            
        # Support both JSONL and txt files
        self.raw_datasets = []
        for path in file_paths:
            if path.endswith('.jsonl'):
                dataset = load_dataset('json', data_files=path)['train']
            else:  # txt file
                with open(path, 'r') as f:
                    texts = f.read().split('\n\n')  # Split on double newline
                dataset = load_dataset('dict', 
                                    data={'text': texts, 
                                         'case_id': [f'doc_{i}' for i in range(len(texts))]})['train']
            self.raw_datasets.append(dataset)
        
        # Combine datasets
        if len(self.raw_datasets) > 1:
            self.dataset = concatenate_datasets(self.raw_datasets)
        else:
            self.dataset = self.raw_datasets[0]
            
        # Split dataset
        if split != "all":
            split_dataset = self.dataset.train_test_split(
                train_size=config.train_split,
                seed=config.seed
            )
            self.dataset = split_dataset["train" if split == "train" else "test"]
            
        # Subsample eval set if needed
        if split == "eval" and config.max_eval_samples:
            self.dataset = self.dataset.select(range(min(len(self.dataset), config.max_eval_samples)))
            
        logger.info(f"Loaded {len(self.dataset)} examples for {split} split")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a tokenized and formatted example."""
        example = self.dataset[idx]
        text = example[self.config.text_column]
        
        # Tokenize
        tokenized = self.tokenizer(
            text,
            max_length=self.config.chunk_size,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Add EOS token if configured
        if self.config.add_eos_token:
            input_ids = tokenized.input_ids.squeeze()
            attention_mask = tokenized.attention_mask.squeeze()
            
            # Find the first padding token
            pad_idx = (input_ids == self.tokenizer.pad_token_id).nonzero()
            if len(pad_idx) > 0:
                # Insert EOS before padding
                eos_idx = pad_idx[0].item()
                input_ids[eos_idx-1] = self.tokenizer.eos_token_id
                
        return {
            "input_ids": tokenized.input_ids.squeeze(),
            "attention_mask": tokenized.attention_mask.squeeze(),
            "labels": tokenized.input_ids.squeeze().clone(),  # For causal LM
        }

def load_and_process_datasets(
    config: DataConfig,
    tokenizer: PreTrainedTokenizer,
) -> Tuple[Dataset, Dataset]:
    """Load and process datasets for training and evaluation."""
    
    # Resolve file patterns
    if isinstance(config.dataset_patterns, str):
        config.dataset_patterns = [config.dataset_patterns]
        
    all_files = []
    for pattern in config.dataset_patterns:
        pattern_path = os.path.join(config.data_dir, pattern)
        if os.path.isfile(pattern_path):
            all_files.append(pattern_path)
        else:
            # Handle glob patterns
            import glob
            all_files.extend(glob.glob(pattern_path))
    
    if not all_files:
        raise ValueError(f"No files found matching patterns: {config.dataset_patterns}")
        
    logger.info(f"Found {len(all_files)} files to process")
    
    # Create train/eval datasets
    train_dataset = LegalChunkDataset(all_files, tokenizer, config, split="train")
    eval_dataset = LegalChunkDataset(all_files, tokenizer, config, split="eval")
    
    return train_dataset, eval_dataset 