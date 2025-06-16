#!/usr/bin/env python3
"""
Proper evaluation script using the existing LegalEvaluator framework.
Usage: python evaluate_checkpoint.py --model-path /path/to/model
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

# Add the scripts directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation import LegalEvaluator, EvaluationConfig
from utils.data_utils import DataConfig, load_and_process_datasets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model_properly(
    model_path: str, 
    data_dir: str,
    dataset_pattern: str = "pre_brexit_2013_2016_chunks.jsonl",
    output_dir: str = None
) -> Dict[str, float]:
    """Evaluate model using the proper LegalEvaluator framework."""
    
    # Auto-generate output directory based on model name if not provided
    if output_dir is None:
        model_name = Path(model_path).name if os.path.exists(model_path) else model_path.split('/')[-1]
        output_dir = f"training/evaluations/{model_name}_evaluation"
        logger.info(f"Auto-generated output directory: {output_dir}")
    
    logger.info(f"Loading model from: {model_path}")
    
    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return {}
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create data config with CORRECT parameters from training
    data_config = DataConfig(
        data_dir=data_dir,
        dataset_patterns=[dataset_pattern],
        train_split=0.90,  # CORRECT: Same as training (90/10 split)
        eval_split=0.10,   # CORRECT: Same as training (90/10 split)
        seed=42,           # CRITICAL: Same seed ensures same split
        text_column="text",
        add_eos_token=True,
        chunk_size=512,    # CORRECT: Same as training
        max_eval_samples=200,  # Evaluate on 200 test samples
        max_train_samples=None
    )
    
    logger.info("Loading test dataset (recreates the exact train/test split from training)...")
    
    # Load the datasets - this recreates the same split as during training
    _, test_dataset = load_and_process_datasets(data_config, tokenizer)
    
    logger.info(f"Loaded {len(test_dataset)} test examples")
    
    # Create dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False
    )
    
    # Create evaluation config
    eval_config = EvaluationConfig(
        max_eval_samples=200,
        max_generate_samples=10,  # Generate 10 samples for quality assessment
        save_predictions=True,
        predictions_dir=output_dir,
        # Generation settings
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        # Skip legal domain metrics (TODOs)
        evaluate_citations=False,
        evaluate_legal_terms=False,
        evaluate_reasoning=False
    )
    
    # Create the proper evaluator
    evaluator = LegalEvaluator(
        model=model,
        tokenizer=tokenizer,
        config=eval_config,
        device=model.device,
    )
    
    logger.info("Running comprehensive evaluation...")
    
    # Run evaluation using the proper framework
    metrics = evaluator.evaluate_model(test_dataloader, split="test")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save test dataset info
    with open(os.path.join(output_dir, 'test_info.json'), 'w') as f:
        json.dump({
            'test_samples': len(test_dataset), 
            'data_dir': data_dir, 
            'dataset_pattern': dataset_pattern,
            'train_split': data_config.train_split,
            'eval_split': data_config.eval_split,
            'seed': data_config.seed
        }, f, indent=2)
    
    # Print results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS ON HELD-OUT TEST SET")
    logger.info("=" * 60)
    
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    logger.info(f"\nDetailed results saved to: {output_dir}")
    logger.info(f"Test info saved to: {os.path.join(output_dir, 'test_info.json')}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate model using proper LegalEvaluator framework")
    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True,
        help="Path to the fine-tuned model directory or HuggingFace model ID"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="preprocessing/outputs/llama_training_ready/pre_brexit_2013_2016",
        help="Directory containing the training data"
    )
    parser.add_argument(
        "--dataset-pattern",
        type=str,
        default="pre_brexit_2013_2016_chunks.jsonl",
        help="Dataset file pattern"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation results (auto-generated from model name if not provided)"
    )
    
    args = parser.parse_args()
    
    # Only check local path if it doesn't look like a HF Hub path
    if '/' in args.model_path and not args.model_path.startswith(('http://', 'https://', 'meta-llama/', 'huggingface/')):
        if not os.path.exists(args.model_path):
            logger.error(f"Model path does not exist: {args.model_path}")
            return
    
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory does not exist: {args.data_dir}")
        return
    
    # Run evaluation
    metrics = evaluate_model_properly(
        args.model_path, 
        args.data_dir,
        args.dataset_pattern,
        args.output_dir
    )
    
    if metrics:
        logger.info("Evaluation completed successfully!")
        logger.info("This evaluation used the proper LegalEvaluator framework with the correct held-out test set.")
    else:
        logger.error("Evaluation failed!")

if __name__ == "__main__":
    main() 