#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
"""
Step 4: Fine-tuning LegalBERT for Case Outcome Prediction
========================================================

This script handles:
1. Loading train/validation datasets
2. Setting up fine-tuning configuration
3. Fine-tuning LegalBERT with HuggingFace Trainer
4. Monitoring training progress and validation metrics
5. Saving the fine-tuned model
"""

import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import warnings
from datetime import datetime
import json

warnings.filterwarnings('ignore')

class LegalBERTFineTuner:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Model and tokenizer
        self.model_name = 'nlpaueb/legal-bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = None
        
        # Data
        self.train_data = None
        self.val_data = None
        self.train_dataset = None
        self.val_dataset = None
        
        # Training
        self.trainer = None
        self.num_labels = None
        
    def load_split_data(self):
        """Load train and validation datasets"""
        print("="*60)
        print("LOADING SPLIT DATA")
        print("="*60)
        
        # Load training data
        if not os.path.exists('train_data.parquet'):
            raise FileNotFoundError("train_data.parquet not found. Run step 2 first.")
        
        self.train_data = pd.read_parquet('train_data.parquet')
        print(f"✓ Training data loaded: {self.train_data.shape}")
        
        # Load validation data
        if not os.path.exists('val_data.parquet'):
            raise FileNotFoundError("val_data.parquet not found. Run step 2 first.")
        
        self.val_data = pd.read_parquet('val_data.parquet')
        print(f"✓ Validation data loaded: {self.val_data.shape}")
        
        # Get number of unique labels
        self.num_labels = len(self.train_data['label'].unique())
        print(f"✓ Number of labels: {self.num_labels}")
        
        # Display label distributions
        print(f"\nTraining set label distribution:")
        train_labels = self.train_data['label'].value_counts().sort_index()
        for label in sorted(self.train_data['label'].unique()):
            count = train_labels[label]
            pct = count / len(self.train_data) * 100
            print(f"  Label {label}: {count:,} cases ({pct:.1f}%)")
        
        print(f"\nValidation set label distribution:")
        val_labels = self.val_data['label'].value_counts().sort_index()
        for label in sorted(self.val_data['label'].unique()):
            count = val_labels[label]
            pct = count / len(self.val_data) * 100
            print(f"  Label {label}: {count:,} cases ({pct:.1f}%)")
        
        return True
    
    def prepare_datasets(self):
        """Prepare datasets for training"""
        print("\n" + "="*60)
        print("PREPARING DATASETS")
        print("="*60)
        
        # Convert to HuggingFace datasets
        train_dict = {
            'text': self.train_data['text'].tolist(),
            'labels': self.train_data['label'].tolist()
        }
        self.train_dataset = Dataset.from_dict(train_dict)
        
        val_dict = {
            'text': self.val_data['text'].tolist(),
            'labels': self.val_data['label'].tolist()
        }
        self.val_dataset = Dataset.from_dict(val_dict)
        
        print(f"✓ Training dataset: {len(self.train_dataset)} samples")
        print(f"✓ Validation dataset: {len(self.val_dataset)} samples")
        
        # Tokenize datasets
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding=False,  # Will be done by data collator
                truncation=True,
                max_length=512
            )
        
        print("Tokenizing datasets...")
        self.train_dataset = self.train_dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=['text']
        )
        
        self.val_dataset = self.val_dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=['text']
        )
        
        print("✓ Tokenization complete")
        
        return True
    
    def setup_model(self):
        """Setup LegalBERT model for classification"""
        print("\n" + "="*60)
        print("SETTING UP MODEL")
        print("="*60)
        
        # Load model with classification head
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="single_label_classification"
        )
        
        # Move to device
        self.model.to(self.device)
        
        print(f"✓ Model loaded with {self.num_labels} labels")
        print(f"✓ Model moved to {self.device}")
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        
        return True
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def setup_training_arguments(self):
        """Setup training configuration"""
        print("\n" + "="*60)
        print("SETTING UP TRAINING CONFIGURATION")
        print("="*60)
        
        # Create output directory
        output_dir = "./fine_tuned_legal_bert"
        os.makedirs(output_dir, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            
            # Training hyperparameters
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            learning_rate=2e-5,
            
            # Evaluation and logging
            eval_strategy="steps",
            eval_steps=500,
            logging_dir='./logs',
            logging_steps=100,
            
            # Saving
            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            
            # Early stopping
            metric_for_best_model="f1",
            greater_is_better=True,
            
            # Other settings
            seed=42,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
            dataloader_pin_memory=False,
            remove_unused_columns=True,
            
            # Reporting
            report_to=None,  # Don't use wandb/tensorboard
        )
        
        print("✓ Training configuration:")
        print(f"  Epochs: {training_args.num_train_epochs}")
        print(f"  Train batch size: {training_args.per_device_train_batch_size}")
        print(f"  Eval batch size: {training_args.per_device_eval_batch_size}")
        print(f"  Learning rate: {training_args.learning_rate}")
        print(f"  Weight decay: {training_args.weight_decay}")
        print(f"  Warmup steps: {training_args.warmup_steps}")
        print(f"  Mixed precision: {training_args.fp16}")
        
        return training_args
    
    def setup_trainer(self, training_args):
        """Setup HuggingFace Trainer"""
        print("\n" + "="*60)
        print("SETTING UP TRAINER")
        print("="*60)
        
        # Data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # Early stopping callback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.001
        )
        
        # Setup trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[early_stopping]
        )
        
        print("✓ Trainer configured with:")
        print(f"  Training samples: {len(self.train_dataset)}")
        print(f"  Validation samples: {len(self.val_dataset)}")
        print("  Early stopping enabled (patience=3)")
        
        return True
    
    def train_model(self):
        """Fine-tune the model"""
        print("\n" + "="*60)
        print("STARTING FINE-TUNING")
        print("="*60)
        
        print("Training will begin. This may take a while...")
        print("Monitor progress in the logs directory.")
        
        # Start training
        start_time = datetime.now()
        train_result = self.trainer.train()
        end_time = datetime.now()
        
        training_time = end_time - start_time
        print(f"\n✓ Training completed in {training_time}")
        
        # Save training results
        train_metrics = train_result.metrics
        train_metrics['training_time'] = str(training_time)
        
        print("\nFinal training metrics:")
        for key, value in train_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        return train_result
    
    def evaluate_model(self):
        """Evaluate the fine-tuned model"""
        print("\n" + "="*60)
        print("EVALUATING FINE-TUNED MODEL")
        print("="*60)
        
        # Evaluate on validation set
        eval_results = self.trainer.evaluate()
        
        print("Validation set results:")
        for key, value in eval_results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        return eval_results
    
    def save_model_and_results(self, train_result, eval_results):
        """Save the fine-tuned model and results"""
        print("\n" + "="*60)
        print("SAVING MODEL AND RESULTS")
        print("="*60)
        
        # Save model and tokenizer
        model_save_path = "./fine_tuned_legal_bert_final"
        self.trainer.save_model(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)
        
        print(f"✓ Model saved to: {model_save_path}")
        
        # Save training results
        results = {
            'model_name': self.model_name,
            'fine_tuning_timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'num_labels': self.num_labels,
            'train_samples': len(self.train_dataset),
            'val_samples': len(self.val_dataset),
            'train_metrics': train_result.metrics,
            'eval_metrics': eval_results,
            'model_save_path': model_save_path
        }
        
        # Save to JSON
        with open('fine_tuning_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print("✓ Saved fine_tuning_results.json")
        
        # Create summary report
        with open('fine_tuning_summary.txt', 'w') as f:
            f.write("FINE-TUNING RESULTS SUMMARY\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Fine-tuning Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Base Model: {self.model_name}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Number of Labels: {self.num_labels}\n")
            f.write(f"Training Samples: {len(self.train_dataset):,}\n")
            f.write(f"Validation Samples: {len(self.val_dataset):,}\n\n")
            
            f.write("FINAL VALIDATION METRICS:\n")
            f.write("-" * 30 + "\n")
            for key, value in eval_results.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            
            f.write(f"\nModel saved to: {model_save_path}\n")
        
        print("✓ Saved fine_tuning_summary.txt")
        
        return True
    
    def run_fine_tuning(self):
        """Run complete fine-tuning pipeline"""
        print("="*60)
        print("CASE OUTCOME PREDICTION - FINE-TUNING")
        print("="*60)
        
        # Load data
        self.load_split_data()
        
        # Prepare datasets
        self.prepare_datasets()
        
        # Setup model
        self.setup_model()
        
        # Setup training
        training_args = self.setup_training_arguments()
        self.setup_trainer(training_args)
        
        # Train model
        train_result = self.train_model()
        
        # Evaluate model
        eval_results = self.evaluate_model()
        
        # Save everything
        self.save_model_and_results(train_result, eval_results)
        
        print("\n" + "="*60)
        print("FINE-TUNING COMPLETE")
        print("="*60)
        print("Files created:")
        print("  - ./fine_tuned_legal_bert_final/ (model directory)")
        print("  - fine_tuning_results.json")
        print("  - fine_tuning_summary.txt")
        print("  - ./logs/ (training logs)")
        print("\nNext step: Run 05_final_evaluation.py")
        
        return True

def main():
    fine_tuner = LegalBERTFineTuner()
    fine_tuner.run_fine_tuning()

if __name__ == "__main__":
    main() 