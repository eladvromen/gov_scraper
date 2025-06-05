#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
"""
Step 1: Data Preprocessing for Case Outcome Prediction
=====================================================

This script handles:
1. Loading training and inference datasets
2. Token validation using LegalBERT tokenizer
3. Text truncation to 512 tokens from the beginning
4. Data preprocessing and validation
"""

import pandas as pd
import pickle
import numpy as np
from transformers import AutoTokenizer
import os
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        # Load LegalBERT tokenizer
        print("Loading LegalBERT tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
        print(f"✓ Tokenizer loaded. Max length: {self.tokenizer.model_max_length}")
        
        self.training_data = None
        self.inference_data = None
        self.processed_training = None
        self.processed_inference = None
        
    def load_datasets(self):
        """Load both training and inference datasets"""
        print("\n" + "="*60)
        print("LOADING DATASETS")
        print("="*60)
        
        # Load training data
        training_path = "/data/shil6369/gov_scraper/data/leaglBERT_training.pkl"
        print(f"Loading training data: {training_path}")
        with open(training_path, 'rb') as f:
            self.training_data = pickle.load(f)
        print(f"✓ Training data loaded: {self.training_data.shape}")
        
        # Load inference data
        inference_path = "/data/shil6369/gov_scraper/data/processed_legal_cases.parquet"
        print(f"Loading inference data: {inference_path}")
        self.inference_data = pd.read_parquet(inference_path)
        print(f"✓ Inference data loaded: {self.inference_data.shape}")
        
        return True
    
    def analyze_labels(self):
        """Analyze label distribution in training data"""
        print("\n" + "="*60)
        print("LABEL ANALYSIS")
        print("="*60)
        
        label_counts = self.training_data['label'].value_counts().sort_index()
        label_percentages = self.training_data['label'].value_counts(normalize=True).sort_index() * 100
        
        print("Label distribution:")
        for label in sorted(self.training_data['label'].unique()):
            count = label_counts[label]
            pct = label_percentages[label]
            print(f"  Label {label}: {count:,} cases ({pct:.1f}%)")
            
        return label_counts
    
    def tokenize_and_validate(self, texts, dataset_name, max_length=512):
        """Tokenize texts and validate/truncate to max_length tokens"""
        print(f"\nTokenizing {dataset_name} data...")
        
        tokenized_data = []
        original_lengths = []
        truncated_count = 0
        
        for i, text in enumerate(texts):
            if i % 5000 == 0:
                print(f"  Processed {i:,}/{len(texts):,} texts...")
                
            # Tokenize the text
            tokens = self.tokenizer.tokenize(str(text))
            original_lengths.append(len(tokens))
            
            # Truncate from beginning if too long (to keep "last part" semantics)
            if len(tokens) > max_length:
                tokens = tokens[-max_length:]  # Keep last max_length tokens
                truncated_count += 1
                
            # Convert back to text
            processed_text = self.tokenizer.convert_tokens_to_string(tokens)
            tokenized_data.append(processed_text)
        
        print(f"✓ Tokenization complete for {dataset_name}")
        print(f"  Original avg length: {np.mean(original_lengths):.1f} tokens")
        print(f"  Texts truncated: {truncated_count:,}/{len(texts):,} ({truncated_count/len(texts)*100:.1f}%)")
        print(f"  Max length after processing: {max([len(self.tokenizer.tokenize(t)) for t in tokenized_data[:100]])}")
        
        return tokenized_data, original_lengths
    
    def process_training_data(self):
        """Process training dataset"""
        print("\n" + "="*60)
        print("PROCESSING TRAINING DATA")
        print("="*60)
        
        # Tokenize and truncate training texts
        processed_texts, original_lengths = self.tokenize_and_validate(
            self.training_data['text'].values, 
            "training"
        )
        
        # Create processed training DataFrame
        self.processed_training = pd.DataFrame({
            'text': processed_texts,
            'label': self.training_data['label'].values,
            'original_token_length': original_lengths,
            'decisionID': self.training_data['decisionID'].values,
            'year': self.training_data['year'].values
        })
        
        print(f"✓ Processed training data shape: {self.processed_training.shape}")
        return self.processed_training
    
    def process_inference_data(self):
        """Process inference dataset"""
        print("\n" + "="*60)
        print("PROCESSING INFERENCE DATA")
        print("="*60)
        
        # Use decision_text_last_section for inference
        inference_texts = self.inference_data['decision_text_last_section'].fillna('').values
        
        # Remove empty texts
        non_empty_mask = [len(str(text).strip()) > 0 for text in inference_texts]
        print(f"Filtering out {sum(~np.array(non_empty_mask)):,} empty texts")
        
        filtered_data = self.inference_data[non_empty_mask].copy()
        filtered_texts = inference_texts[non_empty_mask]
        
        # Tokenize and truncate inference texts
        processed_texts, original_lengths = self.tokenize_and_validate(
            filtered_texts, 
            "inference"
        )
        
        # Create processed inference DataFrame
        self.processed_inference = pd.DataFrame({
            'text': processed_texts,
            'original_token_length': original_lengths,
            'reference_number': filtered_data['reference_number'].values,
            'case_title': filtered_data['case_title'].values,
            'country': filtered_data['country'].values,
            'case_year': filtered_data['case_year'].values,
            'status': filtered_data['status'].values
        })
        
        print(f"✓ Processed inference data shape: {self.processed_inference.shape}")
        return self.processed_inference
    
    def validate_token_lengths(self):
        """Final validation that all texts are within token limits"""
        print("\n" + "="*60)
        print("FINAL TOKEN VALIDATION")
        print("="*60)
        
        # Check training data
        if self.processed_training is not None:
            train_lengths = [len(self.tokenizer.tokenize(text)) for text in self.processed_training['text'].head(100)]
            print(f"Training data - Max tokens in sample: {max(train_lengths)}")
            print(f"Training data - Avg tokens in sample: {np.mean(train_lengths):.1f}")
            
        # Check inference data
        if self.processed_inference is not None:
            inf_lengths = [len(self.tokenizer.tokenize(text)) for text in self.processed_inference['text'].head(100)]
            print(f"Inference data - Max tokens in sample: {max(inf_lengths)}")
            print(f"Inference data - Avg tokens in sample: {np.mean(inf_lengths):.1f}")
            
        print("✓ All texts should be ≤ 512 tokens")
    
    def save_processed_data(self):
        """Save processed datasets"""
        print("\n" + "="*60)
        print("SAVING PROCESSED DATA")
        print("="*60)
        
        # Save training data
        if self.processed_training is not None:
            train_path = "processed_training_data.parquet"
            self.processed_training.to_parquet(train_path, index=False)
            print(f"✓ Saved processed training data: {train_path}")
            
        # Save inference data
        if self.processed_inference is not None:
            inf_path = "processed_inference_data.parquet"
            self.processed_inference.to_parquet(inf_path, index=False)
            print(f"✓ Saved processed inference data: {inf_path}")
            
        return True
    
    def run_preprocessing(self):
        """Run complete preprocessing pipeline"""
        print("="*60)
        print("CASE OUTCOME PREDICTION - DATA PREPROCESSING")
        print("="*60)
        
        # Load datasets
        self.load_datasets()
        
        # Analyze labels
        self.analyze_labels()
        
        # Process training data
        self.process_training_data()
        
        # Process inference data
        self.process_inference_data()
        
        # Final validation
        self.validate_token_lengths()
        
        # Save processed data
        self.save_processed_data()
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        print("Next step: Run 02_train_test_split.py")
        
        return True

def main():
    processor = DataPreprocessor()
    processor.run_preprocessing()

if __name__ == "__main__":
    main() 