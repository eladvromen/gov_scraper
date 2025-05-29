#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
"""
Dataset Analysis Script for Case Outcome Prediction
==================================================

This script analyzes the training and inference datasets for case outcome prediction
to provide insights for model fine-tuning and inference.

Datasets:
1. Training data: leaglBERT_training.pkl
2. Inference data: processed_legal_cases.parquet
"""

import pandas as pd
import pickle
import numpy as np
from collections import Counter
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

class DatasetAnalyzer:
    def __init__(self, training_path, inference_path):
        self.training_path = training_path
        self.inference_path = inference_path
        self.training_data = None
        self.inference_data = None
        
    def load_datasets(self):
        """Load both training and inference datasets"""
        print("="*60)
        print("LOADING DATASETS")
        print("="*60)
        
        # Load training data (pickle)
        try:
            print(f"Loading training data from: {self.training_path}")
            with open(self.training_path, 'rb') as f:
                self.training_data = pickle.load(f)
            print(f"✓ Training data loaded successfully")
            print(f"  Type: {type(self.training_data)}")
        except Exception as e:
            print(f"✗ Error loading training data: {e}")
            return False
            
        # Load inference data (parquet)
        try:
            print(f"Loading inference data from: {self.inference_path}")
            self.inference_data = pd.read_parquet(self.inference_path)
            print(f"✓ Inference data loaded successfully")
            print(f"  Shape: {self.inference_data.shape}")
        except Exception as e:
            print(f"✗ Error loading inference data: {e}")
            return False
            
        return True
    
    def analyze_training_data(self):
        """Analyze the training dataset structure and content"""
        print("\n" + "="*60)
        print("TRAINING DATA ANALYSIS")
        print("="*60)
        
        if self.training_data is None:
            print("Training data not loaded")
            return
            
        # Basic info about the data structure
        print(f"Data type: {type(self.training_data)}")
        
        if isinstance(self.training_data, dict):
            print(f"Dictionary keys: {list(self.training_data.keys())}")
            for key, value in self.training_data.items():
                print(f"  {key}: {type(value)} - {len(value) if hasattr(value, '__len__') else 'N/A'} items")
                
                # If it's a list or array, show first few items
                if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                    print(f"    Sample items: {value[:3]}")
                    
        elif isinstance(self.training_data, pd.DataFrame):
            self.analyze_dataframe(self.training_data, "Training")
            
        elif isinstance(self.training_data, (list, np.ndarray)):
            print(f"Length: {len(self.training_data)}")
            print(f"Sample items: {self.training_data[:3]}")
            
        # Try to identify text and label columns
        self.identify_training_structure()
    
    def identify_training_structure(self):
        """Try to identify the structure of training data for case outcome prediction"""
        print("\n" + "-"*40)
        print("TRAINING DATA STRUCTURE ANALYSIS")
        print("-"*40)
        
        if isinstance(self.training_data, dict):
            # Look for common keys that might contain text and labels
            text_keys = []
            label_keys = []
            
            for key in self.training_data.keys():
                key_lower = key.lower()
                if any(term in key_lower for term in ['text', 'content', 'case', 'document', 'description']):
                    text_keys.append(key)
                elif any(term in key_lower for term in ['label', 'outcome', 'result', 'decision', 'target', 'class']):
                    label_keys.append(key)
                    
            print(f"Potential text columns: {text_keys}")
            print(f"Potential label columns: {label_keys}")
            
            # Analyze labels if found
            for label_key in label_keys:
                labels = self.training_data[label_key]
                if isinstance(labels, (list, np.ndarray)):
                    unique_labels = list(set(labels))
                    print(f"\nLabel analysis for '{label_key}':")
                    print(f"  Unique labels: {unique_labels}")
                    print(f"  Label distribution: {Counter(labels)}")
                    
            # Analyze text if found
            for text_key in text_keys:
                texts = self.training_data[text_key]
                if isinstance(texts, (list, np.ndarray)) and len(texts) > 0:
                    text_lengths = [len(str(text)) for text in texts[:100]]  # Sample first 100
                    print(f"\nText analysis for '{text_key}':")
                    print(f"  Average text length: {np.mean(text_lengths):.1f} characters")
                    print(f"  Text length range: {min(text_lengths)} - {max(text_lengths)}")
                    print(f"  Sample text: {str(texts[0])[:200]}...")
    
    def analyze_dataframe(self, df, name):
        """Analyze a DataFrame structure"""
        print(f"\n{name} DataFrame Analysis:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        
        # Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\nMissing values:\n{missing[missing > 0]}")
        else:
            print("\nNo missing values found")
            
        # Sample data
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        
        # Look for text columns
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
                if isinstance(sample_val, str) and len(sample_val) > 50:
                    text_columns.append(col)
                    
        if text_columns:
            print(f"\nPotential text columns: {text_columns}")
            for col in text_columns:
                text_lengths = df[col].dropna().str.len()
                print(f"  {col}: avg length = {text_lengths.mean():.1f}, range = {text_lengths.min()}-{text_lengths.max()}")
    
    def analyze_inference_data(self):
        """Analyze the inference dataset"""
        print("\n" + "="*60)
        print("INFERENCE DATA ANALYSIS")
        print("="*60)
        
        if self.inference_data is None:
            print("Inference data not loaded")
            return
            
        self.analyze_dataframe(self.inference_data, "Inference")
        
        # Check for overlap in structure with training data
        self.compare_datasets()
    
    def compare_datasets(self):
        """Compare training and inference datasets structure"""
        print("\n" + "="*60)
        print("DATASET COMPARISON")
        print("="*60)
        
        print("Analyzing compatibility between training and inference datasets...")
        
        # If training data is a dict, extract column names
        training_cols = []
        if isinstance(self.training_data, dict):
            training_cols = list(self.training_data.keys())
        elif isinstance(self.training_data, pd.DataFrame):
            training_cols = list(self.training_data.columns)
            
        inference_cols = list(self.inference_data.columns) if self.inference_data is not None else []
        
        print(f"Training data structure: {training_cols}")
        print(f"Inference data columns: {inference_cols}")
        
        # Check for common columns
        if training_cols and inference_cols:
            common_cols = set(training_cols) & set(inference_cols)
            print(f"Common columns: {list(common_cols)}")
            
            training_only = set(training_cols) - set(inference_cols)
            inference_only = set(inference_cols) - set(training_cols)
            
            if training_only:
                print(f"Training-only columns: {list(training_only)}")
            if inference_only:
                print(f"Inference-only columns: {list(inference_only)}")
    
    def generate_modeling_recommendations(self):
        """Generate recommendations for model fine-tuning and inference"""
        print("\n" + "="*60)
        print("MODELING RECOMMENDATIONS")
        print("="*60)
        
        print("Based on the dataset analysis:")
        print("\n1. DATA PREPROCESSING:")
        
        # Check training data structure
        if isinstance(self.training_data, dict):
            print("   - Training data is in dictionary format")
            print("   - Consider converting to pandas DataFrame for easier manipulation")
            
        print("   - Verify text cleaning and normalization requirements")
        print("   - Check for consistent text encoding")
        
        print("\n2. MODEL FINE-TUNING:")
        print("   - Use LegalBERT as base model (already indicated in filename)")
        print("   - Implement proper train/validation split")
        print("   - Consider class balancing if outcomes are imbalanced")
        
        print("\n3. INFERENCE PIPELINE:")
        print("   - Ensure inference data preprocessing matches training preprocessing")
        print("   - Implement batch processing for efficiency")
        print("   - Add confidence scoring for predictions")
        
        print("\n4. EVALUATION:")
        print("   - Use appropriate metrics for case outcome prediction (accuracy, F1, precision, recall)")
        print("   - Consider legal domain-specific evaluation criteria")
    
    def create_summary_report(self):
        """Create a summary report file"""
        report_path = "dataset_summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("CASE OUTCOME PREDICTION - DATASET ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Training Data: {self.training_path}\n")
            f.write(f"Inference Data: {self.inference_path}\n\n")
            
            if self.training_data is not None:
                f.write("TRAINING DATA SUMMARY:\n")
                f.write(f"Type: {type(self.training_data)}\n")
                if isinstance(self.training_data, dict):
                    f.write(f"Keys: {list(self.training_data.keys())}\n")
                elif isinstance(self.training_data, pd.DataFrame):
                    f.write(f"Shape: {self.training_data.shape}\n")
                    f.write(f"Columns: {list(self.training_data.columns)}\n")
                f.write("\n")
                
            if self.inference_data is not None:
                f.write("INFERENCE DATA SUMMARY:\n")
                f.write(f"Shape: {self.inference_data.shape}\n")
                f.write(f"Columns: {list(self.inference_data.columns)}\n")
                
        print(f"\n✓ Summary report saved to: {report_path}")
    
    def run_full_analysis(self):
        """Run complete dataset analysis"""
        if not self.load_datasets():
            return False
            
        self.analyze_training_data()
        self.analyze_inference_data()
        self.generate_modeling_recommendations()
        self.create_summary_report()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("Review the output above and the summary report for next steps.")
        
        return True

def main():
    # Dataset paths
    training_path = "/data/shil6369/gov_scraper/data/leaglBERT_training.pkl"
    inference_path = "/data/shil6369/gov_scraper/data/processed_legal_cases.parquet"
    
    # Create analyzer and run analysis
    analyzer = DatasetAnalyzer(training_path, inference_path)
    success = analyzer.run_full_analysis()
    
    if success:
        print("\nNext steps:")
        print("1. Review the analysis output above")
        print("2. Check the generated summary report")
        print("3. Proceed with model development based on recommendations")
    else:
        print("Analysis failed. Please check the dataset paths and file accessibility.")

if __name__ == "__main__":
    main() 