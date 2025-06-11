#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
"""
Step 2: Train/Validation/Test Split for Case Outcome Prediction
==============================================================

This script handles:
1. Loading processed training data
2. Creating stratified train/validation/test splits (80%/10%/10%)
3. Maintaining class distribution across splits
4. Saving split datasets for model training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

class DataSplitter:
    def __init__(self):
        self.processed_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def load_processed_data(self):
        """Load processed training data"""
        print("="*60)
        print("LOADING PROCESSED DATA")
        print("="*60)
        
        data_path = "processed_training_data.parquet"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Processed data not found: {data_path}")
            
        self.processed_data = pd.read_parquet(data_path)
        print(f"✓ Loaded processed data: {self.processed_data.shape}")
        
        # Display label distribution
        label_counts = self.processed_data['label'].value_counts().sort_index()
        print("\nLabel distribution in processed data:")
        for label in sorted(self.processed_data['label'].unique()):
            count = label_counts[label]
            pct = count / len(self.processed_data) * 100
            print(f"  Label {label}: {count:,} cases ({pct:.1f}%)")
            
        return True
    
    def create_stratified_split(self, test_size=0.1, val_size=0.1, random_state=42):
        """Create stratified train/validation/test splits"""
        print("\n" + "="*60)
        print(f"CREATING STRATIFIED SPLITS")
        print("="*60)
        print(f"Target split: {(1-test_size-val_size)*100:.0f}% train, {val_size*100:.0f}% val, {test_size*100:.0f}% test")
        
        # First split: separate test set
        X = self.processed_data.drop('label', axis=1)
        y = self.processed_data['label']
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            stratify=y, 
            random_state=random_state
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_size_adjusted, 
            stratify=y_temp, 
            random_state=random_state
        )
        
        # Combine features and labels back
        self.train_data = pd.concat([X_train, y_train], axis=1)
        self.val_data = pd.concat([X_val, y_val], axis=1)
        self.test_data = pd.concat([X_test, y_test], axis=1)
        
        print(f"✓ Split created:")
        print(f"  Train: {len(self.train_data):,} cases ({len(self.train_data)/len(self.processed_data)*100:.1f}%)")
        print(f"  Val:   {len(self.val_data):,} cases ({len(self.val_data)/len(self.processed_data)*100:.1f}%)")
        print(f"  Test:  {len(self.test_data):,} cases ({len(self.test_data)/len(self.processed_data)*100:.1f}%)")
        
        return True
    
    def validate_split_distribution(self):
        """Validate that class distributions are maintained across splits"""
        print("\n" + "="*60)
        print("VALIDATING SPLIT DISTRIBUTIONS")
        print("="*60)
        
        datasets = {
            'Original': self.processed_data,
            'Train': self.train_data,
            'Val': self.val_data,
            'Test': self.test_data
        }
        
        print("Label distribution across splits:")
        print("-" * 50)
        
        # Header
        labels = sorted(self.processed_data['label'].unique())
        header = "Dataset".ljust(10)
        for label in labels:
            header += f"Label {label}".rjust(12)
        header += "Total".rjust(10)
        print(header)
        print("-" * len(header))
        
        # Data rows
        for name, data in datasets.items():
            label_counts = data['label'].value_counts().sort_index()
            row = name.ljust(10)
            
            for label in labels:
                count = label_counts.get(label, 0)
                pct = count / len(data) * 100
                row += f"{count:,}({pct:.1f}%)".rjust(12)
            
            row += f"{len(data):,}".rjust(10)
            print(row)
        
        # Check if distributions are similar (within 2% tolerance)
        print("\n" + "Distribution validation:")
        original_dist = self.processed_data['label'].value_counts(normalize=True).sort_index()
        
        for name, data in [('Train', self.train_data), ('Val', self.val_data), ('Test', self.test_data)]:
            split_dist = data['label'].value_counts(normalize=True).sort_index()
            max_diff = max(abs(original_dist - split_dist))
            
            if max_diff < 0.02:  # 2% tolerance
                print(f"  ✓ {name} split: Distribution maintained (max diff: {max_diff:.3f})")
            else:
                print(f"  ⚠ {name} split: Distribution may be skewed (max diff: {max_diff:.3f})")
    
    def save_splits(self):
        """Save train/validation/test splits"""
        print("\n" + "="*60)
        print("SAVING SPLIT DATASETS")
        print("="*60)
        
        # Save each split
        splits = {
            'train_data.parquet': self.train_data,
            'val_data.parquet': self.val_data,
            'test_data.parquet': self.test_data
        }
        
        for filename, data in splits.items():
            data.to_parquet(filename, index=False)
            print(f"✓ Saved {filename}: {data.shape}")
        
        # Save split summary
        summary = {
            'total_samples': len(self.processed_data),
            'train_samples': len(self.train_data),
            'val_samples': len(self.val_data),
            'test_samples': len(self.test_data),
            'train_ratio': len(self.train_data) / len(self.processed_data),
            'val_ratio': len(self.val_data) / len(self.processed_data),
            'test_ratio': len(self.test_data) / len(self.processed_data)
        }
        
        summary_df = pd.DataFrame([summary])
        summary_df.to_parquet('split_summary.parquet', index=False)
        print("✓ Saved split_summary.parquet")
        
        return True
    
    def create_dataset_overview(self):
        """Create an overview of the dataset splits"""
        print("\n" + "="*60)
        print("DATASET OVERVIEW")
        print("="*60)
        
        overview_path = "dataset_overview.txt"
        
        with open(overview_path, 'w') as f:
            f.write("CASE OUTCOME PREDICTION - DATASET SPLITS OVERVIEW\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Total processed cases: {len(self.processed_data):,}\n\n")
            
            f.write("Split Distribution:\n")
            f.write(f"  Train: {len(self.train_data):,} cases ({len(self.train_data)/len(self.processed_data)*100:.1f}%)\n")
            f.write(f"  Val:   {len(self.val_data):,} cases ({len(self.val_data)/len(self.processed_data)*100:.1f}%)\n")
            f.write(f"  Test:  {len(self.test_data):,} cases ({len(self.test_data)/len(self.processed_data)*100:.1f}%)\n\n")
            
            f.write("Label Distribution by Split:\n")
            labels = sorted(self.processed_data['label'].unique())
            
            for split_name, split_data in [('Train', self.train_data), ('Val', self.val_data), ('Test', self.test_data)]:
                f.write(f"\n{split_name}:\n")
                label_counts = split_data['label'].value_counts().sort_index()
                for label in labels:
                    count = label_counts.get(label, 0)
                    pct = count / len(split_data) * 100
                    f.write(f"  Label {label}: {count:,} cases ({pct:.1f}%)\n")
        
        print(f"✓ Created dataset overview: {overview_path}")
        
    def run_splitting(self):
        """Run complete data splitting pipeline"""
        print("="*60)
        print("CASE OUTCOME PREDICTION - DATA SPLITTING")
        print("="*60)
        
        # Load processed data
        self.load_processed_data()
        
        # Create stratified splits
        self.create_stratified_split()
        
        # Validate distributions
        self.validate_split_distribution()
        
        # Save splits
        self.save_splits()
        
        # Create overview
        self.create_dataset_overview()
        
        print("\n" + "="*60)
        print("DATA SPLITTING COMPLETE")
        print("="*60)
        print("Files created:")
        print("  - train_data.parquet")
        print("  - val_data.parquet") 
        print("  - test_data.parquet")
        print("  - split_summary.parquet")
        print("  - dataset_overview.txt")
        print("\nNext step: Run 03_baseline_evaluation.py")
        
        return True

def main():
    splitter = DataSplitter()
    splitter.run_splitting()

if __name__ == "__main__":
    main() 