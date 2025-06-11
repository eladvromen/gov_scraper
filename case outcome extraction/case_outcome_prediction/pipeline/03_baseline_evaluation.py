#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
"""
Step 3: Baseline Evaluation for Case Outcome Prediction
======================================================

This script handles:
1. Loading LegalBERT model and tokenizer
2. Evaluating pre-trained LegalBERT on test set (baseline)
3. Setting up evaluation metrics and framework
4. Preparing for fine-tuning comparison
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import os
import warnings
from datetime import datetime
import json

warnings.filterwarnings('ignore')

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

class BaselineEvaluator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        print("Loading LegalBERT model and tokenizer...")
        self.model_name = 'nlpaueb/legal-bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Note: The pre-trained LegalBERT is not specifically trained for classification
        # We'll need to add a classification head
        self.model = None
        
        # Data
        self.test_data = None
        self.val_data = None
        
    def load_test_data(self):
        """Load test and validation datasets"""
        print("="*60)
        print("LOADING TEST DATA")
        print("="*60)
        
        # Load test data
        if not os.path.exists('test_data.parquet'):
            raise FileNotFoundError("test_data.parquet not found. Run step 2 first.")
        
        self.test_data = pd.read_parquet('test_data.parquet')
        print(f"✓ Test data loaded: {self.test_data.shape}")
        
        # Load validation data for reference
        if os.path.exists('val_data.parquet'):
            self.val_data = pd.read_parquet('val_data.parquet')
            print(f"✓ Validation data loaded: {self.val_data.shape}")
        
        # Display label distribution
        print(f"\nTest set label distribution:")
        test_labels = self.test_data['label'].value_counts().sort_index()
        for label in sorted(self.test_data['label'].unique()):
            count = test_labels[label]
            pct = count / len(self.test_data) * 100
            print(f"  Label {label}: {count:,} cases ({pct:.1f}%)")
        
        return True
    
    def setup_classification_model(self, num_labels=3):
        """Setup LegalBERT for sequence classification"""
        print("\n" + "="*60)
        print("SETTING UP CLASSIFICATION MODEL")
        print("="*60)
        
        # Create model with classification head
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            problem_type="single_label_classification"
        )
        
        # Move to device
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        print(f"✓ Model loaded with {num_labels} output labels")
        print(f"✓ Model moved to {self.device}")
        
        return True
    
    def evaluate_model(self, data, dataset_name="Test"):
        """Evaluate model on given dataset"""
        print(f"\n" + "="*60)
        print(f"EVALUATING ON {dataset_name.upper()} SET")
        print("="*60)
        
        predictions = []
        true_labels = data['label'].values
        
        print(f"Processing {len(data)} samples...")
        
        # Process in batches to avoid memory issues
        batch_size = 16
        
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch_end = min(i + batch_size, len(data))
                batch_texts = data['text'].iloc[i:batch_end].tolist()
                
                if i % (batch_size * 10) == 0:
                    print(f"  Processed {i}/{len(data)} samples...")
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get predictions
                outputs = self.model(**inputs)
                batch_predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                predictions.extend(batch_predictions)
        
        print(f"✓ Evaluation complete")
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            true_labels, predictions, average=None
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Classification report
        class_report = classification_report(true_labels, predictions)
        
        print(f"\n{dataset_name} Set Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Weighted F1: {f1:.4f}")
        print(f"  Weighted Precision: {precision:.4f}")
        print(f"  Weighted Recall: {recall:.4f}")
        
        print(f"\nPer-class metrics:")
        unique_labels = sorted(np.unique(true_labels))
        for i, label in enumerate(unique_labels):
            print(f"  Label {label}: P={precision_per_class[i]:.4f}, R={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}, Support={support[i]}")
        
        print(f"\nConfusion Matrix:")
        print(cm)
        
        print(f"\nDetailed Classification Report:")
        print(class_report)
        
        # Save results
        results = {
            'dataset': dataset_name.lower(),
            'model': 'legal-bert-baseline',
            'timestamp': datetime.now().isoformat(),
            'accuracy': float(accuracy),
            'weighted_f1': float(f1),
            'weighted_precision': float(precision),
            'weighted_recall': float(recall),
            'per_class_precision': precision_per_class.tolist(),
            'per_class_recall': recall_per_class.tolist(),
            'per_class_f1': f1_per_class.tolist(),
            'support': support.tolist(),
            'confusion_matrix': cm.tolist(),
            'predictions': predictions,
            'true_labels': true_labels.tolist()
        }
        
        return results
    
    def random_baseline(self, data, dataset_name="Test"):
        """Create random baseline for comparison"""
        print(f"\n" + "="*60)
        print(f"RANDOM BASELINE ON {dataset_name.upper()} SET")
        print("="*60)
        
        true_labels = data['label'].values
        unique_labels = sorted(np.unique(true_labels))
        
        # Random predictions based on label distribution
        label_distribution = data['label'].value_counts(normalize=True).sort_index()
        random_predictions = np.random.choice(
            unique_labels, 
            size=len(data), 
            p=label_distribution.values
        )
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, random_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, random_predictions, average='weighted'
        )
        
        print(f"Random Baseline Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Weighted F1: {f1:.4f}")
        print(f"  Weighted Precision: {precision:.4f}")
        print(f"  Weighted Recall: {recall:.4f}")
        
        return {
            'accuracy': float(accuracy),
            'weighted_f1': float(f1),
            'weighted_precision': float(precision),
            'weighted_recall': float(recall)
        }
    
    def majority_baseline(self, data, dataset_name="Test"):
        """Create majority class baseline"""
        print(f"\n" + "="*60)
        print(f"MAJORITY BASELINE ON {dataset_name.upper()} SET")
        print("="*60)
        
        true_labels = data['label'].values
        majority_class = data['label'].mode()[0]
        majority_predictions = np.full(len(data), majority_class)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, majority_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, majority_predictions, average='weighted'
        )
        
        print(f"Majority Class Baseline (predicting class {majority_class}):")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Weighted F1: {f1:.4f}")
        print(f"  Weighted Precision: {precision:.4f}")
        print(f"  Weighted Recall: {recall:.4f}")
        
        return {
            'majority_class': int(majority_class.item()),  # Convert numpy/pandas type to Python int
            'accuracy': float(accuracy),
            'weighted_f1': float(f1),
            'weighted_precision': float(precision),
            'weighted_recall': float(recall)
        }
    
    def save_baseline_results(self, legal_bert_results, random_results, majority_results):
        """Save all baseline results"""
        print("\n" + "="*60)
        print("SAVING BASELINE RESULTS")
        print("="*60)
        
        # Combine all results
        baseline_results = {
            'legal_bert_baseline': legal_bert_results,
            'random_baseline': random_results,
            'majority_baseline': majority_results,
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'device': str(self.device)
        }
        
        # Convert any remaining numpy types to Python types
        baseline_results = convert_numpy_types(baseline_results)
        
        # Save to JSON
        with open('baseline_results.json', 'w') as f:
            json.dump(baseline_results, f, indent=2)
        print("✓ Saved baseline_results.json")
        
        # Create summary report
        with open('baseline_summary.txt', 'w') as f:
            f.write("BASELINE EVALUATION SUMMARY\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Test Set Size: {len(self.test_data)}\n\n")
            
            f.write("BASELINE COMPARISON:\n")
            f.write("-" * 30 + "\n")
            f.write(f"LegalBERT (pre-trained):  Acc={legal_bert_results['accuracy']:.4f}, F1={legal_bert_results['weighted_f1']:.4f}\n")
            f.write(f"Random Baseline:          Acc={random_results['accuracy']:.4f}, F1={random_results['weighted_f1']:.4f}\n")
            f.write(f"Majority Class Baseline:  Acc={majority_results['accuracy']:.4f}, F1={majority_results['weighted_f1']:.4f}\n\n")
            
            f.write("NOTES:\n")
            f.write("- LegalBERT baseline uses randomly initialized classification head\n")
            f.write("- Fine-tuning should significantly improve performance\n")
            f.write("- These results serve as lower bounds for fine-tuned model\n")
        
        print("✓ Saved baseline_summary.txt")
        
        return True
    
    def run_baseline_evaluation(self):
        """Run complete baseline evaluation"""
        print("="*60)
        print("CASE OUTCOME PREDICTION - BASELINE EVALUATION")
        print("="*60)
        
        # Load test data
        self.load_test_data()
        
        # Get number of unique labels
        num_labels = len(self.test_data['label'].unique())
        
        # Setup classification model
        self.setup_classification_model(num_labels=num_labels)
        
        # Evaluate LegalBERT baseline (with random classification head)
        legal_bert_results = self.evaluate_model(self.test_data, "Test")
        
        # Create comparison baselines
        random_results = self.random_baseline(self.test_data, "Test")
        majority_results = self.majority_baseline(self.test_data, "Test")
        
        # Save results
        self.save_baseline_results(legal_bert_results, random_results, majority_results)
        
        print("\n" + "="*60)
        print("BASELINE EVALUATION COMPLETE")
        print("="*60)
        print("Files created:")
        print("  - baseline_results.json")
        print("  - baseline_summary.txt")
        print("\nNext step: Run 04_fine_tuning.py")
        
        return True

def main():
    evaluator = BaselineEvaluator()
    evaluator.run_baseline_evaluation()

if __name__ == "__main__":
    main() 