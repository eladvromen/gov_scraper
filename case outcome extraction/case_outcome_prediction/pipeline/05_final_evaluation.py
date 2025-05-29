#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
"""
Step 5: Final Evaluation of Fine-tuned LegalBERT
===============================================

This script handles:
1. Loading the fine-tuned model
2. Evaluating on test set with comprehensive metrics
3. Comparing with baseline performance
4. Generating final performance report
5. Preparing the model for inference
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix, 
    classification_report,
    roc_auc_score
)
import os
import warnings
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class FinalEvaluator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Model and tokenizer
        self.model_path = "./fine_tuned_legal_bert_final"
        self.tokenizer = None
        self.model = None
        
        # Data
        self.test_data = None
        self.baseline_results = None
        
    def load_fine_tuned_model(self):
        """Load the fine-tuned model and tokenizer"""
        print("="*60)
        print("LOADING FINE-TUNED MODEL")
        print("="*60)
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Fine-tuned model not found: {self.model_path}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        
        # Move to device
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Fine-tuned model loaded from: {self.model_path}")
        print(f"✓ Model moved to {self.device}")
        print(f"✓ Number of labels: {self.model.config.num_labels}")
        
        return True
    
    def load_test_data(self):
        """Load test dataset and baseline results"""
        print("\n" + "="*60)
        print("LOADING TEST DATA AND BASELINE")
        print("="*60)
        
        # Load test data
        if not os.path.exists('test_data.parquet'):
            raise FileNotFoundError("test_data.parquet not found. Run step 2 first.")
        
        self.test_data = pd.read_parquet('test_data.parquet')
        print(f"✓ Test data loaded: {self.test_data.shape}")
        
        # Load baseline results for comparison
        if os.path.exists('baseline_results.json'):
            with open('baseline_results.json', 'r') as f:
                self.baseline_results = json.load(f)
            print("✓ Baseline results loaded for comparison")
        else:
            print("⚠ Baseline results not found - will skip comparison")
        
        # Display test set info
        print(f"\nTest set label distribution:")
        test_labels = self.test_data['label'].value_counts().sort_index()
        for label in sorted(self.test_data['label'].unique()):
            count = test_labels[label]
            pct = count / len(self.test_data) * 100
            print(f"  Label {label}: {count:,} cases ({pct:.1f}%)")
        
        return True
    
    def evaluate_on_test_set(self):
        """Evaluate fine-tuned model on test set"""
        print("\n" + "="*60)
        print("EVALUATING ON TEST SET")
        print("="*60)
        
        predictions = []
        prediction_probs = []
        true_labels = self.test_data['label'].values
        
        print(f"Processing {len(self.test_data)} test samples...")
        
        # Process in batches
        batch_size = 16
        
        with torch.no_grad():
            for i in range(0, len(self.test_data), batch_size):
                batch_end = min(i + batch_size, len(self.test_data))
                batch_texts = self.test_data['text'].iloc[i:batch_end].tolist()
                
                if i % (batch_size * 10) == 0:
                    print(f"  Processed {i}/{len(self.test_data)} samples...")
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get predictions and probabilities
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                batch_probs = probs.cpu().numpy()
                
                predictions.extend(batch_predictions)
                prediction_probs.extend(batch_probs)
        
        print("✓ Evaluation complete")
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        prediction_probs = np.array(prediction_probs)
        
        # Calculate comprehensive metrics
        results = self.calculate_comprehensive_metrics(
            true_labels, predictions, prediction_probs
        )
        
        return results
    
    def calculate_comprehensive_metrics(self, true_labels, predictions, prediction_probs):
        """Calculate comprehensive evaluation metrics"""
        print("\n" + "="*60)
        print("CALCULATING METRICS")
        print("="*60)
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            true_labels, predictions, average=None
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Classification report
        class_report = classification_report(true_labels, predictions, output_dict=True)
        
        # ROC AUC (for multi-class)
        try:
            if len(np.unique(true_labels)) > 2:
                auc_score = roc_auc_score(true_labels, prediction_probs, multi_class='ovr', average='weighted')
            else:
                auc_score = roc_auc_score(true_labels, prediction_probs[:, 1])
        except:
            auc_score = None
        
        print("Test Set Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Weighted F1: {f1:.4f}")
        print(f"  Weighted Precision: {precision:.4f}")
        print(f"  Weighted Recall: {recall:.4f}")
        if auc_score:
            print(f"  Weighted AUC: {auc_score:.4f}")
        
        print(f"\nPer-class metrics:")
        unique_labels = sorted(np.unique(true_labels))
        for i, label in enumerate(unique_labels):
            print(f"  Label {label}: P={precision_per_class[i]:.4f}, R={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}, Support={support_per_class[i]}")
        
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Package results
        results = {
            'model_type': 'fine_tuned_legal_bert',
            'evaluation_timestamp': datetime.now().isoformat(),
            'test_samples': len(true_labels),
            'accuracy': float(accuracy),
            'weighted_f1': float(f1),
            'weighted_precision': float(precision),
            'weighted_recall': float(recall),
            'weighted_auc': float(auc_score) if auc_score else None,
            'per_class_precision': precision_per_class.tolist(),
            'per_class_recall': recall_per_class.tolist(),
            'per_class_f1': f1_per_class.tolist(),
            'support': support_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': predictions.tolist(),
            'true_labels': true_labels.tolist(),
            'prediction_probabilities': prediction_probs.tolist()
        }
        
        return results
    
    def compare_with_baseline(self, fine_tuned_results):
        """Compare fine-tuned results with baseline"""
        print("\n" + "="*60)
        print("BASELINE COMPARISON")
        print("="*60)
        
        if self.baseline_results is None:
            print("No baseline results available for comparison")
            return None
        
        # Extract baseline metrics
        baseline_bert = self.baseline_results['legal_bert_baseline']
        baseline_random = self.baseline_results['random_baseline']
        baseline_majority = self.baseline_results['majority_baseline']
        
        # Fine-tuned metrics
        ft_metrics = fine_tuned_results
        
        print("Performance Comparison:")
        print("-" * 50)
        print(f"{'Model':<25} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10}")
        print("-" * 50)
        print(f"{'Fine-tuned LegalBERT':<25} {ft_metrics['accuracy']:<10.4f} {ft_metrics['weighted_f1']:<10.4f} {ft_metrics['weighted_precision']:<10.4f} {ft_metrics['weighted_recall']:<10.4f}")
        print(f"{'LegalBERT Baseline':<25} {baseline_bert['accuracy']:<10.4f} {baseline_bert['weighted_f1']:<10.4f} {baseline_bert['weighted_precision']:<10.4f} {baseline_bert['weighted_recall']:<10.4f}")
        print(f"{'Random Baseline':<25} {baseline_random['accuracy']:<10.4f} {baseline_random['weighted_f1']:<10.4f} {baseline_random['weighted_precision']:<10.4f} {baseline_random['weighted_recall']:<10.4f}")
        print(f"{'Majority Baseline':<25} {baseline_majority['accuracy']:<10.4f} {baseline_majority['weighted_f1']:<10.4f} {baseline_majority['weighted_precision']:<10.4f} {baseline_majority['weighted_recall']:<10.4f}")
        
        # Calculate improvements
        improvements = {
            'accuracy_improvement': ft_metrics['accuracy'] - baseline_bert['accuracy'],
            'f1_improvement': ft_metrics['weighted_f1'] - baseline_bert['weighted_f1'],
            'precision_improvement': ft_metrics['weighted_precision'] - baseline_bert['weighted_precision'],
            'recall_improvement': ft_metrics['weighted_recall'] - baseline_bert['weighted_recall']
        }
        
        print(f"\nImprovements over LegalBERT baseline:")
        for metric, improvement in improvements.items():
            print(f"  {metric}: +{improvement:.4f}")
        
        return {
            'baseline_comparison': {
                'fine_tuned': ft_metrics,
                'legal_bert_baseline': baseline_bert,
                'random_baseline': baseline_random,
                'majority_baseline': baseline_majority
            },
            'improvements': improvements
        }
    
    def create_visualizations(self, results):
        """Create visualization plots"""
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        try:
            # Confusion Matrix Heatmap
            plt.figure(figsize=(8, 6))
            cm = np.array(results['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix - Fine-tuned LegalBERT')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved confusion_matrix.png")
            
            # Performance comparison (if baseline available)
            if self.baseline_results:
                plt.figure(figsize=(10, 6))
                
                models = ['Fine-tuned\nLegalBERT', 'LegalBERT\nBaseline', 'Random\nBaseline', 'Majority\nBaseline']
                accuracy = [
                    results['accuracy'],
                    self.baseline_results['legal_bert_baseline']['accuracy'],
                    self.baseline_results['random_baseline']['accuracy'],
                    self.baseline_results['majority_baseline']['accuracy']
                ]
                f1 = [
                    results['weighted_f1'],
                    self.baseline_results['legal_bert_baseline']['weighted_f1'],
                    self.baseline_results['random_baseline']['weighted_f1'],
                    self.baseline_results['majority_baseline']['weighted_f1']
                ]
                
                x = np.arange(len(models))
                width = 0.35
                
                plt.bar(x - width/2, accuracy, width, label='Accuracy', alpha=0.8)
                plt.bar(x + width/2, f1, width, label='Weighted F1', alpha=0.8)
                
                plt.xlabel('Models')
                plt.ylabel('Score')
                plt.title('Model Performance Comparison')
                plt.xticks(x, models)
                plt.legend()
                plt.ylim(0, 1)
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("✓ Saved performance_comparison.png")
                
        except Exception as e:
            print(f"⚠ Could not create visualizations: {e}")
    
    def save_final_results(self, results, comparison_results=None):
        """Save final evaluation results"""
        print("\n" + "="*60)
        print("SAVING FINAL RESULTS")
        print("="*60)
        
        # Combine all results
        final_results = {
            'evaluation_results': results,
            'baseline_comparison': comparison_results,
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'device': str(self.device)
        }
        
        # Save to JSON
        with open('final_evaluation_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        print("✓ Saved final_evaluation_results.json")
        
        # Create comprehensive report
        with open('final_evaluation_report.txt', 'w') as f:
            f.write("FINAL EVALUATION REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: Fine-tuned LegalBERT\n")
            f.write(f"Model Path: {self.model_path}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Test Set Size: {results['test_samples']:,}\n\n")
            
            f.write("FINAL TEST SET PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Weighted F1: {results['weighted_f1']:.4f}\n")
            f.write(f"Weighted Precision: {results['weighted_precision']:.4f}\n")
            f.write(f"Weighted Recall: {results['weighted_recall']:.4f}\n")
            if results['weighted_auc']:
                f.write(f"Weighted AUC: {results['weighted_auc']:.4f}\n")
            
            f.write(f"\nPER-CLASS PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            unique_labels = sorted(set(results['true_labels']))
            for i, label in enumerate(unique_labels):
                f.write(f"Label {label}:\n")
                f.write(f"  Precision: {results['per_class_precision'][i]:.4f}\n")
                f.write(f"  Recall: {results['per_class_recall'][i]:.4f}\n")
                f.write(f"  F1: {results['per_class_f1'][i]:.4f}\n")
                f.write(f"  Support: {results['support'][i]}\n")
            
            if comparison_results:
                f.write(f"\nIMPROVEMENTS OVER BASELINE:\n")
                f.write("-" * 30 + "\n")
                for metric, improvement in comparison_results['improvements'].items():
                    f.write(f"{metric}: +{improvement:.4f}\n")
            
            f.write(f"\nMODEL READY FOR INFERENCE\n")
            f.write("-" * 30 + "\n")
            f.write(f"The fine-tuned model is saved at: {self.model_path}\n")
            f.write("Use this model for case outcome prediction on new legal cases.\n")
        
        print("✓ Saved final_evaluation_report.txt")
        
        return True
    
    def run_final_evaluation(self):
        """Run complete final evaluation"""
        print("="*60)
        print("CASE OUTCOME PREDICTION - FINAL EVALUATION")
        print("="*60)
        
        # Load fine-tuned model
        self.load_fine_tuned_model()
        
        # Load test data and baseline
        self.load_test_data()
        
        # Evaluate on test set
        results = self.evaluate_on_test_set()
        
        # Compare with baseline
        comparison_results = self.compare_with_baseline(results)
        
        # Create visualizations
        self.create_visualizations(results)
        
        # Save final results
        self.save_final_results(results, comparison_results)
        
        print("\n" + "="*60)
        print("FINAL EVALUATION COMPLETE")
        print("="*60)
        print("Files created:")
        print("  - final_evaluation_results.json")
        print("  - final_evaluation_report.txt")
        print("  - confusion_matrix.png")
        print("  - performance_comparison.png (if baseline available)")
        print(f"\nFinal Model Performance:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Weighted F1: {results['weighted_f1']:.4f}")
        print(f"\nModel ready for inference at: {self.model_path}")
        print("\nNext step: Run 06_inference_pipeline.py for production inference")
        
        return True

def main():
    evaluator = FinalEvaluator()
    evaluator.run_final_evaluation()

if __name__ == "__main__":
    main() 