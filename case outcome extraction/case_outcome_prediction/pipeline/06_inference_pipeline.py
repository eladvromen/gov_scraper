#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
"""
Step 6: Inference Pipeline for Case Outcome Prediction
=====================================================

This script handles:
1. Loading the fine-tuned LegalBERT model
2. Processing inference dataset (decision_text_last_section)
3. Making predictions on new legal cases
4. Generating confidence scores and final results
5. Saving predictions for analysis
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import warnings
from datetime import datetime
import json
from tqdm import tqdm

warnings.filterwarnings('ignore')

class CaseOutcomeInference:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Model and tokenizer
        self.model_path = "../models/fine_tuned_legal_bert_final"
        self.tokenizer = None
        self.model = None
        
        # Data
        self.inference_data = None
        self.predictions = None
        
        # Label mapping (will be inferred from model)
        self.label_mapping = {0: "Outcome_0", 1: "Outcome_1", 2: "Outcome_2"}
        
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        print("="*60)
        print("LOADING FINE-TUNED MODEL")
        print("="*60)
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Fine-tuned model not found: {self.model_path}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        
        # Move to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded from: {self.model_path}")
        print(f"✓ Model moved to {self.device}")
        print(f"✓ Number of labels: {self.model.config.num_labels}")
        
        # Update label mapping based on actual number of labels
        num_labels = self.model.config.num_labels
        self.label_mapping = {i: f"Outcome_{i}" for i in range(num_labels)}
        print(f"✓ Label mapping: {self.label_mapping}")
        
        return True
    
    def load_inference_data(self):
        """Load processed inference dataset"""
        print("\n" + "="*60)
        print("LOADING INFERENCE DATA")
        print("="*60)
        
        # Load processed inference data
        inference_path = "../data/processed/processed_inference_data.parquet"
        if not os.path.exists(inference_path):
            raise FileNotFoundError(f"Processed inference data not found: {inference_path}")
        
        self.inference_data = pd.read_parquet(inference_path)
        print(f"✓ Inference data loaded: {self.inference_data.shape}")
        
        # Display basic statistics
        print(f"\nInference dataset statistics:")
        print(f"  Total cases: {len(self.inference_data):,}")
        print(f"  Text column: 'text' (preprocessed decision_text_last_section)")
        
        # Check for any missing data
        missing_text = self.inference_data['text'].isna().sum()
        if missing_text > 0:
            print(f"  ⚠ Warning: {missing_text} cases have missing text")
        
        # Show text length distribution
        text_lengths = self.inference_data['text'].str.len()
        print(f"  Text length - Mean: {text_lengths.mean():.0f}, Median: {text_lengths.median():.0f}")
        print(f"  Text length - Min: {text_lengths.min()}, Max: {text_lengths.max()}")
        
        return True
    
    def predict_batch(self, texts, batch_size=16):
        """Make predictions on a batch of texts"""
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_end = min(i + batch_size, len(texts))
                batch_texts = texts[i:batch_end]
                
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
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                # Convert to predictions
                batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                batch_probs = probs.cpu().numpy()
                
                predictions.extend(batch_predictions)
                probabilities.extend(batch_probs)
        
        return np.array(predictions), np.array(probabilities)
    
    def run_inference(self, batch_size=16):
        """Run inference on all cases"""
        print("\n" + "="*60)
        print("RUNNING INFERENCE")
        print("="*60)
        
        # Filter out cases with missing text
        valid_mask = self.inference_data['text'].notna() & (self.inference_data['text'].str.len() > 0)
        valid_data = self.inference_data[valid_mask].copy()
        
        print(f"Processing {len(valid_data):,} valid cases...")
        
        # Run predictions in batches with progress bar
        all_predictions = []
        all_probabilities = []
        
        texts = valid_data['text'].tolist()
        
        # Use tqdm for progress tracking
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_end = min(i + batch_size, len(texts))
            batch_texts = texts[i:batch_end]
            
            # Get predictions for this batch
            batch_preds, batch_probs = self.predict_batch(batch_texts, batch_size=len(batch_texts))
            
            all_predictions.extend(batch_preds)
            all_probabilities.extend(batch_probs)
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities)
        
        print(f"✓ Inference complete for {len(predictions):,} cases")
        
        # Create results dataframe
        results_df = valid_data.copy()
        results_df['predicted_label'] = predictions
        results_df['predicted_outcome'] = [self.label_mapping[pred] for pred in predictions]
        
        # Add confidence scores
        results_df['prediction_confidence'] = np.max(probabilities, axis=1)
        
        # Add probabilities for each class
        for i, label in self.label_mapping.items():
            results_df[f'prob_{label}'] = probabilities[:, i]
        
        # Add prediction metadata
        results_df['inference_timestamp'] = datetime.now().isoformat()
        results_df['model_path'] = self.model_path
        
        self.predictions = results_df
        
        return results_df
    
    def analyze_predictions(self):
        """Analyze prediction results"""
        print("\n" + "="*60)
        print("ANALYZING PREDICTIONS")
        print("="*60)
        
        if self.predictions is None:
            print("No predictions available")
            return
        
        # Prediction distribution
        print("Prediction distribution:")
        pred_counts = self.predictions['predicted_outcome'].value_counts()
        total_cases = len(self.predictions)
        
        for outcome in pred_counts.index:
            count = pred_counts[outcome]
            pct = count / total_cases * 100
            print(f"  {outcome}: {count:,} cases ({pct:.1f}%)")
        
        # Confidence analysis
        confidence_stats = self.predictions['prediction_confidence'].describe()
        print(f"\nPrediction confidence statistics:")
        print(f"  Mean: {confidence_stats['mean']:.4f}")
        print(f"  Median: {confidence_stats['50%']:.4f}")
        print(f"  Min: {confidence_stats['min']:.4f}")
        print(f"  Max: {confidence_stats['max']:.4f}")
        
        # High/low confidence cases
        high_conf_threshold = 0.8
        low_conf_threshold = 0.6
        
        high_conf_cases = (self.predictions['prediction_confidence'] >= high_conf_threshold).sum()
        low_conf_cases = (self.predictions['prediction_confidence'] <= low_conf_threshold).sum()
        
        print(f"\nConfidence analysis:")
        print(f"  High confidence (≥{high_conf_threshold:.1f}): {high_conf_cases:,} cases ({high_conf_cases/total_cases*100:.1f}%)")
        print(f"  Low confidence (≤{low_conf_threshold:.1f}): {low_conf_cases:,} cases ({low_conf_cases/total_cases*100:.1f}%)")
        
        # Country-wise analysis (if available)
        if 'country' in self.predictions.columns:
            print(f"\nPredictions by country:")
            country_preds = self.predictions.groupby(['country', 'predicted_outcome']).size().unstack(fill_value=0)
            for country in country_preds.index:
                total_country_cases = country_preds.loc[country].sum()
                print(f"  {country}: {total_country_cases:,} cases")
                for outcome in country_preds.columns:
                    count = country_preds.loc[country, outcome]
                    pct = count / total_country_cases * 100 if total_country_cases > 0 else 0
                    print(f"    {outcome}: {count:,} ({pct:.1f}%)")
        
        return True
    
    def save_predictions(self):
        """Save prediction results"""
        print("\n" + "="*60)
        print("SAVING PREDICTIONS")
        print("="*60)
        
        if self.predictions is None:
            print("No predictions to save")
            return False
        
        # Save full predictions
        predictions_path = "../results/predictions/case_outcome_predictions.parquet"
        self.predictions.to_parquet(predictions_path, index=False)
        print(f"✓ Saved full predictions: {predictions_path}")
        
        # Save summary CSV for easy viewing
        summary_columns = [
            'reference_number', 'case_title', 'country', 'case_year',
            'predicted_outcome', 'prediction_confidence'
        ]
        # Only include columns that exist
        summary_columns = [col for col in summary_columns if col in self.predictions.columns]
        
        summary_df = self.predictions[summary_columns].copy()
        summary_df.to_csv("../results/predictions/case_outcome_predictions_summary.csv", index=False)
        print(f"✓ Saved summary CSV: ../results/predictions/case_outcome_predictions_summary.csv")
        
        # Save high confidence predictions separately
        high_conf_threshold = 0.8
        high_conf_predictions = self.predictions[
            self.predictions['prediction_confidence'] >= high_conf_threshold
        ].copy()
        
        if len(high_conf_predictions) > 0:
            high_conf_path = "../results/predictions/high_confidence_predictions.parquet"
            high_conf_predictions.to_parquet(high_conf_path, index=False)
            print(f"✓ Saved high confidence predictions: {high_conf_path}")
        
        # Create prediction summary report
        self.create_prediction_report()
        
        return True
    
    def create_prediction_report(self):
        """Create a comprehensive prediction report"""
        print("Creating prediction report...")
        
        report_path = "../results/reports/case_outcome_prediction_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("CASE OUTCOME PREDICTION REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Prediction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: Fine-tuned LegalBERT\n")
            f.write(f"Model Path: {self.model_path}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Total Predictions: {len(self.predictions):,}\n\n")
            
            f.write("PREDICTION DISTRIBUTION:\n")
            f.write("-" * 30 + "\n")
            pred_counts = self.predictions['predicted_outcome'].value_counts()
            total_cases = len(self.predictions)
            
            for outcome in pred_counts.index:
                count = pred_counts[outcome]
                pct = count / total_cases * 100
                f.write(f"{outcome}: {count:,} cases ({pct:.1f}%)\n")
            
            f.write(f"\nCONFIDENCE ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            confidence_stats = self.predictions['prediction_confidence'].describe()
            f.write(f"Mean Confidence: {confidence_stats['mean']:.4f}\n")
            f.write(f"Median Confidence: {confidence_stats['50%']:.4f}\n")
            f.write(f"Min Confidence: {confidence_stats['min']:.4f}\n")
            f.write(f"Max Confidence: {confidence_stats['max']:.4f}\n")
            
            high_conf_cases = (self.predictions['prediction_confidence'] >= 0.8).sum()
            low_conf_cases = (self.predictions['prediction_confidence'] <= 0.6).sum()
            f.write(f"\nHigh Confidence (≥0.8): {high_conf_cases:,} cases ({high_conf_cases/total_cases*100:.1f}%)\n")
            f.write(f"Low Confidence (≤0.6): {low_conf_cases:,} cases ({low_conf_cases/total_cases*100:.1f}%)\n")
            
            f.write(f"\nFILES GENERATED:\n")
            f.write("-" * 30 + "\n")
            f.write("- case_outcome_predictions.parquet (full results)\n")
            f.write("- case_outcome_predictions_summary.csv (summary view)\n")
            f.write("- high_confidence_predictions.parquet (high confidence cases)\n")
            f.write("- case_outcome_prediction_report.txt (this report)\n")
        
        print(f"✓ Saved prediction report: {report_path}")
    
    def run_inference_pipeline(self):
        """Run complete inference pipeline"""
        print("="*60)
        print("CASE OUTCOME PREDICTION - INFERENCE PIPELINE")
        print("="*60)
        
        # Load fine-tuned model
        self.load_model()
        
        # Load inference data
        self.load_inference_data()
        
        # Run inference
        predictions = self.run_inference()
        
        # Analyze results
        self.analyze_predictions()
        
        # Save results
        self.save_predictions()
        
        print("\n" + "="*60)
        print("INFERENCE PIPELINE COMPLETE")
        print("="*60)
        print("Files created:")
        print("  - case_outcome_predictions.parquet")
        print("  - case_outcome_predictions_summary.csv")
        print("  - high_confidence_predictions.parquet")
        print("  - case_outcome_prediction_report.txt")
        print(f"\nPredicted outcomes for {len(predictions):,} legal cases")
        print("Check the generated files for detailed results and analysis.")
        
        return True

def main():
    # Create inference pipeline
    inference = CaseOutcomeInference()
    
    # Run complete pipeline
    inference.run_inference_pipeline()

if __name__ == "__main__":
    main() 