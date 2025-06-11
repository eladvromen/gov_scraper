# Case Outcome Prediction with LegalBERT

A complete pipeline for predicting legal case outcomes using fine-tuned LegalBERT.

## 🎯 Overview

This project implements an end-to-end solution for case outcome prediction using:
- **Input**: Legal case text (last 512 tokens of cases)
- **Model**: Fine-tuned LegalBERT 
- **Output**: Case outcome predictions (0, 1, 2) with confidence scores
- **Data Split**: 80% train, 10% validation, 10% test

## 📁 Project Structure

```
case_outcome_prediction/
├── pipeline/          # Core pipeline scripts (01-06 + runner)
├── data/             # Processed datasets and splits
├── models/           # Trained models and baselines
├── results/          # Predictions, evaluations, reports
├── analysis/         # Analysis and diagnostic scripts
├── outputs/          # Plots, visualizations, logs
├── config/           # Configuration files
├── utils/            # Utility functions
└── notebooks/        # Jupyter notebooks
```

## 🚀 Quick Start

```bash
# Run complete pipeline
cd pipeline/
python run_pipeline.py

# Run specific steps
python run_pipeline.py --steps 1,2,3

# Skip inference (for testing)
python run_pipeline.py --skip-inference
```

## 📊 Results

- **Fine-tuned Model Accuracy**: 84.6%
- **Baseline Accuracy**: 44.0%
- **Improvement**: +40.6 percentage points
- **Inference Dataset**: 31,071 legal cases processed

## 📋 Pipeline Steps

1. **Data Preprocessing** (`01_data_preprocessing.py`)
2. **Train/Test Split** (`02_train_test_split.py`)
3. **Baseline Evaluation** (`03_baseline_evaluation.py`)
4. **Fine-tuning** (`04_fine_tuning.py`)
5. **Final Evaluation** (`05_final_evaluation.py`)
6. **Inference Pipeline** (`06_inference_pipeline.py`)

## 💻 Requirements

- Python 3.8+
- GPU recommended (CUDA compatible)
- 8GB+ RAM
- Dependencies: see `requirements.txt`

## 📄 Data

- **Training Data**: 23,824 cases with labels
- **Inference Data**: 31,071 valid cases (73% of total dataset)
- **Missing Cases**: 11,306 cases (Word-document only, not HTML-scrapable)

## 📈 Key Files

- `results/predictions/case_outcome_predictions.parquet` - Full predictions
- `results/evaluation/final_evaluation_results.json` - Performance metrics
- `models/fine_tuned_legal_bert_final/` - Trained model
- `outputs/plots/` - Visualizations and plots

For detailed analysis and results, see the files in the `results/` directory.
