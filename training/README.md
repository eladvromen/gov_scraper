# LLaMA Legal Domain Fine-tuning Pipeline

## 🏛️ Project Overview

This pipeline fine-tunes LLaMA models on UK Immigration Tribunal legal decisions with Brexit temporal analysis capabilities. It processes 42,378+ legal decisions (2013-2025) across three temporal periods: Pre-Brexit (2013-2016), Transitional (2017), and Post-Brexit (2018-2025).

### ✅ Key Features
- **Data-agnostic pipeline** supporting JSONL and txt formats
- **Hardware optimization** for 2× A100 80GB + 2× L40S GPUs
- **DeepSpeed ZeRO-3** integration with optional flash attention
- **Comprehensive evaluation** (perplexity, ROUGE, BERTScore, legal-specific metrics)
- **Hyperparameter optimization** using Optuna
- **Temporal analysis** capabilities for Brexit impact studies
- **Distributed training** support

## 📁 Directory Structure

```
training/
├── scripts/              # Core pipeline modules
│   ├── data_utils.py     # Dataset loading & processing (148 lines)
│   ├── model_utils.py    # A100 optimized model loading (193 lines)
│   ├── trainer.py        # Distributed training logic (247 lines)
│   ├── train.py          # CLI entry point (99 lines)
│   ├── evaluation.py     # Legal metrics & evaluation (276 lines)
│   └── hyperopt.py       # Optuna optimization (212 lines)
├── tests/                # Comprehensive test suite
│   ├── test_pipeline.py  # 6 passing tests
│   └── __init__.py
├── configs/              # Training configurations
│   ├── base_config.yaml  # Default configuration
│   └── pre_brexit_config.yaml  # Pre-Brexit specific settings
├── requirements.txt      # Dependencies
├── run_training.sh       # Training launcher script
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Activate your virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m pytest tests/ -v  # Should show 6/6 passing tests
```

### 2. Data Preparation
Ensure your data is in the expected format:
```
../preprocessing/outputs/llama_training_ready/
├── pre_brexit_2013_2016/
│   ├── chunk_001.jsonl
│   ├── chunk_002.jsonl
│   └── ...
├── transitional_2017/
│   └── ...
└── post_brexit_2018_2025/
    └── ...
```

### 3. Basic Training
```bash
# Train on pre-Brexit data
./run_training.sh \
  --config configs/pre_brexit_config.yaml \
  --dataset-pattern "pre_brexit_*/*.jsonl" \
  --output-dir "./models/pre_brexit_run" \
  --run-name "llama_pre_brexit_legal"
```

## ⚙️ Configuration Options

### Model Configuration
```yaml
model:
  name: "meta-llama/Llama-2-7b-hf"  # or Llama-3-8B-Instruct
  torch_dtype: "bfloat16"           # Optimized for A100s
  attn_implementation: "flash_attention_2"
```

### Training Configuration
```yaml
training:
  learning_rate: 2e-5
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4    # Effective batch = 16
  deepspeed_stage: 3               # ZeRO-3 optimization
```

### Data Configuration
```yaml
data:
  data_dir: "../preprocessing/outputs/llama_training_ready"
  dataset_patterns: ["pre_brexit_*.jsonl"]  # Flexible patterns
  train_split: 0.95
  chunk_size: 2048
```

## 🎯 Training Modes

### 1. Single Temporal Period
```bash
# Pre-Brexit only
./run_training.sh --dataset-pattern "pre_brexit_*" --run-name "pre_brexit_model"

# Post-Brexit only  
./run_training.sh --dataset-pattern "post_brexit_*" --run-name "post_brexit_model"

# Transitional period only
./run_training.sh --dataset-pattern "transitional_*" --run-name "transitional_model"
```

### 2. Combined Temporal Training
```bash
# All periods combined
./run_training.sh --dataset-pattern "**/*" --run-name "combined_temporal_model"

# Pre + Post Brexit (skip transitional)
python scripts/train.py configs/base_config.yaml \
  --dataset-patterns "pre_brexit_*.jsonl" \
  --dataset-patterns "post_brexit_*.jsonl" \
  --run-name "pre_post_brexit_comparison"
```

### 3. Custom Year Ranges
```bash
# Specific years
./run_training.sh --dataset-pattern "*_2019_*" --run-name "brexit_year_2019"

# Multi-year range
./run_training.sh --dataset-pattern "*_201[6-8]_*" --run-name "brexit_transition_period"
```

## 📊 Evaluation Framework

### Automatic Evaluation During Training
The pipeline automatically evaluates:

#### 1. Basic Language Model Metrics
- **Perplexity**: Model uncertainty on legal text
- **Cross-entropy Loss**: Prediction accuracy
- **Token Accuracy**: Word-level precision

#### 2. Text Generation Quality
- **ROUGE Scores**: Content overlap with reference text
- **BERTScore**: Semantic similarity using DeBERTa

#### 3. Legal Domain Metrics (Planned)
- **Citation Accuracy**: Proper legal citation formats
- **Legal Terminology**: Domain-specific term usage
- **Legal Reasoning**: Argument structure analysis

### Manual Evaluation
```bash
# Run comprehensive evaluation
python scripts/evaluation.py \
  --model-path "./models/pre_brexit_run/final_checkpoint" \
  --eval-data "test_dataset.jsonl" \
  --output-dir "./evaluation_results"
```

## 🔬 Hyperparameter Optimization

### Optuna Integration
```bash
# Optimize hyperparameters
python scripts/hyperopt.py \
  --config configs/base_config.yaml \
  --n-trials 20 \
  --study-name "llama_legal_hpo"
```

### Optimization Options
- Learning rate (1e-6 to 1e-4, log scale)
- Batch size (1 to 32, powers of 2)
- Warmup ratio (0.0 to 0.1)
- Weight decay (0.0 to 0.1)
- Generation parameters (temperature, top_p)

## 🖥️ Hardware Optimization

### Multi-GPU Training
```bash
# 2x A100 80GB
./run_training.sh --num-gpus 2 --run-name "dual_a100_run"

# 4x GPUs (2x A100 + 2x L40S)
./run_training.sh --num-gpus 4 --run-name "quad_gpu_run"
```

### Memory Optimization
- **DeepSpeed ZeRO-3**: Enables training larger models
- **Gradient Checkpointing**: Reduces memory usage
- **bfloat16**: Optimized for A100 hardware
- **Flash Attention**: Faster and more memory-efficient attention

## 📈 Monitoring & Logging

### Weights & Biases Integration
```bash
# Set your W&B entity in config
wandb_entity: "your_username"
wandb_project: "llama-legal-pretraining"
```

### Log Locations
- **Training logs**: `./logs/`
- **Model checkpoints**: `./models/checkpoints/`
- **Evaluation results**: `./evaluation/`
- **W&B dashboard**: Real-time metrics and visualizations

## 🧪 Testing

### Run Test Suite
```bash
# Full test suite (should show 6/6 passing)
python -m pytest tests/ -v

# Individual test components
python -m pytest tests/test_pipeline.py::test_data_loading -v
python -m pytest tests/test_pipeline.py::test_evaluation -v
python -m pytest tests/test_pipeline.py::test_full_pipeline_integration -v
```

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
--per-device-train-batch-size 2

# Increase gradient accumulation
--gradient-accumulation-steps 8

# Enable gradient checkpointing
gradient_checkpointing: true
```

#### 2. DeepSpeed Issues
```bash
# Verify DeepSpeed installation
pip install deepspeed

# Check GPU compatibility
python -c "import deepspeed; print(deepspeed.__version__)"
```

#### 3. Flash Attention Issues
```bash
# Install flash-attn (optional)
pip install flash-attn>=2.3.0

# Or disable in config
attn_implementation: "eager"  # fallback to standard attention
```

## 📚 Brexit Temporal Analysis Examples

### Comparative Training
```bash
# Train separate models for comparison
for period in "pre_brexit" "transitional" "post_brexit"; do
  ./run_training.sh \
    --dataset-pattern "${period}_*" \
    --run-name "llama_${period}_model" \
    --output-dir "./models/${period}_run"
done
```

### Evaluation Comparison
```python
# Compare perplexity across periods
periods = ["pre_brexit", "transitional", "post_brexit"]
for period in periods:
    metrics = evaluate_model(f"./models/{period}_run/final_checkpoint")
    print(f"{period} perplexity: {metrics['validation/perplexity']}")
```

## 🎯 Next Steps

1. **Start with small-scale validation** (100 samples, 1 epoch)
2. **Scale to single temporal period** (e.g., pre-Brexit)
3. **Train comparative models** across all periods
4. **Analyze temporal differences** in legal language
5. **Fine-tune evaluation metrics** for legal domain
6. **Deploy best model** for production use

## 📞 Support

- **Test Suite**: Run `python -m pytest tests/ -v` to verify setup
- **Configuration**: Check `configs/` for examples
- **Logs**: Monitor training progress in W&B dashboard
- **Hardware**: Optimized for 2× A100 80GB + 2× L40S setup

---

**Ready to transform legal AI with temporal language analysis!** 🚀⚖️ 