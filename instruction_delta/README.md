# Instruction Delta Tool Chain

This directory contains tools for computing and applying instruction deltas between LLaMA base and instruct models. This allows you to:

1. **Create an instruction delta** from base and instruct models once
2. **Apply that delta** to any models fine-tuned from the same base

## 🎯 Purpose for Brexit Research

Perfect for your Brexit legal analysis research! This approach allows you to:
- Train stable models on LLaMA-3-8B base using continued pre-training
- Add instruction capabilities later without retraining
- Maintain comparable models across temporal periods (pre/post-Brexit)

## 📁 Files

- `create_delta.py` - Compute instruction residual (Δ = INSTRUCT - BASE)
- `apply_delta.py` - Apply delta to fine-tuned models (MERGED = MODEL + Δ)
- `test_delta.py` - Unit tests for validation
- `requirements.txt` - Dependencies
- `README.md` - This documentation

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r instruction_delta/requirements.txt
```

### 2. Create the Instruction Delta (Once)

```bash
python instruction_delta/create_delta.py \
    --base_path meta-llama/Meta-Llama-3-8B \
    --instruct_path meta-llama/Meta-Llama-3-8B-Instruct \
    --delta_out ./models/llama3_8b_instruction_delta \
    --dtype fp16
```

**Output:**
- `./models/llama3_8b_instruction_delta/delta.safetensors`
- `./models/llama3_8b_instruction_delta/delta_meta.json`

### 3. Apply Delta to Fine-tuned Models

```bash
# Apply to pre-Brexit model
python instruction_delta/apply_delta.py \
    --model_path ./models/llama3_8b_pre_brexit_2013_2016 \
    --delta_path ./models/llama3_8b_instruction_delta \
    --merged_out ./models/llama3_8b_pre_brexit_2013_2016_instruct

# Apply to post-Brexit model  
python instruction_delta/apply_delta.py \
    --model_path ./models/llama3_8b_post_brexit_2020_2025 \
    --delta_path ./models/llama3_8b_instruction_delta \
    --merged_out ./models/llama3_8b_post_brexit_2020_2025_instruct
```

## 🔧 Advanced Usage

### Working with HuggingFace Hub

```bash
# Create delta from HF models
python instruction_delta/create_delta.py \
    --base_path meta-llama/Meta-Llama-3-8B \
    --instruct_path meta-llama/Meta-Llama-3-8B-Instruct \
    --delta_out your_username/llama3-instruction-delta \
    --dtype fp16

# Apply to local model, push result to hub
python instruction_delta/apply_delta.py \
    --model_path ./models/my_fine_tuned_model \
    --delta_path your_username/llama3-instruction-delta \
    --merged_out ./models/my_instructified_model \
    --push_to_hub your_username/my-instructified-model
```

### Data Types

- `--dtype fp16` (default): Faster, less memory, maintains quality
- `--dtype fp32`: Full precision (larger files)

Embeddings and layer norms are always kept in fp32 for stability.

### Caching

```bash
python instruction_delta/create_delta.py \
    --base_path meta-llama/Meta-Llama-3-8B \
    --instruct_path meta-llama/Meta-Llama-3-8B-Instruct \
    --delta_out ./models/delta \
    --cache_dir ./hf_cache  # Specify cache location
```

## 🧪 Testing

Run the test suite to verify everything works:

```bash
python instruction_delta/test_delta.py
```

Or use pytest:

```bash
cd instruction_delta
pytest test_delta.py -v
```

## 📊 Expected Results

### Sanity Checks

The `apply_delta.py` script automatically performs sanity checks:

✅ **Parameter count**: All parameters should be modified  
✅ **L2 norm**: Delta magnitude (logged for reference)  
✅ **Inference test**: "What is Article 3 ECHR?" → Should get legal response

### Brexit Research Workflow

1. **Create delta once** from LLaMA-3-8B → LLaMA-3-8B-Instruct
2. **Train temporal models** using continued pre-training on base model:
   - `llama3_8b_pre_brexit_2013_2016`
   - `llama3_8b_post_brexit_2020_2025`
3. **Apply delta** to both models:
   - `llama3_8b_pre_brexit_2013_2016_instruct`
   - `llama3_8b_post_brexit_2020_2025_instruct`
4. **Compare** instruction-capable models for Brexit bias analysis

## 🔍 File Structure

```
instruction_delta/
├── create_delta.py       # Create instruction delta
├── apply_delta.py        # Apply delta to models  
├── test_delta.py         # Unit tests
├── requirements.txt      # Dependencies
└── README.md            # This file

models/                   # Your models directory
├── llama3_8b_instruction_delta/
│   ├── delta.safetensors
│   └── delta_meta.json
├── llama3_8b_pre_brexit_2013_2016/
├── llama3_8b_post_brexit_2020_2025/
├── llama3_8b_pre_brexit_2013_2016_instruct/
└── llama3_8b_post_brexit_2020_2025_instruct/
```

## ⚠️ Important Notes

1. **Memory**: Loads full models into CPU memory (~32GB for LLaMA-3-8B)
2. **Compatibility**: Delta must be from same base model as fine-tuned models
3. **Integrity**: Automatic hash verification prevents applying wrong deltas
4. **Performance**: No GPU required for delta operations

## 🎉 Benefits for Your Research

- **Stability**: Train on base model (more stable than instruct)
- **Consistency**: Same delta applied to all temporal variants
- **Efficiency**: Create delta once, apply to many models
- **Comparability**: Identical instruction enhancement across time periods
- **Flexibility**: Can evaluate both base and instruction versions 