# Inference Pipeline

This directory contains the inference pipeline for running model inference on all possible vignette permutations.

## 📁 **Organized Directory Structure**

```
inference/
├── 📁 tests/                          # All testing files
│   └── test_new_permutations.py       # Permutation logic tests
│
├── 📁 tools/                          # Interactive utilities
│   ├── run_custom_vignette.py         # Test custom vignette text
│   └── run_specific_vignette.py       # Test specific vignette by name
│
├── 📁 analysis/                       # Post-inference analysis
│   ├── postprocessing/               # Result processing tools
│   ├── basic_analysis/               # Basic inference analysis
│   └── vignettes_analysis/           # Vignette-specific analysis
│
├── 📁 results/                        # Organized output directories
│   ├── legacy/                       # Legacy pipeline outputs
│   ├── production/                   # Production pipeline outputs
│   └── attention/                    # Attention pipeline outputs
│
├── 🔧 Core Pipeline Files
├── inference_pipeline.py              # Main inference pipeline class (LEGACY)
├── production_attention_pipeline.py   # Production pipeline with optimizations
├── attention_pipeline.py              # Token-level attention collection
├── utils.py                          # Shared utility functions
├── calculate_permutations.py          # Permutation counting utilities
│
├── 🚀 Pipeline Runners
├── run_inference.py                   # Legacy pipeline runner
├── run_subset_inference.py            # Legacy subset inference
├── run_subset_inference_production.py # Production subset inference
├── run_attention_analysis.py          # Attention analysis runner
├── example_usage.py                   # Usage examples
│
└── 📚 Documentation
    ├── README.md                      # This file (main documentation)
    ├── PRODUCTION_PIPELINE_README.md  # Production pipeline details
    └── ATTENTION_README.md            # Attention pipeline details
```

## Pipeline Options

### 🚀 **Production Pipeline (RECOMMENDED)** 
- **File:** `production_attention_pipeline.py`
- **Optimized for large-scale inference** (25,000+ samples)
- **Key features:** Proper `sample_id` generation, 5-10x faster batching
- **Critical for analysis:** Enables accurate dataset matching
- **Entry point:** `run_subset_inference_production.py`

### 🏛️ **Legacy Pipeline** 
- **File:** `inference_pipeline.py`
- **Original inference system** with basic functionality
- **Use cases:** Simple inference, backwards compatibility
- **Entry points:** `run_inference.py`, `run_subset_inference.py`

### 🔬 **Attention Pipeline**
- **File:** `attention_pipeline.py`
- **Token-level attention collection** for research analysis
- **Use cases:** Understanding model reasoning, bias detection
- **Entry point:** `run_attention_analysis.py`

## Quick Start

### **🚀 Production Pipeline (Recommended)**

```bash
# Run optimized subset inference with proper sample_id generation
python run_subset_inference_production.py your_model_name \
  --topic-keywords settlement --samples 50 --batch-size 16 \
  --output results/production/settlement_analysis.json
```

### **🛠️ Interactive Tools**

```bash
# Test specific vignette by name
python tools/run_specific_vignette.py your_model_name \
  --vignette-name "Religious persecution & mental health"

# Test custom vignette text
python tools/run_custom_vignette.py your_model_name \
  --text "Custom case description here..."
```

### **🔬 Attention Analysis**

```bash
# Run attention analysis on subset
python run_attention_analysis.py your_model_name \
  --vignettes 3 --samples 10 --attention-rate 0.2
```

## Migration from Unorganized Structure

The directory has been reorganized for clarity:

- **All test files** → `tests/` directory
- **Interactive tools** → `tools/` directory  
- **Analysis & postprocessing** → `analysis/` directory
- **Structured output** → `results/` subdirectories
- **Core pipeline files** remain in root for compatibility

All functionality is preserved - only the organization has improved! 