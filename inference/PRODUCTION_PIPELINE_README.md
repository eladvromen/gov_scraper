# Production Pipeline Tools

## Overview

The production pipeline provides optimized inference capabilities with proper `sample_id` generation that is critical for accurate dataset matching in analysis. This addresses the core issue we discovered where analysis was failing due to incorrect case matching between datasets.

## Key Features

### 1. Proper Sample ID Generation
- **Critical Fix**: Generates sequential `sample_id` values (1, 2, 3, ...) for each vignette permutation
- **Dataset Matching**: Enables accurate matching between pre-Brexit and post-Brexit datasets
- **Replaces**: Unreliable composite key matching that caused analysis errors

### 2. Optimized Batch Processing
- **5-10x Faster**: Batched inference instead of individual processing
- **Memory Efficient**: GPU memory management with automatic cache clearing
- **Scalable**: Handles thousands of samples efficiently

### 3. Flexible Attention Collection
- **Two Modes**: With or without attention data collection
- **Performance**: Non-attention mode is significantly faster
- **Storage**: Efficient HDF5 storage for attention data when needed

## Tools

### 1. `production_attention_pipeline.py`

The main pipeline class with two primary methods:

```bash
# Non-attention mode (fast inference only)
python production_attention_pipeline.py my_model --no-attention --output results.json

# Attention mode (inference + attention data collection)
python production_attention_pipeline.py my_model --output results.json --attention-rate 0.8
```

**Key Methods:**
- `generate_inference_records_optimized()` - Fast inference without attention
- `generate_inference_records_with_attention_optimized()` - Inference with attention collection

### 2. `run_subset_inference_production.py`

Subset inference using the production pipeline:

```bash
# Run 3 samples from settlement-related vignettes
python run_subset_inference_production.py my_model --topic-keywords settlement --samples 3

# Run from specific topics with larger batch size
python run_subset_inference_production.py my_model --topics "Firm settlement" --samples 5 --batch-size 32

# Use HF Hub model
python run_subset_inference_production.py "meta-llama/Meta-Llama-3-8B-Instruct" --use-hf-hub --samples 3
```

### 3. `test_production_pipeline.py`

Test script to verify both modes work correctly:

```bash
python test_production_pipeline.py
```

## Sample ID Generation

**Critical for Analysis**: The pipeline generates sequential `sample_id` values that serve as the authoritative identifier for matching cases between datasets.

**Example Output Format:**
```json
{
  "sample_id": 1,
  "meta_topic": "Persecution",
  "topic": "Firm settlement",
  "fields": {
    "country": "Syria",
    "gender": "male"
  },
  "vignette_text": "...",
  "model_response": "...",
  "inference_timestamp": "2025-01-15T10:30:00"
}
```

## Performance Comparison

| Mode | Speed | Memory | Use Case |
|------|-------|--------|----------|
| Non-attention | 5-10x faster | Lower | Analysis, evaluation |
| Attention | Baseline | Higher | Research, debugging |

## Integration with Analysis

The production pipeline outputs are compatible with existing analysis tools:

- `compare_inference_results.py` - Now works with proper sample_id matching
- `vignette_viewer.py` - Can use sample_id for accurate case linking
- `process_inference_results.py` - Preserves sample_id in processed outputs

## Migration from Basic Pipeline

**Before (unreliable matching):**
```python
# Old way with composite keys
from inference_pipeline import InferencePipeline
pipeline = InferencePipeline(model)
results = pipeline.run_inference_auto(texts)
# No sample_id, unreliable matching
```

**After (reliable matching):**
```python
# New way with proper sample_id
from production_attention_pipeline import ProductionAttentionPipeline
pipeline = ProductionAttentionPipeline(model, collect_attention=False)
results = pipeline.generate_inference_records_optimized(vignettes)
# Every record has proper sample_id
```

## Attention Data Storage

When attention collection is enabled:

- **HDF5 Format**: Efficient compressed storage
- **Metadata**: JSON file with sample information
- **Analysis Tools**: Load with `load_attention_data()` and `analyze_sample_attention()`

## Best Practices

1. **Use non-attention mode** for routine inference and analysis
2. **Enable attention mode** only when you need attention data for research
3. **Preserve sample_id** in all downstream processing
4. **Use sample_id** as the primary key for dataset matching
5. **Set appropriate batch sizes** based on GPU memory (16-32 for A100s)

## Troubleshooting

**Memory Issues:**
- Reduce `batch_size` parameter
- Use non-attention mode
- Clear GPU cache between runs

**Performance Issues:**
- Increase `batch_size` if memory allows
- Use non-attention mode
- Consider GPU vs CPU processing

**Matching Issues:**
- Verify `sample_id` fields exist in all datasets
- Use `sample_id` instead of composite keys for matching
- Check that processing preserves `sample_id` values 