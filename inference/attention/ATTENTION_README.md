# Token-Level Attention Analysis for UK Immigration Study

This directory now includes enhanced inference capabilities with **token-level attention collection** for detailed template analysis.

## 🎯 Key Features

### Token-Level Attention Analysis
- **Individual token attention scores** - See exactly which words the model focuses on
- **Template region mapping** - Identify attention to demographics vs. narrative vs. legal arguments  
- **Top-k attended tokens** per generation step with scores
- **Multi-head analysis** - Compare attention patterns across different attention heads
- **CSV export** for easy analysis and visualization

### Perfect for Template Analysis
- **Word-level granularity** - Essential for understanding bias in demographic field attention
- **Template component breakdown** - Separate attention to names, ages, countries, etc.
- **Generation step tracking** - See how attention evolves during decision generation
- **Analysis-ready formats** - JSON, CSV, and summary files for different use cases

## 📁 Files

- **`attention_pipeline.py`** - Main token-level attention pipeline class
- **`run_attention_analysis.py`** - Demo script showing usage
- **`ATTENTION_README.md`** - This documentation

## 🚀 Quick Start

### Basic Usage

```bash
# Run demo with your model
python run_attention_analysis.py llama3_8b_post_brexit_2019_2025 --vignettes 3 --samples 10 --attention-rate 0.2

# Full pipeline with attention (10% of samples)
python attention_pipeline.py your_model --vignettes vignettes/complete_vignettes.json --output results/inference_with_attention.json --attention-dir results/attention_data --attention-rate 0.1
```

### Programmatic Usage

```python
from attention_pipeline import TokenLevelAttentionPipeline

# Create pipeline
pipeline = TokenLevelAttentionPipeline(
    "your_model_name",
    collect_attention=True,
    attention_sample_rate=0.1  # 10% of samples
)

# Run inference with attention
records = []
for vignette in vignettes:
    samples = pipeline.generate_samples([vignette], num_samples=50)
    
    for sample in samples:
        prompt = pipeline._create_prompt(sample['vignette_text'])
        
        # This will automatically collect attention for selected samples
        response = pipeline._run_inference(prompt, {
            'sample_id': len(records),
            'vignette_topic': sample['topic'],
            'vignette_text': sample['vignette_text']
        })
        
        records.append({**sample, 'model_response': response})

# Save attention data
pipeline.save_attention_data("attention_output", "my_analysis")
```

## 📊 Output Format

### Attention Data Structure

```json
{
  "sample_id": 123,
  "vignette_topic": "Firm settlement",
  "input_tokens": [
    ["You", "You"],
    ["▁are", " are"],
    ["▁a", " a"],
    ["▁UK", " UK"],
    ...
  ],
  "template_regions": {
    "prompt_prefix": [0, 1, 2, 3, 4, 5, 6],
    "vignette_content": [7, 8, 9, 10, 11, 12, 13, 14, 15],
    "prompt_suffix": [16, 17, 18, 19, 20]
  },
  "generation_attention": [
    {
      "generation_step": 0,
      "generated_token": "DECISION",
      "layers": [
        {
          "layer_idx": 0,
          "attention_heads": [
            {
              "head_idx": 0,
              "top_attended_tokens": [
                {
                  "token_idx": 12,
                  "token": "▁Christian",
                  "text": " Christian",
                  "attention_score": 0.342
                },
                {
                  "token_idx": 9,
                  "token": "Omar",
                  "text": "Omar",
                  "attention_score": 0.128
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

### CSV Analysis Files

The pipeline automatically generates analysis-ready CSV files:

```csv
sample_id,vignette_topic,generation_step,generated_token,layer_idx,head_idx,attended_token,attended_text,attention_score,token_position
123,Firm settlement,0,DECISION,0,0,▁Christian, Christian,0.342,12
123,Firm settlement,0,DECISION,0,0,Omar,Omar,0.128,9
123,Firm settlement,0,DECISION,16,2,▁Male, Male,0.156,14
```

## 🔬 Analysis Applications

### 1. Demographic Bias Detection
```python
# Analyze attention to demographic fields
df = pd.read_csv("attention_analysis.csv")

# Compare attention to age vs religion vs gender
demographic_attention = df[df['attended_text'].str.contains('Christian|Muslim|Male|Female|12|25|40')]
bias_analysis = demographic_attention.groupby(['vignette_topic', 'attended_text'])['attention_score'].mean()
```

### 2. Template Field Analysis
```python
# See which template fields get most attention during decision-making
field_attention = df.groupby('attended_text')['attention_score'].agg(['mean', 'std', 'count'])
high_attention_fields = field_attention.sort_values('mean', ascending=False).head(20)
```

### 3. Decision Evolution Tracking
```python
# Track how attention changes across generation steps
decision_evolution = df.groupby(['sample_id', 'generation_step'])['attention_score'].mean()
```

## 🛠 Integration with Existing Pipeline

The attention pipeline extends your existing inference system:

- **Compatible with existing vignette system** ✅
- **Same output format + attention data** ✅  
- **Configurable sample rate** (don't collect for all 18k samples) ✅
- **Memory efficient** (stores only key layers/heads) ✅
- **Stratified sampling** (ensures coverage across vignettes) ✅

## 📈 Performance Considerations

### Memory Usage
- **Token-level data**: ~2-5MB per 100 attention samples
- **Configurable storage**: Choose layers/heads to store
- **Efficient sampling**: Default 10% collection rate

### Runtime Impact
- **Minimal overhead**: Attention collection during generation (no extra forward pass)
- **Parallel friendly**: Can be combined with batch inference
- **Checkpoint compatible**: Save/resume attention collection

## 🎨 Visualization Recommendations

Based on your table, here are the recommended tools:

### 1. Quick Inspection - BertViz
```python
from bertviz import head_view
from transformers import AutoTokenizer

# Load your attention data
attention_data = pipeline.attention_data[0]
tokens = [token[1] for token in attention_data['input_tokens']]
attention_matrix = attention_data['generation_attention'][0]['layers'][0]['attention_aggregated']

# Visualize
head_view(attention_matrix, tokens)
```

### 2. Analysis - Custom Heatmaps
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create attention heatmap for template analysis
plt.figure(figsize=(12, 8))
sns.heatmap(attention_matrix, 
           xticklabels=tokens, 
           yticklabels=tokens,
           cmap='Blues')
plt.title("Token Attention Matrix")
plt.show()
```

### 3. Research Output - Interactive Notebooks
Create Jupyter notebooks with widgets for examiner interaction:

```python
import ipywidgets as widgets
from IPython.display import display

# Interactive attention explorer
@widgets.interact
def explore_attention(sample_id=widgets.IntSlider(min=0, max=100, value=0),
                     layer=widgets.IntSlider(min=0, max=31, value=0)):
    # Show attention for selected sample and layer
    show_attention_heatmap(sample_id, layer)
```

## 💡 Tips for Your Brexit Analysis

1. **Focus on demographic tokens**: Look for bias in attention to age, religion, gender, country
2. **Compare pre/post Brexit**: Analyze attention pattern differences across time periods  
3. **Legal reasoning attention**: See if model focuses on legal arguments vs. demographics
4. **Decision consistency**: Track whether similar attention patterns lead to similar decisions

Your pipeline is now ready for sophisticated attention analysis! 🎉 

# Production Attention Pipeline

## Overview
This production attention pipeline is optimized for large-scale inference (25,000+ samples) with efficient token-level attention collection.

## Key Efficiency Features
✅ **Batched Inference**: Processes multiple samples simultaneously (8-16 per batch)  
✅ **GPU Optimization**: Maximizes hardware utilization with proper batching  
✅ **Memory Efficiency**: float16 precision + HDF5 compression  
✅ **Smart Sampling**: Configurable attention collection rate (default 80%)  
✅ **Asynchronous I/O**: Batched writes to reduce storage overhead  

## Performance Estimates
For 25,000 samples:
- **Optimized Pipeline**: ~30-45 minutes (batched processing)
- **Original Pipeline**: ~4-6 hours (individual processing) 
- **Speedup**: 5-10x improvement

Storage:
- ~0.5MB per attention sample
- 20k samples @ 80% = ~8-10GB total storage

## Usage

### Basic Usage (Optimized)
```python
from production_attention_pipeline import ProductionAttentionPipeline

# Create pipeline with attention collection
pipeline = ProductionAttentionPipeline(
    model_subdir="llama3_8b_post_brexit_2019_2025",
    collect_attention=True,
    attention_sample_rate=0.8,  # Collect 80% of samples
    storage_dir="attention_data"
)

# Load vignettes
vignettes = load_vignettes("vignettes/complete_vignettes.json")

# Run OPTIMIZED inference with batching
records = pipeline.generate_inference_records_with_attention_optimized(
    vignettes, 
    batch_size=8  # Adjust based on GPU memory
)
```

### Performance Tuning

**GPU Memory Management:**
```python
# High memory GPU (24GB+): Use larger batches
batch_size = 16

# Medium memory GPU (12-16GB): Use moderate batches  
batch_size = 8

# Lower memory GPU (8GB): Use smaller batches
batch_size = 4
```

**Attention Collection Rate:**
```python
# Maximum coverage (100% - slower)
attention_sample_rate = 1.0

# Balanced (80% - recommended)
attention_sample_rate = 0.8

# Quick analysis (20% - faster)
attention_sample_rate = 0.2
```

## Technical Details

### Batched Processing Architecture
1. **Phase 1**: Generate all prompts in memory (fast)
2. **Phase 2**: Run batched inference with attention collection
3. **Phase 3**: Build final records

### Attention Extraction
- **When**: During generation (not post-hoc)
- **What**: Token-level attention from generated tokens to input tokens
- **Layers**: 3 key layers (10, 20, 30) to reduce storage
- **Steps**: First 3 generation steps (most informative)

### Storage Optimization
- **Format**: HDF5 with gzip compression
- **Precision**: float16 (halves memory usage)
- **Batching**: Writes every 100 samples to reduce I/O
- **Structure**: Hierarchical storage for easy analysis

## Efficiency Analysis

### Memory Usage
- **Inference**: ~6-8GB GPU memory (batch_size=8)
- **Attention**: ~2-4GB additional for collection
- **Storage**: ~0.5MB per sample with compression

### I/O Optimization
- Batched HDF5 writes (100 samples at a time)
- Separate metadata storage (JSON)
- Compressed attention matrices
- Minimal redundant data

### Compute Optimization
- Uses existing model weights (no additional loading)
- `output_attentions=True` during generation (no extra forward pass)
- GPU memory clearing between batches
- Fallback to individual processing on batch errors 