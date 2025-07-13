"""
Quick analysis of attention patterns across multiple samples to identify bias patterns.
Focus on demographic token attention vs. format token attention.
"""

import h5py
import json
import numpy as np
import os
from pathlib import Path

def find_latest_run():
    """Find the latest attention run directory."""
    current_dir = Path(__file__).parent  # attention directory
    inference_dir = current_dir.parent   # inference directory  
    base_dir = inference_dir / "results" / "attention"
    
    if not base_dir.exists():
        return None
    
    run_dirs = [d for d in os.listdir(base_dir) if d.startswith('run_')]
    if not run_dirs:
        return None
    
    run_dirs.sort(reverse=True)
    return str(base_dir / run_dirs[0])

def clean_token(token):
    """Clean token for display."""
    if token.startswith('Ġ'):
        return token[1:]  # Remove Ġ prefix (space marker)
    elif token.startswith('▁'):
        return token[1:]  # Remove ▁ prefix (space marker for some tokenizers)
    return token

def analyze_demographic_attention(h5_file, json_file, num_samples=5):
    """
    Analyze attention to demographic vs. legal content across multiple samples.
    """
    
    # Load metadata
    with open(json_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"🔍 QUICK BIAS ANALYSIS - {num_samples} Samples")
    print("=" * 60)
    
    # Demographic terms to look for
    demographic_terms = [
        'muslim', 'christian', 'islam', 'religion', 'religious',
        'male', 'female', 'man', 'woman', 'gender',
        'ukraine', 'syria', 'afghanistan', 'myanmar', 'country', 'national',
        'poor', 'wealthy', 'money', 'economic', 'financial'
    ]
    
    format_tokens = ['<|begin_of_text|>', '<|end_of_text|>', ':', '.', ',', 'Ċ', 'ĊĊ']
    
    with h5py.File(h5_file, 'r') as f:
        sample_keys = [k for k in f.keys() if k.startswith('sample_')][:num_samples]
        
        layer_summaries = {18: [], 22: [], 25: [], 28: []}
        
        for sample_key in sample_keys:
            sample_id = sample_key.replace('sample_', '')
            sample_group = f[sample_key]
            
            # Get sample info
            sample_info = metadata['samples'].get(sample_id, {})
            topic = sample_info.get('topic', f'Sample {sample_id}')
            
            print(f"\n📊 Sample {sample_id}: {topic}")
            
            # Get input tokens
            if 'input_tokens' in sample_group:
                tokens = sample_group['input_tokens'][:]
                if isinstance(tokens[0], bytes):
                    tokens = [t.decode('utf-8') for t in tokens]
                clean_tokens = [clean_token(t).lower() for t in tokens]
            else:
                continue
            
            # Analyze each attention layer
            attention_datasets = [k for k in sample_group.keys() if k.startswith('attention_')]
            
            for dataset_name in attention_datasets:
                dataset = sample_group[dataset_name]
                layer = dataset.attrs.get('layer', 'unknown')
                if layer not in [18, 22, 25, 28]:
                    continue
                    
                weights = dataset[:]
                min_length = min(len(clean_tokens), len(weights))
                
                # Calculate attention to different categories
                format_attention = 0
                demographic_attention = 0
                legal_attention = 0
                
                for i in range(min_length):
                    token = clean_tokens[i]
                    weight = weights[i]
                    
                    if any(fmt in tokens[i] for fmt in format_tokens):
                        format_attention += weight
                    elif any(demo in token for demo in demographic_terms):
                        demographic_attention += weight
                    elif any(legal in token for legal in ['asylum', 'persecution', 'refugee', 'protection', 'court', 'judge', 'law', 'legal']):
                        legal_attention += weight
                
                total_attention = format_attention + demographic_attention + legal_attention
                
                # Store results
                layer_summaries[layer].append({
                    'sample_id': sample_id,
                    'topic': topic,
                    'format_pct': (format_attention / max(total_attention, 1e-6)) * 100,
                    'demographic_pct': (demographic_attention / max(total_attention, 1e-6)) * 100,
                    'legal_pct': (legal_attention / max(total_attention, 1e-6)) * 100,
                    'max_attention': weights.max(),
                    'entropy': -np.sum(weights * np.log(weights + 1e-10))
                })
                
                print(f"  Layer {layer}: Format {format_attention:.3f} | Demo {demographic_attention:.3f} | Legal {legal_attention:.3f}")
    
    # Summary across layers and samples
    print(f"\n📈 BIAS DETECTION SUMMARY")
    print("=" * 60)
    
    for layer in [18, 22, 25, 28]:
        if not layer_summaries[layer]:
            continue
            
        results = layer_summaries[layer]
        avg_format = np.mean([r['format_pct'] for r in results])
        avg_demo = np.mean([r['demographic_pct'] for r in results])
        avg_legal = np.mean([r['legal_pct'] for r in results])
        avg_entropy = np.mean([r['entropy'] for r in results])
        
        print(f"\n🎯 Layer {layer} (Bias Detection):")
        print(f"   Format tokens:     {avg_format:.1f}% attention")
        print(f"   Demographic terms: {avg_demo:.1f}% attention")
        print(f"   Legal terms:       {avg_legal:.1f}% attention")
        print(f"   Entropy (avg):     {avg_entropy:.3f}")
        
        # Flag potential bias
        if avg_demo > 10:
            print(f"   🚨 HIGH demographic attention - potential bias!")
        elif avg_demo > 5:
            print(f"   ⚠️  Moderate demographic attention")
        else:
            print(f"   ✅ Low demographic attention")
    
    # Overall assessment
    print(f"\n🔍 OVERALL ASSESSMENT:")
    all_format = np.mean([r['format_pct'] for results in layer_summaries.values() for r in results])
    all_demo = np.mean([r['demographic_pct'] for results in layer_summaries.values() for r in results])
    all_legal = np.mean([r['legal_pct'] for results in layer_summaries.values() for r in results])
    
    print(f"   Average format attention:     {all_format:.1f}%")
    print(f"   Average demographic attention: {all_demo:.1f}%")
    print(f"   Average legal attention:       {all_legal:.1f}%")
    
    if all_format > 60:
        print(f"   🤔 Model heavily focused on format tokens - may not be doing deep reasoning")
    if all_demo > 5:
        print(f"   ⚠️  Significant demographic attention detected")
    if all_legal < 10:
        print(f"   📚 Low attention to legal terms - may need different layers for legal reasoning")

def main():
    run_dir = find_latest_run()
    if not run_dir:
        print("❌ No attention run directory found!")
        return
    
    # Find files
    h5_file = None
    json_file = None
    
    for file in Path(run_dir).iterdir():
        if file.suffix == '.h5':
            h5_file = str(file)
        elif file.suffix == '.json':
            json_file = str(file)
    
    if not h5_file or not json_file:
        print("❌ Could not find both .h5 and .json files")
        return
    
    analyze_demographic_attention(h5_file, json_file, num_samples=5)

if __name__ == "__main__":
    main() 