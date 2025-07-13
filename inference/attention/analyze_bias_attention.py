#!/usr/bin/env python3
"""
Analyze attention patterns for demographic bias detection in legal vignettes.
Focus on detecting unfair attention to identity markers (religion, gender, nationality, etc.)
"""

import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import argparse

def find_latest_run(base_dir="inference/results/attention"):
    """Find the latest attention run directory."""
    if not os.path.exists(base_dir):
        return None
    
    run_dirs = [d for d in os.listdir(base_dir) if d.startswith('run_')]
    if not run_dirs:
        return None
    
    run_dirs.sort(reverse=True)
    return os.path.join(base_dir, run_dirs[0])

def get_sample_info(json_file, sample_id):
    """Get sample metadata from JSON file."""
    with open(json_file, 'r') as f:
        metadata = json.load(f)
    
    return metadata['samples'].get(str(sample_id), {})

def clean_token(token):
    """Clean token for display."""
    if token.startswith('Ġ'):
        return token[1:]  # Remove Ġ prefix (space marker)
    elif token.startswith('▁'):
        return token[1:]  # Remove ▁ prefix (space marker for some tokenizers)
    return token

def identify_demographic_tokens(tokens):
    """
    Identify demographic/identity tokens that could indicate bias.
    """
    # Define demographic categories based on common vignette attributes
    demographic_categories = {
        'religion': ['muslim', 'christian', 'hindu', 'atheist', 'buddhist', 'jewish'],
        'gender': ['male', 'female', 'man', 'woman', 'boy', 'girl'],
        'age': ['12-year-old', '18-year-old', '25-year-old', '40-year-old', '70-year-old'],
        'nationality': ['ukraine', 'myanmar', 'syria', 'china', 'pakistan'],
        'economic': ['poor', 'wealthy', 'financial', 'money', 'poverty']
    }
    
    found_tokens = defaultdict(list)
    
    for i, token in enumerate(tokens):
        clean_token_text = clean_token(token).lower()
        
        for category, terms in demographic_categories.items():
            if clean_token_text in terms:
                found_tokens[category].append({
                    'position': i,
                    'token': token,
                    'clean_text': clean_token_text,
                    'category': category
                })
    
    return found_tokens

def analyze_sample_bias(h5_file, json_file, sample_id=1):
    """Analyze attention patterns for demographic bias in a specific sample."""
    print(f"🔍 Analyzing demographic attention for Sample {sample_id}")
    
    # Load sample metadata
    with open(json_file, 'r') as f:
        metadata = json.load(f)
    
    sample_info = metadata['samples'].get(str(sample_id), {})
    topic = sample_info.get('topic', f'Sample {sample_id}')
    
    # Load attention data from HDF5
    with h5py.File(h5_file, 'r') as f:
        sample_key = f'sample_{sample_id}'
        if sample_key not in f:
            print(f"❌ Sample {sample_id} not found")
            return None
        
        sample_group = f[sample_key]
        
        # Get tokens
        if 'input_tokens' in sample_group:
            tokens = sample_group['input_tokens'][:]
            if isinstance(tokens[0], bytes):
                tokens = [t.decode('utf-8') for t in tokens]
        else:
            print("❌ No input tokens found")
            return None
        
        # Get attention data
        attention_data = []
        attention_datasets = [k for k in sample_group.keys() if k.startswith('attention_')]
        
        for dataset_name in attention_datasets:
            dataset = sample_group[dataset_name]
            layer = dataset.attrs.get('layer', 'unknown')
            layer_name = dataset.attrs.get('layer_name', f'Layer_{layer}')
            attention_weights = dataset[:]
            
            attention_data.append({
                'layer': layer,
                'layer_name': layer_name,
                'weights': attention_weights
            })
        
        attention_data.sort(key=lambda x: x['layer'] if isinstance(x['layer'], int) else 999)
    
    # Identify demographic tokens
    clean_tokens = [clean_token(token) for token in tokens]
    demographic_tokens = identify_demographic_tokens(clean_tokens)
    
    print(f"Topic: {topic}")
    print(f"Found demographic tokens:")
    for category, token_list in demographic_tokens.items():
        if token_list:
            print(f"  {category}: {[t['clean_text'] for t in token_list]}")
    
    # Analyze bias across layers
    bias_results = {
        'sample_id': sample_id,
        'topic': topic,
        'demographic_tokens': demographic_tokens,
        'layers': []
    }
    
    print(f"\n🎯 BIAS ANALYSIS:")
    for layer_data in attention_data:
        layer = layer_data['layer']
        weights = layer_data['weights'][:len(clean_tokens)]  # Match lengths
        
        print(f"\nLayer {layer}:")
        
        # Calculate attention to demographic categories
        for category, token_list in demographic_tokens.items():
            if token_list:
                category_attention = []
                for token_info in token_list:
                    pos = token_info['position']
                    if pos < len(weights):
                        attention_val = weights[pos]
                        category_attention.append(attention_val)
                        print(f"  {category} '{token_info['clean_text']}': {attention_val:.4f}")
                
                if category_attention:
                    avg_attention = np.mean(category_attention)
                    if avg_attention > 0.02:  # Flag high attention
                        print(f"  🚨 HIGH {category.upper()} ATTENTION: {avg_attention:.4f}")
    
    return bias_results

def create_bias_visualization(bias_analysis, save_path=None):
    """Create visualization of demographic bias across layers."""
    
    if not bias_analysis:
        print("❌ No bias analysis data to visualize")
        return
    
    # Extract data for plotting
    layers = []
    layer_names = []
    demographic_data = defaultdict(list)
    baseline_data = []
    
    for layer_info in bias_analysis['layer_analysis']:
        layers.append(layer_info['layer'])
        layer_names.append(layer_info['layer_name'])
        baseline_data.append(layer_info['baseline_attention'])
        
        # Get attention for each demographic category
        for category in ['religion', 'gender', 'nationality', 'economic']:
            if category in layer_info['demographic_attention']:
                demographic_data[category].append(layer_info['demographic_attention'][category]['avg_attention'])
            else:
                demographic_data[category].append(0.0)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    topic = bias_analysis['topic']
    meta_topic = bias_analysis['meta_topic']
    fig.suptitle(f'Demographic Bias Analysis: {topic}\n({meta_topic})', fontsize=14, fontweight='bold')
    
    # Plot 1: Attention to demographic categories by layer
    x_pos = np.arange(len(layers))
    width = 0.15
    
    colors = ['red', 'blue', 'green', 'orange']
    for i, (category, values) in enumerate(demographic_data.items()):
        if any(v > 0 for v in values):  # Only plot if there's data
            offset = (i - 1.5) * width
            ax1.bar(x_pos + offset, values, width, label=category.title(), color=colors[i], alpha=0.7)
    
    ax1.plot(x_pos, baseline_data, 'k--', label='Baseline (non-demographic)', linewidth=2)
    ax1.axhline(y=0.05, color='red', linestyle=':', alpha=0.7, label='Bias threshold (5%)')
    
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Average Attention Weight')
    ax1.set_title('Attention to Demographic Categories by Layer')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'L{l}' for l in layers])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Bias ratio (demographic attention / baseline attention)
    bias_ratios = defaultdict(list)
    for layer_info in bias_analysis['layer_analysis']:
        baseline = layer_info['baseline_attention']
        for category in ['religion', 'gender', 'nationality', 'economic']:
            if category in layer_info['demographic_attention'] and baseline > 0:
                demo_attention = layer_info['demographic_attention'][category]['avg_attention']
                ratio = demo_attention / baseline
                bias_ratios[category].append(ratio)
            else:
                bias_ratios[category].append(1.0)  # No bias = ratio of 1
    
    for i, (category, ratios) in enumerate(bias_ratios.items()):
        if any(r != 1.0 for r in ratios):
            offset = (i - 1.5) * width
            ax2.bar(x_pos + offset, ratios, width, label=category.title(), color=colors[i], alpha=0.7)
    
    ax2.axhline(y=1.0, color='black', linestyle='-', alpha=0.5, label='No bias (ratio = 1)')
    ax2.axhline(y=2.0, color='red', linestyle=':', alpha=0.7, label='2x bias threshold')
    
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Attention Ratio (Demographic / Baseline)')
    ax2.set_title('Bias Ratio: Demographic vs. Baseline Attention')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'L{l}' for l in layers])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add bias flags as text
    if bias_analysis['bias_flags']:
        bias_text = "🚨 BIAS FLAGS:\n" + "\n".join(bias_analysis['bias_flags'][:3])  # Show first 3
        ax2.text(0.02, 0.98, bias_text, transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8), fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Bias analysis plot saved to: {save_path}")
    
    plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze attention patterns for demographic bias detection")
    parser.add_argument("--sample", type=int, default=1, help="Sample ID to analyze (default: 1)")
    parser.add_argument("--run-dir", type=str, help="Specific run directory to use")
    
    args = parser.parse_args()
    
    print("🔍 DEMOGRAPHIC BIAS ANALYSIS")
    print("=" * 40)
    
    # Find latest run
    run_dir = args.run_dir
    if not run_dir:
        run_dir = find_latest_run()
    
    if not run_dir or not os.path.exists(run_dir):
        print("❌ No attention runs found!")
        return
    
    print(f"📁 Using: {run_dir}")
    
    # Find files
    h5_file = None
    json_file = None
    for file in os.listdir(run_dir):
        if file.endswith('.h5'):
            h5_file = os.path.join(run_dir, file)
        elif file.endswith('.json'):
            json_file = os.path.join(run_dir, file)
    
    if not h5_file or not json_file:
        print("❌ Missing files")
        return
    
    # Analyze first sample
    analyze_sample_bias(h5_file, json_file, sample_id=args.sample)

if __name__ == "__main__":
    main() 