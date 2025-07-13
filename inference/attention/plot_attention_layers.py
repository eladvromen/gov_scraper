#!/usr/bin/env python3
"""
Compare attention patterns across multiple layers to find meaningful legal reasoning.
"""

import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def find_latest_run():
    """Find the latest attention run directory."""
    # Navigate from current location (inference/attention/) to results/attention/
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

def compare_layers_attention(h5_file, json_file, sample_id=1, save_path=None):
    """
    Compare attention patterns across all available layers for a sample.
    """
    
    # Get sample info
    sample_info = get_sample_info(json_file, sample_id)
    topic = sample_info.get('topic', f'Sample {sample_id}')
    meta_topic = sample_info.get('meta_topic', 'Unknown')
    
    print(f"📊 Comparing attention across layers for Sample {sample_id}")
    print(f"   Topic: {topic}")
    print(f"   Meta-topic: {meta_topic}")
    
    # Load data from HDF5
    with h5py.File(h5_file, 'r') as f:
        sample_key = f'sample_{sample_id}'
        if sample_key not in f:
            print(f"❌ Sample {sample_id} not found in HDF5 file")
            return
        
        sample_group = f[sample_key]
        
        # Get input tokens
        if 'input_tokens' in sample_group:
            tokens = sample_group['input_tokens'][:]
            if isinstance(tokens[0], bytes):
                tokens = [t.decode('utf-8') for t in tokens]
        else:
            print("❌ No input tokens found")
            return
        
        # Get all attention datasets
        attention_datasets = [k for k in sample_group.keys() if k.startswith('attention_')]
        attention_data = []
        
        for dataset_name in attention_datasets:
            dataset = sample_group[dataset_name]
            layer = dataset.attrs.get('layer', 'unknown')
            layer_name = dataset.attrs.get('layer_name', f'Layer_{layer}')
            attention_weights = dataset[:]
            
            attention_data.append({
                'layer': layer,
                'layer_name': layer_name,
                'weights': attention_weights,
                'dataset_name': dataset_name
            })
        
        # Sort by layer number
        attention_data.sort(key=lambda x: x['layer'] if isinstance(x['layer'], int) else 999)
    
    # Clean tokens for display
    display_tokens = [clean_token(token) for token in tokens]
    
    # Create comparison plot
    num_layers = len(attention_data)
    fig, axes = plt.subplots(num_layers, 1, figsize=(16, 4*num_layers))
    if num_layers == 1:
        axes = [axes]
    
    fig.suptitle(f'Layer Comparison: {topic}\n({meta_topic})', fontsize=16, fontweight='bold')
    
    # Track statistics for comparison
    layer_stats = []
    
    for i, (ax, layer_data) in enumerate(zip(axes, attention_data)):
        weights = layer_data['weights']
        layer_name = layer_data['layer_name']
        
        # Handle length mismatch
        min_length = min(len(display_tokens), len(weights))
        plot_tokens = display_tokens[:min_length]
        plot_weights = weights[:min_length]
        
        # Create bar plot
        positions = np.arange(len(plot_weights))
        bars = ax.bar(positions, plot_weights, alpha=0.7, color='skyblue', edgecolor='navy', linewidth=0.5)
        
        # Highlight top attention positions
        top_indices = np.argsort(plot_weights)[-5:]  # Top 5
        for idx in top_indices:
            bars[idx].set_color('red')
            bars[idx].set_alpha(0.8)
        
        ax.set_title(f'{layer_name} (Max: {plot_weights.max():.4f}, Mean: {plot_weights.mean():.4f})')
        ax.set_ylabel('Attention')
        ax.grid(True, alpha=0.3)
        
        # Calculate statistics
        entropy = -np.sum(plot_weights * np.log(plot_weights + 1e-10))
        max_attention = plot_weights.max()
        top_5_positions = np.argsort(plot_weights)[-5:][::-1]
        
        layer_stats.append({
            'layer_name': layer_name,
            'entropy': entropy,
            'max_attention': max_attention,
            'top_positions': top_5_positions,
            'top_tokens': [plot_tokens[pos] for pos in top_5_positions[:3]]  # Top 3 tokens
        })
        
        # Add token labels for top positions (every 10th position + top 3)
        label_positions = list(range(0, len(plot_tokens), 10)) + list(top_5_positions[:3])
        label_positions = sorted(set(label_positions))
        
        for pos in label_positions:
            if pos < len(plot_tokens):
                token = plot_tokens[pos][:8] + '...' if len(plot_tokens[pos]) > 8 else plot_tokens[pos]
                ax.text(pos, plot_weights[pos] + max_attention * 0.05, token, 
                       rotation=45, ha='left', va='bottom', fontsize=8)
    
    # Add final x-axis label
    axes[-1].set_xlabel('Token Position')
    
    plt.tight_layout()
    
    # Print layer analysis
    print("\n🔍 LAYER ANALYSIS SUMMARY:")
    print("-" * 60)
    
    for stats in layer_stats:
        print(f"{stats['layer_name']}:")
        print(f"  Entropy: {stats['entropy']:.3f} (higher = more distributed)")
        print(f"  Max attention: {stats['max_attention']:.4f}")
        print(f"  Top 3 tokens: {stats['top_tokens']}")
        print()
    
    # Find the most interesting layer (balanced entropy and reasonable max attention)
    interesting_layers = []
    for stats in layer_stats:
        # Look for layers with reasonable distribution (entropy > 2) but not too scattered
        if 2.0 < stats['entropy'] < 4.5 and stats['max_attention'] < 0.5:
            interesting_layers.append(stats)
    
    if interesting_layers:
        print("🎯 MOST PROMISING LAYERS FOR LEGAL REASONING:")
        for stats in interesting_layers:
            print(f"  {stats['layer_name']}: Entropy {stats['entropy']:.3f}, Max {stats['max_attention']:.4f}")
    else:
        print("⚠️  No layers show ideal attention distribution patterns")
    
    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Layer comparison saved to: {save_path}")
    
    plt.show()
    plt.close()

def main():
    print("🔬 MULTI-LAYER ATTENTION ANALYSIS")
    print("=" * 50)
    
    # Find latest run
    run_dir = find_latest_run()
    if not run_dir or not os.path.exists(run_dir):
        print("❌ No attention run directory found!")
        return
    
    print(f"📁 Using run: {run_dir}")
    
    # Find files
    h5_file = None
    json_file = None
    
    for file in os.listdir(run_dir):
        if file.endswith('.h5'):
            h5_file = os.path.join(run_dir, file)
        elif file.endswith('.json'):
            json_file = os.path.join(run_dir, file)
    
    if not h5_file or not json_file:
        print("❌ Could not find both .h5 and .json files")
        return
    
    # Create save path
    current_dir = Path(__file__).parent  # attention directory
    plots_dir = current_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    save_path = plots_dir / "attention_layers_comparison.png"
    
    # Analyze
    compare_layers_attention(h5_file, json_file, sample_id=1, save_path=save_path)
    
    print("✅ Multi-layer analysis complete!")

if __name__ == "__main__":
    main() 