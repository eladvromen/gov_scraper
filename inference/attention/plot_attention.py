#!/usr/bin/env python3
"""
Plot attention patterns from collected attention data.
"""

import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import argparse
from pathlib import Path

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

def plot_attention_pattern(h5_file, json_file, sample_id=1, layer=0, save_path=None, show_plot=True):
    """
    Plot attention pattern for a specific sample and layer.
    
    Args:
        h5_file (str): Path to HDF5 file
        json_file (str): Path to JSON metadata file
        sample_id (int): Sample ID to plot
        layer (int): Layer to plot (0 for first layer, -1 for last layer)
        save_path (str): Path to save the plot (optional)
        show_plot (bool): Whether to display the plot
    """
    
    # Get sample info
    sample_info = get_sample_info(json_file, sample_id)
    topic = sample_info.get('topic', f'Sample {sample_id}')
    meta_topic = sample_info.get('meta_topic', 'Unknown')
    
    print(f"📊 Plotting attention for Sample {sample_id}")
    print(f"   Topic: {topic}")
    print(f"   Meta-topic: {meta_topic}")
    print(f"   Layer: {layer}")
    
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
        
        # Find the right attention matrix
        attention_datasets = [k for k in sample_group.keys() if k.startswith('attention_')]
        target_dataset = None
        
        for dataset_name in attention_datasets:
            dataset = sample_group[dataset_name]
            if dataset.attrs.get('layer') == layer:
                target_dataset = dataset
                break
        
        if target_dataset is None:
            print(f"❌ No attention data found for layer {layer}")
            print(f"   Available layers: {[sample_group[d].attrs.get('layer') for d in attention_datasets]}")
            return
        
        attention_weights = target_dataset[:]
    
    # Clean tokens for display
    display_tokens = [clean_token(token) for token in tokens]
    
    # Handle length mismatch between tokens and attention weights
    min_length = min(len(display_tokens), len(attention_weights))
    display_tokens = display_tokens[:min_length]
    attention_weights = attention_weights[:min_length]
    
    print(f"   Tokens: {len(display_tokens)}, Attention weights: {len(attention_weights)}")
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(f'Attention Pattern: {topic}\n({meta_topic}) - Layer {layer}', fontsize=14, fontweight='bold')
    
    # Plot 1: Bar chart of attention weights
    positions = np.arange(len(attention_weights))
    bars = ax1.bar(positions, attention_weights, alpha=0.7, color='skyblue', edgecolor='navy', linewidth=0.5)
    
    # Highlight top attention positions
    top_indices = np.argsort(attention_weights)[-10:]  # Top 10
    for idx in top_indices:
        bars[idx].set_color('red')
        bars[idx].set_alpha(0.8)
    
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Attention Weight')
    ax1.set_title('Attention Weights Across All Tokens')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f'Max: {attention_weights.max():.4f} | Mean: {attention_weights.mean():.4f} | Std: {attention_weights.std():.4f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Token visualization with attention intensity
    ax2.set_xlim(0, len(display_tokens))
    ax2.set_ylim(-1, 1)
    ax2.set_title('Tokens with Attention Intensity (Red = High Attention)')
    
    # Show tokens with attention as background color intensity
    max_attention = attention_weights.max()
    
    for i, (token, weight) in enumerate(zip(display_tokens, attention_weights)):
        # Color intensity based on attention weight
        intensity = weight / max_attention if max_attention > 0 else 0
        color = plt.cm.Reds(intensity)
        
        # Add rectangle background
        rect = patches.Rectangle((i-0.4, -0.4), 0.8, 0.8, 
                               linewidth=1, edgecolor='black', facecolor=color, alpha=0.7)
        ax2.add_patch(rect)
        
        # Add token text
        display_token = token[:8] + '...' if len(token) > 8 else token  # Truncate long tokens
        ax2.text(i, 0, display_token, ha='center', va='center', 
                rotation=45 if len(display_token) > 3 else 0, fontsize=8, fontweight='bold')
    
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=max_attention))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, orientation='horizontal', pad=0.1, shrink=0.8)
    cbar.set_label('Attention Weight')
    
    # Show top attended tokens
    top_5_indices = np.argsort(attention_weights)[-5:][::-1]
    top_tokens_text = "Top 5 Attended Tokens:\n"
    for i, idx in enumerate(top_5_indices):
        if idx < len(display_tokens):  # Safety check
            token = display_tokens[idx][:15] + '...' if len(display_tokens[idx]) > 15 else display_tokens[idx]
            top_tokens_text += f"{i+1}. '{token}' (pos {idx}): {attention_weights[idx]:.4f}\n"
    
    ax2.text(1.02, 0.5, top_tokens_text, transform=ax2.transAxes, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontsize=9)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot attention patterns from collected data")
    parser.add_argument("--sample", type=int, default=1, help="Sample ID to plot (default: 1)")
    parser.add_argument("--layer", type=int, default=0, help="Layer to plot: 0 for first, -1 for last (default: 0)")
    parser.add_argument("--save", type=str, help="Path to save the plot")
    parser.add_argument("--no-show", action="store_true", help="Don't display the plot")
    parser.add_argument("--run-dir", type=str, help="Specific run directory to use")
    
    args = parser.parse_args()
    
    print("🎨 ATTENTION VISUALIZATION SCRIPT")
    print("=" * 40)
    
    # Find run directory
    if args.run_dir:
        run_dir = args.run_dir
    else:
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
    
    # Generate save path if not provided
    save_path = args.save
    if not save_path and not args.no_show:
        os.makedirs("inference/plots", exist_ok=True)
        save_path = f"inference/plots/attention_sample_{args.sample}_layer_{args.layer}.png"
    
    # Plot
    plot_attention_pattern(
        h5_file=h5_file,
        json_file=json_file,
        sample_id=args.sample,
        layer=args.layer,
        save_path=save_path,
        show_plot=not args.no_show
    )
    
    print("✅ Visualization complete!")

if __name__ == "__main__":
    main() 