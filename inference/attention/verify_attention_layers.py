"""
Verify which layers are captured in attention data files.
Quick check to ensure we're collecting the intended bias detection layers.
"""

import h5py
import json
import os
from pathlib import Path

def check_attention_layers():
    """Check what layers are stored in the latest attention data."""
    
    # Find latest attention data - navigate from current location
    current_dir = Path(__file__).parent  # attention directory
    inference_dir = current_dir.parent   # inference directory
    attention_dir = inference_dir / "results" / "attention"
    latest_run = max(attention_dir.glob("run_*"))
    
    h5_file = list(latest_run.glob("attention_*.h5"))[0]
    metadata_file = list(latest_run.glob("metadata_*.json"))[0]
    
    print(f"Checking: {h5_file}")
    print(f"Metadata: {metadata_file}")
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"\nModel: {metadata['model_name']}")
    print(f"Samples collected: {metadata['total_samples_collected']}")
    print(f"Sample rate: {metadata['sample_rate']}")
    
    # Check HDF5 contents
    layer_counts = {}
    sample_layers = {}
    
    with h5py.File(h5_file, 'r') as f:
        print(f"\nSample groups in HDF5: {len(list(f.keys()))}")
        
        # Check first few samples
        for sample_name in sorted(list(f.keys()))[:5]:
            sample_group = f[sample_name]
            sample_id = sample_name.replace('sample_', '')
            
            print(f"\n=== Sample {sample_id} ===")
            print(f"  Input length: {sample_group.attrs.get('input_length', 'N/A')}")
            
            # Find attention datasets
            attention_datasets = [key for key in sample_group.keys() if key.startswith('attention_')]
            sample_layers[sample_id] = []
            
            for att_key in sorted(attention_datasets):
                dataset = sample_group[att_key]
                layer = dataset.attrs.get('layer', 'Unknown')
                layer_name = dataset.attrs.get('layer_name', 'Unknown')
                att_type = dataset.attrs.get('type', 'Unknown')
                
                print(f"  {att_key}: Layer {layer} ({layer_name}) - {dataset.shape} - Type: {att_type}")
                
                sample_layers[sample_id].append(layer)
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
    
    # Summary
    print(f"\n=== LAYER SUMMARY ===")
    print(f"Layers captured across all samples:")
    for layer in sorted(layer_counts.keys()):
        print(f"  Layer {layer}: {layer_counts[layer]} samples")
    
    # Check if we got the intended bias detection layers
    intended_layers = [18, 22, 25, 28]
    print(f"\n=== BIAS DETECTION LAYERS CHECK ===")
    print(f"Intended layers: {intended_layers}")
    
    captured_layers = list(layer_counts.keys())
    print(f"Captured layers: {sorted(captured_layers)}")
    
    missing = set(intended_layers) - set(captured_layers)
    extra = set(captured_layers) - set(intended_layers)
    
    if missing:
        print(f"❌ MISSING layers: {sorted(missing)}")
    if extra:
        print(f"ℹ️  EXTRA layers: {sorted(extra)}")
    
    if set(intended_layers) == set(captured_layers):
        print("✅ SUCCESS: All intended bias detection layers are captured!")
    elif set(intended_layers).issubset(set(captured_layers)):
        print("✅ SUCCESS: All intended layers captured (plus extras)")
    else:
        print("❌ PROBLEM: Some intended layers are missing")
    
    return captured_layers, metadata

if __name__ == "__main__":
    check_attention_layers() 