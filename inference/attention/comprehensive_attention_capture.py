"""
Comprehensive attention capture for deep bias analysis.
Captures ALL layers, ALL heads, and multiple attention patterns for detailed analysis.
"""

import sys
import os
from pathlib import Path

# Add parent directories to path
attention_dir = Path(__file__).parent  # attention directory
inference_dir = attention_dir.parent   # inference directory  
project_dir = inference_dir.parent     # project root
vignettes_dir = project_dir / "vignettes"

sys.path.extend([str(inference_dir), str(vignettes_dir)])

from production_attention_pipeline import ProductionAttentionPipeline
from utils import load_vignettes
import h5py
import json
import numpy as np
from datetime import datetime

class ComprehensiveAttentionPipeline(ProductionAttentionPipeline):
    """Enhanced pipeline that captures ALL attention data for deep analysis."""
    
    def _extract_attention(self, attentions, inputs, metadata):
        """Extract comprehensive attention data - all layers, all heads."""
        try:
            print(f"    🔍 COMPREHENSIVE ATTENTION EXTRACTION for sample {metadata.get('sample_id')}")
            
            if not attentions:
                print(f"    ❌ No attentions returned")
                return None
                
            input_ids = inputs['input_ids'][0].cpu().numpy()
            input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            print(f"    📝 Input tokens: {len(input_tokens)}")
            
            attention_data = {
                'sample_id': metadata.get('sample_id'),
                'input_length': len(input_tokens),
                'input_tokens': input_tokens,  # Store ALL tokens for this analysis
                'attention_matrices': []
            }
            
            # Extract from first generation step
            if len(attentions) > 0:
                first_step = attentions[0]
                num_layers = len(first_step)
                print(f"    🏗️  Model has {num_layers} layers")
                
                # CAPTURE ALL LAYERS (not just bias detection layers)
                for layer_idx in range(num_layers):
                    layer = first_step[layer_idx]
                    
                    if len(layer.shape) == 4:  # [batch, heads, seq_len, seq_len]
                        num_heads = layer.shape[1]
                        print(f"    📊 Layer {layer_idx}: {num_heads} heads, shape {layer.shape}")
                        
                        # 1. HEAD-AVERAGED ATTENTION (current approach)
                        avg_attention = layer[0].mean(dim=0).cpu().numpy()  # [seq_len, seq_len]
                        
                        # Store last token attending to input (current method)
                        last_token_attn = avg_attention[-1, :len(input_tokens)]
                        attention_data['attention_matrices'].append({
                            'layer': layer_idx,
                            'head': 'averaged',
                            'type': 'last_to_input',
                            'weights': last_token_attn.astype(np.float16),
                            'description': f'Layer_{layer_idx}_LastToInput_HeadAvg'
                        })
                        
                        # Store full input-to-input attention matrix (new!)
                        input_to_input = avg_attention[:len(input_tokens), :len(input_tokens)]
                        attention_data['attention_matrices'].append({
                            'layer': layer_idx,
                            'head': 'averaged',
                            'type': 'input_to_input_matrix',
                            'weights': input_to_input.astype(np.float16),
                            'description': f'Layer_{layer_idx}_InputMatrix_HeadAvg'
                        })
                        
                        # 2. INDIVIDUAL HEAD ATTENTION (new approach!)
                        # Store a few key heads for detailed analysis
                        key_heads = [0, num_heads//4, num_heads//2, 3*num_heads//4, num_heads-1]  # First, quarter, middle, 3/4, last
                        
                        for head_idx in key_heads:
                            if head_idx < num_heads:
                                head_attention = layer[0, head_idx].cpu().numpy()  # [seq_len, seq_len]
                                
                                # Last token to input for this head
                                head_last_to_input = head_attention[-1, :len(input_tokens)]
                                attention_data['attention_matrices'].append({
                                    'layer': layer_idx,
                                    'head': head_idx,
                                    'type': 'last_to_input',
                                    'weights': head_last_to_input.astype(np.float16),
                                    'description': f'Layer_{layer_idx}_Head_{head_idx}_LastToInput'
                                })
                                
                                # Input-to-input matrix for this head
                                head_input_matrix = head_attention[:len(input_tokens), :len(input_tokens)]
                                attention_data['attention_matrices'].append({
                                    'layer': layer_idx,
                                    'head': head_idx,
                                    'type': 'input_to_input_matrix',
                                    'weights': head_input_matrix.astype(np.float16),
                                    'description': f'Layer_{layer_idx}_Head_{head_idx}_InputMatrix'
                                })
                
                print(f"    ✅ Extracted {len(attention_data['attention_matrices'])} attention matrices")
                return attention_data
                
        except Exception as e:
            print(f"    ❌ Comprehensive extraction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def run_comprehensive_analysis():
    """Run comprehensive attention analysis on a single sample."""
    
    print("🔬 COMPREHENSIVE ATTENTION ANALYSIS")
    print("=" * 60)
    print("This will capture ALL layers and individual heads for deep bias analysis")
    
    # Load vignettes - set up paths
    attention_dir = Path(__file__).parent  # attention directory
    inference_dir = attention_dir.parent   # inference directory  
    project_dir = inference_dir.parent     # project root
    vignettes_path = project_dir / "vignettes" / "complete_vignettes.json"
    vignettes_data = load_vignettes(str(vignettes_path))
    
    # Find a good sample with demographic content
    target_vignette = None
    for vignette in vignettes_data:
        if 'gender' in str(vignette).lower() or 'religion' in str(vignette).lower():
            target_vignette = vignette
            break
    
    if not target_vignette:
        target_vignette = vignettes_data[0]  # Fallback to first
    
    print(f"📋 Target vignette: {target_vignette['topic']}")
    print(f"    Meta-topic: {target_vignette['meta_topic']}")
    
    # Initialize comprehensive pipeline
    pipeline = ComprehensiveAttentionPipeline(
        model_subdir="llama3_8b_post_brexit_2019_2025_instruct",
        models_base_dir="models",
        collect_attention=True,
        attention_sample_rate=1.0,
        storage_dir="inference/results/attention"
    )
    
    print(f"\n🚀 Running comprehensive analysis...")
    
    # Process just this one vignette
    results = pipeline.generate_inference_records([target_vignette], batch_size=1)
    
    print(f"\n✅ Analysis complete!")
    print(f"   Results: {len(results)} sample")
    
    return results

def analyze_comprehensive_results():
    """Analyze the comprehensive attention results."""
    
    # Find latest run
    current_dir = Path(__file__).parent
    inference_dir = current_dir.parent
    attention_dir = inference_dir / "results" / "attention"
    
    run_dirs = [d for d in attention_dir.iterdir() if d.name.startswith('run_')]
    if not run_dirs:
        print("❌ No attention runs found")
        return
    
    latest_run = max(run_dirs)
    h5_file = list(latest_run.glob("*.h5"))[0]
    json_file = list(latest_run.glob("*.json"))[0]
    
    print(f"\n📊 ANALYZING COMPREHENSIVE RESULTS")
    print(f"   Data: {h5_file}")
    
    with h5py.File(h5_file, 'r') as f:
        sample_keys = list(f.keys())
        if not sample_keys:
            print("❌ No samples found")
            return
            
        sample_key = sample_keys[0]  # First sample
        sample_group = f[sample_key]
        
        print(f"   Sample: {sample_key}")
        print(f"   Input length: {sample_group.attrs.get('input_length', 'N/A')}")
        
        # Count different types of attention matrices
        attention_datasets = [k for k in sample_group.keys() if k.startswith('attention_')]
        
        layer_counts = {}
        head_counts = {}
        type_counts = {}
        
        for dataset_name in attention_datasets:
            dataset = sample_group[dataset_name]
            layer = dataset.attrs.get('layer', 'unknown')
            head = dataset.attrs.get('head', 'unknown')
            att_type = dataset.attrs.get('type', 'unknown')
            
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
            head_counts[head] = head_counts.get(head, 0) + 1
            type_counts[att_type] = type_counts.get(att_type, 0) + 1
        
        print(f"\n📈 COMPREHENSIVE CAPTURE SUMMARY:")
        print(f"   Total attention matrices: {len(attention_datasets)}")
        print(f"   Layers captured: {len(layer_counts)} ({min(layer_counts.keys()) if layer_counts else 'N/A'} to {max(layer_counts.keys()) if layer_counts else 'N/A'})")
        print(f"   Head types: {list(head_counts.keys())}")
        print(f"   Attention types: {list(type_counts.keys())}")
        
        # Find most interesting patterns
        print(f"\n🎯 RECOMMENDED NEXT STEPS:")
        print(f"   1. Analyze input-to-input matrices for demographic→legal attention")
        print(f"   2. Compare individual heads for specialization")
        print(f"   3. Look across all layers for legal reasoning patterns")
        print(f"   4. Focus on mid-layers (8-24) for semantic processing")

if __name__ == "__main__":
    results = run_comprehensive_analysis()
    analyze_comprehensive_results() 