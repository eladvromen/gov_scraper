#!/usr/bin/env python3
"""
Example script for running token-level attention analysis on vignettes.
This demonstrates how to use the TokenLevelAttentionPipeline for template analysis.
"""

import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from attention_pipeline import TokenLevelAttentionPipeline
from utils import load_vignettes


def run_attention_demo(model_name, num_vignettes=2, samples_per_vignette=5, attention_rate=0.5):
    """
    Run a demonstration of token-level attention analysis.
    
    Args:
        model_name (str): Name of the model to use
        num_vignettes (int): Number of vignettes to process
        samples_per_vignette (int): Number of samples per vignette
        attention_rate (float): Fraction of samples to collect attention for
    """
    print(f"🚀 Starting Token-Level Attention Analysis Demo")
    print(f"   Model: {model_name}")
    print(f"   Vignettes: {num_vignettes}")
    print(f"   Samples per vignette: {samples_per_vignette}")
    print(f"   Attention collection rate: {attention_rate:.1%}")
    print()
    
    # Create pipeline
    pipeline = TokenLevelAttentionPipeline(
        model_name,
        collect_attention=True,
        attention_sample_rate=attention_rate
    )
    
    # Load vignettes (use first few for demo)
    vignettes = load_vignettes("../vignettes/complete_vignettes.json")
    demo_vignettes = vignettes[:num_vignettes]
    
    print(f"📚 Loaded vignettes:")
    for i, v in enumerate(demo_vignettes):
        print(f"   {i+1}. {v['topic']} ({v['meta_topic']})")
    print()
    
    # Generate samples and run inference
    records = []
    total_samples = 0
    
    for vignette_idx, vignette in enumerate(demo_vignettes):
        print(f"🔬 Processing vignette {vignette_idx + 1}: {vignette['topic']}")
        
        # Generate limited samples for demo
        sample_records = pipeline.generate_samples([vignette], num_samples=samples_per_vignette)
        
        for sample_idx, record in enumerate(sample_records):
            total_samples += 1
            should_collect = pipeline._should_collect_attention(total_samples, num_vignettes * samples_per_vignette)
            
            if should_collect:
                print(f"   📊 Collecting attention for sample {sample_idx + 1}")
                
                # Create metadata for attention analysis
                sample_metadata = {
                    'sample_id': total_samples,
                    'vignette_topic': record['topic'],
                    'meta_topic': record['meta_topic'],
                    'vignette_text': record['vignette_text'],
                    'fields': record['fields']
                }
                
                # Run inference with attention
                prompt = pipeline._create_prompt(record['vignette_text'])
                response = pipeline._run_inference(prompt, sample_metadata)
                record['model_response'] = response
                record['attention_collected'] = True
            else:
                print(f"   ⚡ Standard inference for sample {sample_idx + 1}")
                
                # Run standard inference
                prompt = pipeline._create_prompt(record['vignette_text'])
                response = pipeline._run_inference(prompt)
                record['model_response'] = response
                record['attention_collected'] = False
            
            # Add inference metadata
            record['inference_timestamp'] = datetime.now().isoformat()
            records.append(record)
    
    print()
    print(f"✅ Inference completed!")
    print(f"   Total samples: {len(records)}")
    print(f"   Attention collected: {len(pipeline.attention_data)}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save inference results
    import json
    results_file = f"results/attention_demo_{model_name}_{timestamp}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved: {results_file}")
    
    # Save attention data if collected
    if pipeline.attention_data:
        attention_dir = f"results/attention_analysis_{timestamp}"
        pipeline.save_attention_data(attention_dir, f"demo_{model_name}")
        
        print(f"🧠 Attention data saved: {attention_dir}/")
        print(f"   - Token-level attention patterns")
        print(f"   - Template region mappings")
        print(f"   - CSV files for analysis")
        
        # Show preview of attention analysis
        if pipeline.attention_data:
            sample_attention = pipeline.attention_data[0]
            print()
            print("📋 Preview of attention analysis:")
            print(f"   Sample: {sample_attention.get('vignette_topic')}")
            print(f"   Input tokens: {len(sample_attention.get('input_tokens', []))}")
            print(f"   Generation steps: {len(sample_attention.get('generation_attention', []))}")
            
            if sample_attention.get('generation_attention'):
                first_step = sample_attention['generation_attention'][0]
                if first_step.get('layers') and first_step['layers'][0].get('attention_heads'):
                    top_tokens = first_step['layers'][0]['attention_heads'][0].get('top_attended_tokens', [])[:3]
                    print(f"   Top attended tokens in first step:")
                    for token_info in top_tokens:
                        print(f"     - '{token_info['text']}' (score: {token_info['attention_score']:.3f})")
    
    print()
    print("🎯 Next steps for your analysis:")
    print("   1. Use bertviz for interactive attention visualization")
    print("   2. Analyze the CSV files for attention patterns")
    print("   3. Compare attention across demographic groups")
    print("   4. Look for bias in template field attention")
    
    return records, pipeline.attention_data


def main():
    """Command line interface for attention demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Token-level attention analysis demo")
    parser.add_argument("model_name", help="Model name (e.g., llama3_8b_post_brexit_2019_2025)")
    parser.add_argument("--vignettes", type=int, default=2, help="Number of vignettes to process")
    parser.add_argument("--samples", type=int, default=5, help="Samples per vignette")
    parser.add_argument("--attention-rate", type=float, default=0.5, help="Attention collection rate")
    
    args = parser.parse_args()
    
    try:
        records, attention_data = run_attention_demo(
            args.model_name,
            args.vignettes,
            args.samples,
            args.attention_rate
        )
        
        print(f"\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 