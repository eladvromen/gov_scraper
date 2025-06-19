#!/usr/bin/env python3
"""
Run inference on a specific vignette by name for testing purposes
"""

import argparse
import json
import sys
import os
from datetime import datetime

# Add inference directory to path
sys.path.insert(0, os.path.dirname(__file__))

from inference_pipeline import InferencePipeline
from utils import load_vignettes

def main():
    parser = argparse.ArgumentParser(description='Run inference on a specific vignette')
    parser.add_argument('model_name', help='Name of the model to use')
    parser.add_argument('vignette_name', help='Name of the vignette to test')
    parser.add_argument('--samples', type=int, default=3, help='Number of random samples to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--dry-run', action='store_true', help='Show generated samples without running inference')
    parser.add_argument('--list-vignettes', action='store_true', help='List all available vignette names')
    
    args = parser.parse_args()
    
    # Load vignettes
    vignettes = load_vignettes("vignettes/complete_vignettes.json")
    
    # List vignettes if requested
    if args.list_vignettes:
        print("Available vignettes:")
        for i, vignette in enumerate(vignettes):
            print(f"{i+1:2d}. {vignette['topic']}")
        return
    
    # Find the specific vignette
    target_vignette = None
    for vignette in vignettes:
        if vignette['topic'] == args.vignette_name:
            target_vignette = vignette
            break
    
    if not target_vignette:
        print(f"Error: Vignette '{args.vignette_name}' not found.")
        print("Use --list-vignettes to see available vignettes.")
        sys.exit(1)
    
    print(f"Testing vignette: {args.vignette_name}")
    print(f"Template: {target_vignette['vignette_template'][:100]}...")
    
    # Initialize pipeline
    pipeline = InferencePipeline(args.model_name)
    
    # Generate samples for this specific vignette
    samples = pipeline.generate_samples([target_vignette], num_samples=args.samples, random_seed=args.seed)
    
    print(f"\nGenerated {len(samples)} samples:")
    
    if args.dry_run:
        # Just show the generated samples
        for i, sample in enumerate(samples, 1):
            print(f"\n--- Sample {i} ---")
            print(f"Fields: {sample['fields']}")
            print(f"Text: {sample['vignette_text']}")
    else:
        # Run actual inference
        print("Running inference...")
        results = []
        
        for i, sample in enumerate(samples, 1):
            print(f"Processing sample {i}/{len(samples)}")
            
            response = pipeline.run_inference(sample['vignette_text'])
            
            result = {
                'meta_topic': sample['meta_topic'],
                'topic': sample['topic'], 
                'fields': sample['fields'],
                'vignette_text': sample['vignette_text'],
                'model_response': response,
                'prompt_used': pipeline.get_last_prompt(),
                'inference_timestamp': datetime.now().isoformat(),
                'sampling_info': {
                    'num_samples_requested': args.samples,
                    'random_seed': args.seed
                }
            }
            results.append(result)
            
            print(f"Response: {response[:100]}...")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        vignette_safe_name = args.vignette_name.replace(' ', '_').replace('–', '-').replace(':', '')
        output_file = f"inference/results/specific_vignette_{vignette_safe_name}_{args.model_name}_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")
        
        # Show brief summary
        print(f"\n--- Results Summary ---")
        for i, result in enumerate(results, 1):
            response_start = result['model_response'].split('\n')[0][:50]
            print(f"Sample {i}: {response_start}...")

if __name__ == "__main__":
    main() 