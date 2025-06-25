#!/usr/bin/env python3
"""
Run inference on a custom vignette text provided as input
"""

import argparse
import json
import sys
import os
from datetime import datetime

# Add inference directory to path  
inference_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, inference_dir)

from inference_pipeline import InferencePipeline

def main():
    parser = argparse.ArgumentParser(description='Run inference on custom vignette text')
    parser.add_argument('model_name', help='Name of the model to use')
    parser.add_argument('vignette_text', help='The vignette text to run inference on')
    parser.add_argument('--output', help='Output file path (optional)')
    parser.add_argument('--show-prompt', action='store_true', help='Show the full prompt sent to the model')
    
    args = parser.parse_args()
    
    print(f"Model: {args.model_name}")
    print(f"Vignette text: {args.vignette_text}")
    print()
    
    # Initialize pipeline
    try:
        pipeline = InferencePipeline(args.model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Run inference
    print("Running inference...")
    try:
        response = pipeline.run_inference(args.vignette_text)
        
        # Show results
        print("=" * 60)
        print("VIGNETTE TEXT:")
        print(args.vignette_text)
        print()
        print("MODEL RESPONSE:")
        print(response)
        print("=" * 60)
        
        if args.show_prompt:
            print("\nFULL PROMPT SENT TO MODEL:")
            print("-" * 40)
            print(pipeline.get_last_prompt())
            print("-" * 40)
        
        # Prepare result record
        result = {
            'vignette_text': args.vignette_text,
            'model_response': response,
            'prompt_used': pipeline.get_last_prompt(),
            'model_name': args.model_name,
            'inference_timestamp': datetime.now().isoformat()
        }
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nResult saved to: {args.output}")
        else:
            # Auto-save with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"inference/results/custom_vignette_{args.model_name}_{timestamp}.json"
            
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nResult auto-saved to: {output_file}")
            
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 