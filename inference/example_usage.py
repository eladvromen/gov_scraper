#!/usr/bin/env python3
"""
Example usage of the inference pipeline.
This script demonstrates how to use the inference pipeline programmatically.
"""

import os
import sys
from datetime import datetime

# Add the inference directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from inference_pipeline import InferencePipeline
from calculate_permutations import calculate_vignette_permutations

def example_basic_usage():
    """Basic example of running inference with a model."""
    print("=== Basic Usage Example ===")
    
    # Configuration
    model_subdir = "my_model"  # Replace with your actual model directory name
    vignettes_path = "vignettes/complete_vignettes.json"
    output_path = f"inference/results/example_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        # Count permutations first
        from utils import load_vignettes
        vignettes = load_vignettes(vignettes_path)
        total_perms = sum(calculate_vignette_permutations(v)['total'] for v in vignettes)
        print(f"Total permutations to process: {total_perms:,}")
        
        # Create and run pipeline
        pipeline = InferencePipeline(model_subdir)
        results = pipeline.run_pipeline(vignettes_path, output_path)
        
        print(f"✓ Generated {len(results)} inference results")
        print(f"✓ Saved to: {output_path}")
        
    except FileNotFoundError as e:
        print(f"❌ Model not found: {e}")
        print("Make sure you have a model in the 'models/' directory")
    except Exception as e:
        print(f"❌ Error: {e}")

def example_custom_configuration():
    """Example with custom configuration options."""
    print("\n=== Custom Configuration Example ===")
    
    # Custom configuration
    model_subdir = "my_custom_model"
    models_base_dir = "custom_models"  # Different base directory
    vignettes_path = "vignettes/complete_vignettes.json"
    output_path = "inference/results/custom_output.json"
    
    try:
        # Initialize with custom models directory
        pipeline = InferencePipeline(model_subdir, models_base_dir)
        
        # Run inference
        results = pipeline.run_pipeline(vignettes_path, output_path)
        
        print(f"✓ Custom configuration inference completed")
        print(f"✓ Results: {len(results)} records")
        
    except Exception as e:
        print(f"❌ Custom configuration failed: {e}")

def example_process_subset():
    """Example of processing only a subset of vignettes."""
    print("\n=== Subset Processing Example ===")
    
    import json
    from utils import load_vignettes
    
    try:
        # Load all vignettes
        all_vignettes = load_vignettes("vignettes/complete_vignettes.json")
        
        # Select only vignettes from a specific meta_topic
        subset_vignettes = [
            v for v in all_vignettes 
            if v['meta_topic'] == 'National security vs. human rights'
        ]
        
        print(f"Selected {len(subset_vignettes)} vignettes from 'National security vs. human rights'")
        
        # Save subset to temporary file
        subset_path = "inference/temp_subset_vignettes.json"
        with open(subset_path, 'w', encoding='utf-8') as f:
            json.dump(subset_vignettes, f, indent=2, ensure_ascii=False)
        
        # Process subset
        model_subdir = "my_model"
        output_path = "inference/results/subset_output.json"
        
        pipeline = InferencePipeline(model_subdir)
        results = pipeline.run_pipeline(subset_path, output_path)
        
        print(f"✓ Processed subset: {len(results)} records")
        
        # Clean up temporary file
        os.remove(subset_path)
        
    except Exception as e:
        print(f"❌ Subset processing failed: {e}")

def example_analyze_results():
    """Example of analyzing inference results after generation."""
    print("\n=== Results Analysis Example ===")
    
    # This assumes you have some results file from previous runs
    results_path = "inference/results/example_output.json"
    
    try:
        import json
        
        # Check if results file exists
        if not os.path.exists(results_path):
            print(f"No results file found at {results_path}")
            print("Run inference first to generate results")
            return
        
        # Load and analyze results
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"Loaded {len(results)} inference results")
        
        # Analyze by meta_topic
        meta_topics = {}
        for result in results:
            topic = result['meta_topic']
            meta_topics[topic] = meta_topics.get(topic, 0) + 1
        
        print("\nResults by meta topic:")
        for topic, count in meta_topics.items():
            print(f"  {topic}: {count} results")
        
        # Analyze model responses
        response_lengths = [len(result['model_response']) for result in results]
        avg_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
        
        print(f"\nResponse statistics:")
        print(f"  Average response length: {avg_length:.1f} characters")
        print(f"  Min response length: {min(response_lengths) if response_lengths else 0}")
        print(f"  Max response length: {max(response_lengths) if response_lengths else 0}")
        
    except Exception as e:
        print(f"❌ Results analysis failed: {e}")

def main():
    """Run all examples."""
    print("Inference Pipeline Usage Examples")
    print("=" * 50)
    
    # Note: These examples assume you have models available
    print("\nNOTE: These examples require models in the 'models/' directory")
    print("Replace 'my_model' with your actual model directory name\n")
    
    # Run examples (commented out since they require actual models)
    # example_basic_usage()
    # example_custom_configuration()
    # example_process_subset()
    example_analyze_results()
    
    print("\n" + "=" * 50)
    print("To run inference with your own model:")
    print("1. Place your model in models/your_model_name/")
    print("2. Use: python run_inference.py your_model_name")
    print("3. Or use the InferencePipeline class programmatically")

if __name__ == "__main__":
    main() 