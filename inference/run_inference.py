#!/usr/bin/env python3
"""
Utility script to run inference pipeline on vignettes.
This script provides an easy interface to run inference with different models
and configurations.
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from utils import load_vignettes
from calculate_permutations import calculate_vignette_permutations

def validate_model_path(models_dir, model_subdir):
    """Validate that the model directory exists and contains necessary files."""
    model_path = os.path.join(models_dir, model_subdir)
    
    if not os.path.exists(model_path):
        print(f"Error: Model directory not found: {model_path}")
        return False
    
    # Check for essential model files
    config_file = os.path.join(model_path, "config.json")
    if not os.path.exists(config_file):
        print(f"Warning: config.json not found in {model_path}")
    
    return True

def estimate_runtime(total_permutations, inference_time_per_sample=2.0):
    """Estimate total runtime for inference."""
    total_seconds = total_permutations * inference_time_per_sample
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    
    if hours > 0:
        return f"~{hours:.0f}h {minutes:.0f}m"
    elif minutes > 0:
        return f"~{minutes:.0f}m"
    else:
        return f"~{total_seconds:.0f}s"

def main():
    parser = argparse.ArgumentParser(
        description="Run inference pipeline on vignette permutations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference with a specific model
  python run_inference.py my_model_name
  
  # Use custom vignettes and output paths
  python run_inference.py my_model --vignettes custom_vignettes.json --output results/my_inference.json
  
  # Just count permutations without running inference
  python run_inference.py --count-only
  
  # Run with verbose output
  python run_inference.py my_model --verbose
        """
    )
    
    parser.add_argument("model_subdir", nargs='?', 
                       help="Model subdirectory under models/ (required unless --count-only)")
    parser.add_argument("--vignettes", default="vignettes/complete_vignettes.json",
                       help="Path to vignettes JSON file (default: vignettes/complete_vignettes.json)")
    parser.add_argument("--output", 
                       help="Output path for inference results (default: auto-generated)")
    parser.add_argument("--models-dir", default="models",
                       help="Base directory containing model subdirectories (default: models)")
    parser.add_argument("--count-only", action="store_true",
                       help="Only count permutations, don't run inference")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without actually running inference")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.count_only and not args.model_subdir:
        parser.error("model_subdir is required unless --count-only is specified")
    
    # Load and analyze vignettes
    print("Loading vignettes...")
    try:
        vignettes = load_vignettes(args.vignettes)
        print(f"Loaded {len(vignettes)} vignette configurations from {args.vignettes}")
    except Exception as e:
        print(f"Error loading vignettes: {e}")
        return 1
    
    # Count total permutations
    print("\nAnalyzing vignette permutations...")
    try:
        total_permutations = 0
        vignette_details = []
        
        for vignette in vignettes:
            result = calculate_vignette_permutations(vignette)
            total_permutations += result['total']
            vignette_details.append((vignette, result))
        
        print(f"Total permutations across all vignettes: {total_permutations:,}")
        
        # Show breakdown by vignette
        if args.verbose:
            print("\nPermutation breakdown by vignette:")
            for i, (vignette, result) in enumerate(vignette_details):
                print(f"  {i+1:2d}. {vignette['topic']}: {result['total']:,} permutations")
        
        # Estimate runtime
        estimated_time = estimate_runtime(total_permutations)
        print(f"Estimated inference time: {estimated_time}")
        
    except Exception as e:
        print(f"Error counting permutations: {e}")
        return 1
    
    # If count-only, exit here
    if args.count_only:
        return 0
    
    # Validate model
    if not validate_model_path(args.models_dir, args.model_subdir):
        return 1
    
    # Generate output filename if not provided
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"inference_results_{args.model_subdir}_{timestamp}.json"
        args.output = os.path.join("inference", "results", output_filename)
    
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_subdir}")
    print(f"  Vignettes: {args.vignettes}")
    print(f"  Output: {args.output}")
    print(f"  Total permutations: {total_permutations:,}")
    print(f"  Estimated time: {estimated_time}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would run inference with above configuration")
        return 0
    
    # Confirm before running large inference jobs
    if total_permutations > 1000:
        response = input(f"\nThis will generate {total_permutations:,} inference samples. Continue? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Inference cancelled.")
            return 0
    
    # Run inference
    try:
        print(f"\nStarting inference pipeline...")
        # Import here to avoid torch dependency for count-only mode
        from inference_pipeline import InferencePipeline
        pipeline = InferencePipeline(args.model_subdir, args.models_dir)
        results = pipeline.run_pipeline(args.vignettes, args.output)
        
        print(f"\n✓ Inference completed successfully!")
        print(f"  Results saved to: {args.output}")
        print(f"  Total records: {len(results):,}")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n\nInference interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError during inference: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 