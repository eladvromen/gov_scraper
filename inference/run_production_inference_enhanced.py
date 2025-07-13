#!/usr/bin/env python3
"""
Enhanced Production Inference Script - Optimized for large-scale inference (25,000+ vignettes)
with comprehensive logging, checkpoint recovery, and resilience features.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
import psutil
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from inference_pipeline import InferencePipeline, setup_logging
from utils import load_vignettes
from calculate_permutations import calculate_vignette_permutations

def get_system_info():
    """Get system information for monitoring."""
    return {
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    }

def estimate_runtime_and_resources(total_permutations, system_info):
    """Estimate runtime and resource requirements."""
    # Conservative estimates based on system capabilities
    if system_info['gpu_count'] > 0:
        # GPU inference - roughly 2-5 samples per second depending on model size
        samples_per_sec = 2.5 * system_info['gpu_count']
    else:
        # CPU inference - much slower
        samples_per_sec = 0.1
    
    total_hours = total_permutations / (samples_per_sec * 3600)
    
    # Memory estimation (rough)
    memory_per_sample_mb = 10  # Conservative estimate
    peak_memory_gb = (memory_per_sample_mb * 1000) / 1024  # Batch processing peak
    
    return {
        'estimated_hours': total_hours,
        'estimated_peak_memory_gb': peak_memory_gb,
        'samples_per_sec': samples_per_sec,
        'recommended_batch_size': min(32, max(1, int(system_info['memory_gb'] / 4)))
    }

def run_enhanced_production_inference(
    model_subdir,
    vignettes_path,
    output_path=None,
    use_hf_hub=False,
    enable_checkpointing=True,
    checkpoint_interval=1000,
    log_level=logging.INFO,
    dry_run=False,
    max_samples=None
):
    """
    Run enhanced production inference with full monitoring and resilience.
    
    Args:
        model_subdir (str): Model directory or HF Hub model name
        vignettes_path (str): Path to vignettes JSON file
        output_path (str): Path to save results
        use_hf_hub (bool): Whether to use HF Hub model
        enable_checkpointing (bool): Enable checkpoint recovery
        checkpoint_interval (int): Checkpoint every N samples
        log_level: Logging level
        dry_run (bool): Just analyze without running inference
        max_samples (int): Limit number of samples for testing
    """
    # Setup enhanced logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"inference/logs/production_inference_{timestamp}.log"
    logger = setup_logging(log_level=log_level, log_file=log_file)
    
    logger.info("🚀 ENHANCED PRODUCTION INFERENCE PIPELINE STARTING")
    logger.info(f"📅 Start time: {datetime.now().isoformat()}")
    
    # System information
    system_info = get_system_info()
    logger.info(f"🖥️  System: {system_info['cpu_count']} CPUs, "
               f"{system_info['memory_gb']:.1f}GB RAM, "
               f"{system_info['gpu_count']} GPUs")
    if system_info['gpu_names']:
        logger.info(f"🎮 GPUs: {', '.join(system_info['gpu_names'])}")
    
    try:
        # Load and analyze vignettes
        logger.info(f"📄 Loading vignettes from: {vignettes_path}")
        vignettes = load_vignettes(vignettes_path)
        logger.info(f"✅ Loaded {len(vignettes)} vignette configurations")
        
        # Calculate total permutations
        logger.info("🔢 Calculating total permutations...")
        total_permutations = 0
        vignette_details = []
        
        for vignette in vignettes:
            result = calculate_vignette_permutations(vignette)
            total_permutations += result['total']
            vignette_details.append((vignette, result))
        
        # Apply sample limit if specified
        if max_samples and total_permutations > max_samples:
            logger.warning(f"⚠️  Limiting to {max_samples:,} samples (from {total_permutations:,})")
            total_permutations = max_samples
        
        logger.info(f"📊 Total permutations: {total_permutations:,}")
        
        # Detailed breakdown
        logger.info("📋 Vignette breakdown:")
        for i, (vignette, result) in enumerate(vignette_details[:10]):  # Show first 10
            logger.info(f"   {i+1:2d}. {vignette['topic']}: {result['total']:,} permutations")
        if len(vignette_details) > 10:
            logger.info(f"   ... and {len(vignette_details)-10} more vignettes")
        
        # Resource estimation
        estimates = estimate_runtime_and_resources(total_permutations, system_info)
        logger.info(f"⏱️  Estimated runtime: {estimates['estimated_hours']:.1f} hours")
        logger.info(f"💾 Estimated peak memory: {estimates['estimated_peak_memory_gb']:.1f}GB")
        logger.info(f"⚡ Expected rate: {estimates['samples_per_sec']:.1f} samples/sec")
        logger.info(f"📦 Recommended batch size: {estimates['recommended_batch_size']}")
        
        # Generate output path if not provided
        if not output_path:
            output_filename = f"production_inference_{model_subdir}_{timestamp}.json"
            output_path = os.path.join("inference", "results", output_filename)
        
        logger.info(f"💾 Output path: {output_path}")
        
        if dry_run:
            logger.info("🧪 DRY RUN COMPLETE - No inference performed")
            return
        
        # Confirm for large runs
        if total_permutations > 5000:
            print(f"\n⚠️  LARGE SCALE INFERENCE")
            print(f"   Samples: {total_permutations:,}")
            print(f"   Estimated time: {estimates['estimated_hours']:.1f} hours")
            print(f"   Log file: {log_file}")
            print(f"   Checkpointing: {'Enabled' if enable_checkpointing else 'Disabled'}")
            
            response = input(f"\nProceed with inference? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                logger.info("❌ Inference cancelled by user")
                return
        
        # Initialize pipeline
        logger.info("🤖 Initializing inference pipeline...")
        pipeline = InferencePipeline(
            model_subdir=model_subdir,
            use_hf_hub=use_hf_hub,
            enable_checkpointing=enable_checkpointing,
            checkpoint_interval=checkpoint_interval,
            log_file=log_file
        )
        
        # Run inference
        logger.info("🚀 Starting inference generation...")
        results = pipeline.generate_inference_records(vignettes)
        
        # Save results
        logger.info(f"💾 Saving results to: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Final statistics
        logger.info("✅ PRODUCTION INFERENCE COMPLETED SUCCESSFULLY")
        logger.info(f"📊 Results: {len(results):,} records generated")
        logger.info(f"📁 Output: {output_path}")
        logger.info(f"📋 Log: {log_file}")
        
        return results
        
    except KeyboardInterrupt:
        logger.warning("🛑 Inference interrupted by user")
        raise
    except Exception as e:
        logger.error(f"❌ Critical error during production inference: {str(e)}")
        logger.exception("Full error traceback:")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Production Inference Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full production inference
  python run_production_inference_enhanced.py llama3_8b_post_brexit_2020_2025
  
  # Dry run to estimate resources
  python run_production_inference_enhanced.py llama3_8b_post_brexit_2020_2025 --dry-run
  
  # Test with limited samples
  python run_production_inference_enhanced.py llama3_8b_post_brexit_2020_2025 --max-samples 1000
  
  # Use Hugging Face model
  python run_production_inference_enhanced.py meta-llama/Llama-2-7b-chat-hf --use-hf-hub
        """
    )
    
    parser.add_argument("model_subdir", 
                       help="Model subdirectory under models/ or HF Hub model name")
    parser.add_argument("--vignettes", default="vignettes/complete_vignettes.json",
                       help="Path to vignettes JSON file")
    parser.add_argument("--output", 
                       help="Output path for results (auto-generated if not specified)")
    parser.add_argument("--use-hf-hub", action="store_true",
                       help="Use Hugging Face Hub model")
    parser.add_argument("--no-checkpointing", action="store_true",
                       help="Disable checkpoint recovery")
    parser.add_argument("--checkpoint-interval", type=int, default=1000,
                       help="Save checkpoint every N samples (default: 1000)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--dry-run", action="store_true",
                       help="Analyze requirements without running inference")
    parser.add_argument("--max-samples", type=int,
                       help="Limit number of samples (for testing)")
    
    args = parser.parse_args()
    
    # Set log level
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    try:
        run_enhanced_production_inference(
            model_subdir=args.model_subdir,
            vignettes_path=args.vignettes,
            output_path=args.output,
            use_hf_hub=args.use_hf_hub,
            enable_checkpointing=not args.no_checkpointing,
            checkpoint_interval=args.checkpoint_interval,
            log_level=log_level,
            dry_run=args.dry_run,
            max_samples=args.max_samples
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Inference interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 