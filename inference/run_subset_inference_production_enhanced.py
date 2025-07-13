#!/usr/bin/env python3
"""
Enhanced wrapper for run_subset_inference_production.py - adds logging and monitoring
for large-scale production runs while preserving the tested pipeline logic.
"""

import os
import sys
import json
import logging
import time
import argparse
from datetime import datetime

# Setup production logging
def setup_production_logging():
    """Setup comprehensive logging for production inference."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"inference/logs/production_inference_{timestamp}.log"
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 PRODUCTION INFERENCE LOG STARTED")
    logger.info(f"📋 Log file: {log_file}")
    
    return logger, log_file

def run_production_inference_with_monitoring(args):
    """Run production inference with enhanced monitoring."""
    logger, log_file = setup_production_logging()
    start_time = time.time()
    
    try:
        # Import your tested pipeline
        from run_subset_inference_production import run_production_subset_inference_ultra_optimized
        
        logger.info("🎯 USING YOUR TESTED PIPELINE: run_subset_inference_production")
        logger.info(f"   Model: {args.model_subdir}")
        logger.info(f"   Vignettes: {args.vignettes}")
        logger.info(f"   Batch size: {args.batch_size}")
        logger.info(f"   All vignettes: {args.all_vignettes}")
        
        # Build filter criteria for your existing function
        filter_criteria = {}
        if args.topics:
            filter_criteria['topics'] = args.topics
        if args.meta_topics:
            filter_criteria['meta_topics'] = args.meta_topics
        if args.topic_keywords:
            filter_criteria['topic_keywords'] = args.topic_keywords
        
        # Generate output path if not provided
        if not args.output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = args.model_subdir.replace("/", "_").replace("\\", "_")
            output_filename = f"production_25k_inference_{model_name}_{timestamp}.json"
            args.output = os.path.join("inference", "results", output_filename)
        
        logger.info(f"💾 Output will be saved to: {args.output}")
        
        # Call your existing tested function
        logger.info("🚀 CALLING YOUR TESTED PIPELINE...")
        results = run_production_subset_inference_ultra_optimized(
            model_subdir=args.model_subdir,
            vignettes_path=args.vignettes,
            filter_criteria=filter_criteria,
            num_samples=None,  # All permutations for 25K run
            output_path=args.output,
            seed=args.seed,
            dry_run=args.dry_run,
            all_vignettes=args.all_vignettes,
            use_hf_hub=args.use_hf_hub,
            batch_size=args.batch_size,
            collect_attention=False,  # Ensure attention is off for speed
            chunk_size=1000
        )
        
        # Final statistics
        total_time = time.time() - start_time
        
        logger.info("✅ PRODUCTION INFERENCE COMPLETED SUCCESSFULLY!")
        logger.info(f"📊 FINAL STATISTICS:")
        logger.info(f"   Total records: {len(results):,}")
        logger.info(f"   Total time: {total_time/3600:.1f} hours")
        logger.info(f"   Average rate: {len(results)/total_time:.1f} samples/sec")
        logger.info(f"   Output file: {args.output}")
        logger.info(f"   Log file: {log_file}")
        
        return results
        
    except KeyboardInterrupt:
        logger.warning("🛑 Inference interrupted by user")
        elapsed = time.time() - start_time
        logger.info(f"⏱️  Ran for {elapsed/3600:.1f} hours before interruption")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Critical error: {str(e)}")
        logger.exception("Full error traceback:")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Production Inference with Monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PRODUCTION READY WRAPPER for your tested pipeline!

Examples for 25K vignettes:
  # Run all vignettes (recommended for 25K inference)
  python run_subset_inference_production_enhanced.py llama3_8b_post_brexit_2020_2025 --all-vignettes --batch-size 32
  
  # Dry run to estimate time
  python run_subset_inference_production_enhanced.py llama3_8b_post_brexit_2020_2025 --all-vignettes --dry-run
  
  # Specific topics only
  python run_subset_inference_production_enhanced.py llama3_8b_post_brexit_2020_2025 --topics "Firm settlement" "Safe third country"
        """
    )
    
    parser.add_argument("model_subdir", help="Model subdirectory under models/ or HF Hub model name")
    parser.add_argument("--vignettes", default="vignettes/complete_vignettes.json",
                       help="Path to vignettes JSON file")
    parser.add_argument("--output", 
                       help="Output path for results (auto-generated if not specified)")
    parser.add_argument("--topics", nargs='+', 
                       help="Specific vignette topics to include")
    parser.add_argument("--meta-topics", nargs='+',
                       help="Specific meta topics to include")
    parser.add_argument("--topic-keywords", nargs='+',
                       help="Keywords that must appear in vignette topic")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for inference (default: 16, try 32-64 for 25K)")
    parser.add_argument("--seed", type=int,
                       help="Random seed for reproducible sampling")
    parser.add_argument("--dry-run", action="store_true",
                       help="Analyze requirements without running inference")
    parser.add_argument("--all-vignettes", action="store_true",
                       help="Process all vignettes (RECOMMENDED for 25K run)")
    parser.add_argument("--use-hf-hub", action="store_true",
                       help="Load model from Hugging Face Hub")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all_vignettes and not any([args.topics, args.meta_topics, args.topic_keywords]):
        print("Error: You must specify --all-vignettes OR filtering criteria")
        print("For 25K inference, use --all-vignettes")
        return 1
    
    # Run production inference
    run_production_inference_with_monitoring(args)
    return 0

if __name__ == "__main__":
    sys.exit(main()) 