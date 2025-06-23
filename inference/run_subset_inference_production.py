#!/usr/bin/env python3
"""
Production subset inference script - Run inference on selected vignettes with proper sample_id generation.
This script uses the ProductionAttentionPipeline for optimized processing and consistent sample_id generation
that is critical for proper dataset matching in analysis.
"""

import os
import sys
import json
import random
import argparse
from datetime import datetime
from pathlib import Path
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from utils import load_vignettes, resolve_field_reference
from calculate_permutations import calculate_vignette_permutations
from production_attention_pipeline import ProductionAttentionPipeline
from itertools import product

# Add vignettes directory to path for imports
vignettes_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../vignettes'))
sys.path.insert(0, vignettes_path)
from field_definitions import get_name_for_country_gender, get_pronoun, systems_to_countries_map, safety_to_countries_map

def filter_vignettes(vignettes, filter_criteria):
    """
    Filter vignettes based on specified criteria.
    
    Args:
        vignettes (list): List of all vignettes
        filter_criteria (dict): Filtering criteria
    
    Returns:
        list: Filtered vignettes
    """
    filtered = []
    
    for vignette in vignettes:
        include = True
        
        # Filter by topic
        if 'topics' in filter_criteria:
            if vignette['topic'] not in filter_criteria['topics']:
                include = False
        
        # Filter by meta_topic
        if 'meta_topics' in filter_criteria:
            if vignette['meta_topic'] not in filter_criteria['meta_topics']:
                include = False
        
        # Filter by keywords in topic
        if 'topic_keywords' in filter_criteria:
            topic_lower = vignette['topic'].lower()
            if not any(keyword.lower() in topic_lower for keyword in filter_criteria['topic_keywords']):
                include = False
        
        if include:
            filtered.append(vignette)
    
    return filtered

def generate_subset_samples(vignettes, num_samples_per_vignette=3, seed=None):
    """
    Generate exact sample records for a subset of permutations from each vignette.
    This pre-generates the actual sample combinations instead of modifying vignette configs.
    
    Args:
        vignettes (list): List of vignette configurations
        num_samples_per_vignette (int): Number of samples to generate per vignette
        seed (int): Random seed for reproducibility
    
    Returns:
        list: Pre-generated sample records ready for inference
    """
    if seed is not None:
        random.seed(seed)
    
    all_samples = []
    current_sample_id = 1
    
    for vignette in vignettes:
        print(f"Generating samples for: {vignette['topic']}")
        
        # Calculate total permutations for this vignette
        result = calculate_vignette_permutations(vignette)
        total_perms = result['total']
        
        print(f"  Total permutations: {total_perms:,}")
        
        # Sample field combinations
        generic_fields = vignette.get('generic_fields', {})
        ordinal_fields = vignette.get('ordinal_fields', {})
        horizontal_fields = vignette.get('horizontal_fields', {})
        derived_fields = vignette.get('derived_fields', {})

        # Get all possible combinations
        generic_keys = list(generic_fields.keys())
        generic_lists = [resolve_field_reference(generic_fields[k]) for k in generic_keys]

        ordinal_keys = list(ordinal_fields.keys())
        ordinal_lists = [list(ordinal_fields[k].keys()) for k in ordinal_keys]

        horizontal_keys = list(horizontal_fields.keys())
        horizontal_lists = [horizontal_fields[k] for k in horizontal_keys]

        # Generate all possible combinations
        all_combinations = []
        for generic_vals in product(*generic_lists):
            for ordinal_vals in product(*ordinal_lists) if ordinal_lists else [()]:
                for horizontal_vals in product(*horizontal_lists) if horizontal_lists else [()]:
                    all_combinations.append((generic_vals, ordinal_vals, horizontal_vals))
        
        # Sample the desired number of combinations
        sample_size = min(num_samples_per_vignette, len(all_combinations))
        if sample_size == len(all_combinations):
            print(f"  Using all {sample_size} combinations")
            sampled_combinations = all_combinations
        else:
            print(f"  Sampling {sample_size} out of {len(all_combinations)} combinations")
            sampled_combinations = random.sample(all_combinations, sample_size)
        
        # Generate sample records
        for generic_vals, ordinal_vals, horizontal_vals in sampled_combinations:
            sample_values = {}
            
            # Fill generic fields
            for k, v in zip(generic_keys, generic_vals):
                sample_values[k] = v
                
            # Fill ordinal fields (label and value)
            for k, v in zip(ordinal_keys, ordinal_vals):
                sample_values[k] = v
                sample_values[f"{k}__ordinal"] = ordinal_fields[k][v]
                
            # Fill horizontal fields
            for k, v in zip(horizontal_keys, horizontal_vals):
                sample_values[k] = v
                
            # Handle derived fields
            if derived_fields:
                for dfield, dspec in derived_fields.items():
                    if dfield == "name" and "country" in sample_values and "gender" in sample_values:
                        sample_values["name"] = get_name_for_country_gender(sample_values["country"], sample_values["gender"])
                    elif dfield == "country_B":
                        mapping = dspec["mapping"]
                        source_field = dspec["source_field"]
                        if mapping == "systems_to_countries_map":
                            val = sample_values.get(source_field)
                            if val and val in systems_to_countries_map:
                                sample_values["country_B"] = systems_to_countries_map[val][0]
                        elif mapping == "safety_to_countries_map":
                            val = sample_values.get(source_field)
                            if val and val in safety_to_countries_map:
                                sample_values["country_B"] = safety_to_countries_map[val][0]
            
            # Add pronoun
            if 'gender' in sample_values:
                sample_values['pronoun'] = get_pronoun(sample_values['gender'])
            
            # Generate vignette text
            try:
                vignette_text = vignette["vignette_template"].format(**sample_values)
            except KeyError as e:
                print(f"Warning: Missing field {e} for vignette template")
                continue
            
            # Build sample record
            sample_record = {
                'sample_id': current_sample_id,
                'meta_topic': vignette['meta_topic'],
                'topic': vignette['topic'],
                'fields': {k: sample_values.get(k) for k in list(generic_keys) + list(ordinal_keys) + list(horizontal_keys) + list(derived_fields.keys()) if k in sample_values},
                'vignette_text': vignette_text,
            }
            
            # Add ordinal values as separate fields
            for k in ordinal_keys:
                if f"{k}__ordinal" in sample_values:
                    sample_record[f"fields.{k}__ordinal"] = sample_values[f"{k}__ordinal"]
            
            all_samples.append(sample_record)
            current_sample_id += 1
    
    return all_samples

def run_production_subset_inference(model_subdir, vignettes_path, filter_criteria, num_samples=3, 
                                  output_path=None, seed=None, dry_run=False, all_vignettes=False, 
                                  use_hf_hub=False, batch_size=16, collect_attention=False):
    """
    Run inference on a subset of vignettes using the ProductionAttentionPipeline.
    
    Args:
        model_subdir (str): Model subdirectory name or HF Hub model name
        vignettes_path (str): Path to vignettes JSON file
        filter_criteria (dict): Criteria for filtering vignettes
        num_samples (int): Number of random samples per vignette
        output_path (str): Output file path
        seed (int): Random seed for reproducible sampling
        dry_run (bool): If True, don't run actual inference
        all_vignettes (bool): If True, process all vignettes (skip filtering)
        use_hf_hub (bool): If True, load model from Hugging Face Hub
        batch_size (int): Batch size for inference
        collect_attention (bool): If True, collect attention data
    
    Returns:
        list: Inference results
    """
    # Load vignettes
    print("Loading vignettes...")
    vignettes = load_vignettes(vignettes_path)
    print(f"Loaded {len(vignettes)} total vignettes")
    
    # Filter vignettes (or use all if all_vignettes is True)
    if all_vignettes:
        filtered_vignettes = vignettes
        print(f"Processing all {len(filtered_vignettes)} vignettes")
    else:
        filtered_vignettes = filter_vignettes(vignettes, filter_criteria)
        print(f"Filtered to {len(filtered_vignettes)} vignettes matching criteria")
    
    if not filtered_vignettes:
        print("No vignettes to process!")
        return []
    
    print("\nSelected vignettes:")
    for i, vignette in enumerate(filtered_vignettes):
        result = calculate_vignette_permutations(vignette)
        print(f"  {i+1}. {vignette['topic']} ({result['total']:,} total permutations)")
    
    # Generate subset samples directly
    print(f"\nGenerating subset samples ({num_samples} samples per vignette)...")
    sample_records = generate_subset_samples(
        filtered_vignettes, 
        num_samples_per_vignette=num_samples, 
        seed=seed
    )
    
    print(f"\nTotal samples to process: {len(sample_records)}")
    
    if dry_run:
        print("\n[DRY RUN] Would run inference on above samples")
        return sample_records
    
    # Run inference using ProductionAttentionPipeline
    print(f"\nInitializing ProductionAttentionPipeline...")
    if collect_attention:
        print(f"⚠️  Attention collection ENABLED - this will be slower but collect attention data")
    else:
        print(f"🚀 Attention collection DISABLED - optimized for speed")
        
    pipeline = ProductionAttentionPipeline(
        model_subdir, 
        use_hf_hub=use_hf_hub,
        collect_attention=collect_attention
    )
    
    # Create prompts for all samples
    print(f"Running {'attention-enabled' if collect_attention else 'optimized'} inference...")
    prompts = []
    for record in sample_records:
        prompt = pipeline._create_prompt(record['vignette_text'])
        prompts.append(prompt)
    
    # Run batch inference
    if collect_attention:
        # For attention mode, we need to use the attention-enabled method
        # This requires creating temporary vignette-like structures
        print("Note: Attention collection with pre-generated samples not fully implemented")
        print("Falling back to non-attention mode for now...")
        responses = pipeline._run_inference_batch_optimized(prompts, batch_size=batch_size)
    else:
        responses = pipeline._run_inference_batch_optimized(prompts, batch_size=batch_size)
    
    # Combine sample records with model responses
    inference_results = []
    for record, response in zip(sample_records, responses):
        result = record.copy()
        result['model_response'] = response
        result['inference_timestamp'] = datetime.now().isoformat()
        inference_results.append(result)
    
    # Add sampling metadata to each record
    for record in inference_results:
        record['sampling_info'] = {
            'num_samples_requested': num_samples,
            'random_seed': seed,
            'production_pipeline': True,
            'batch_size': batch_size
        }
    
    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(inference_results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to: {output_path}")
        print(f"✓ All records include proper sample_id for dataset matching")
    
    return inference_results

def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a subset of vignettes using ProductionAttentionPipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 3 random samples from vignettes containing "settlement" in topic (local model)
  python run_subset_inference_production.py my_model --topic-keywords settlement --samples 3
  
  # Run 5 samples from specific topics (local model)
  python run_subset_inference_production.py my_model --topics "Firm settlement" "Safe third country" --samples 5
  
  # Run samples from HF Hub model with larger batch size
  python run_subset_inference_production.py "meta-llama/Meta-Llama-3-8B-Instruct" --use-hf-hub --topic-keywords persecution --samples 3 --batch-size 32
  
  # Run samples from specific meta topic
  python run_subset_inference_production.py my_model --meta-topics "National security vs. human rights" --samples 2
  
  # Run samples from all vignettes
  python run_subset_inference_production.py my_model --all-vignettes --samples 9
  
  # Enable attention collection (slower but collects attention data)
  python run_subset_inference_production.py my_model --topic-keywords settlement --samples 3 --collect-attention
  
  # Dry run to see what would be processed
  python run_subset_inference_production.py my_model --topic-keywords persecution --dry-run
        """
    )
    
    parser.add_argument("model_subdir", help="Model subdirectory under models/ or HF Hub model name")
    parser.add_argument("--vignettes", default="vignettes/complete_vignettes.json",
                       help="Path to vignettes JSON file")
    parser.add_argument("--output", 
                       help="Output path for inference results (default: auto-generated)")
    parser.add_argument("--topics", nargs='+', 
                       help="Specific vignette topics to include")
    parser.add_argument("--meta-topics", nargs='+',
                       help="Specific meta topics to include")
    parser.add_argument("--topic-keywords", nargs='+',
                       help="Keywords that must appear in vignette topic")
    parser.add_argument("--samples", type=int, default=3,
                       help="Number of random samples per vignette (default: 3)")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for inference (default: 16)")
    parser.add_argument("--seed", type=int,
                       help="Random seed for reproducible sampling")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be processed without running inference")
    parser.add_argument("--list-topics", action="store_true",
                       help="List all available topics and exit")
    parser.add_argument("--all-vignettes", action="store_true",
                       help="Process all vignettes (skip filtering)")
    parser.add_argument("--use-hf-hub", action="store_true",
                       help="Load model from Hugging Face Hub instead of local directory")
    parser.add_argument("--collect-attention", action="store_true",
                       help="Enable attention data collection (slower but useful for research)")
    
    args = parser.parse_args()
    
    # List topics if requested
    if args.list_topics:
        vignettes = load_vignettes(args.vignettes)
        print("Available topics:")
        topics = sorted(set(v['topic'] for v in vignettes))
        for topic in topics:
            print(f"  - {topic}")
        print("\nAvailable meta topics:")
        meta_topics = sorted(set(v['meta_topic'] for v in vignettes))
        for meta_topic in meta_topics:
            print(f"  - {meta_topic}")
        return 0
    
    # Build filter criteria
    filter_criteria = {}
    if args.topics:
        filter_criteria['topics'] = args.topics
    if args.meta_topics:
        filter_criteria['meta_topics'] = args.meta_topics
    if args.topic_keywords:
        filter_criteria['topic_keywords'] = args.topic_keywords
    
    if not filter_criteria and not args.all_vignettes:
        print("Error: You must specify at least one filtering criterion")
        print("Use --topics, --meta-topics, --topic-keywords, or --all-vignettes")
        print("Or use --list-topics to see available options")
        return 1
    
    # Generate output filename if not provided
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = args.model_subdir.replace("/", "_").replace("\\", "_")
        output_filename = f"production_subset_inference_{model_name}_{timestamp}.json"
        args.output = os.path.join("inference", "results", output_filename)
    
    # Run inference
    try:
        results = run_production_subset_inference(
            model_subdir=args.model_subdir,
            vignettes_path=args.vignettes,
            filter_criteria=filter_criteria,
            num_samples=args.samples,
            output_path=args.output,
            seed=args.seed,
            dry_run=args.dry_run,
            all_vignettes=args.all_vignettes,
            use_hf_hub=args.use_hf_hub,
            batch_size=args.batch_size,
            collect_attention=args.collect_attention
        )
        
        if not args.dry_run:
            print(f"\n✓ Production subset inference completed!")
            print(f"  Total samples processed: {len(results)}")
            print(f"  Results saved to: {args.output}")
            print(f"  All records include sample_id for proper dataset matching")
        
        return 0
        
    except Exception as e:
        print(f"\nError during production subset inference: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 