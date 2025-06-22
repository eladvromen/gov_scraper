#!/usr/bin/env python3
"""
Subset inference script - Run inference on selected vignettes with random sampling.
This script allows you to test inference on specific vignettes and a limited number
of random permutations per vignette.
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

def generate_random_permutations(vignette, num_samples=3, seed=None):
    """
    Generate random permutations for a vignette.
    
    Args:
        vignette (dict): Vignette configuration
        num_samples (int): Number of random samples to generate
        seed (int): Random seed for reproducibility
    
    Returns:
        list: List of sampled permutation records
    """
    if seed is not None:
        random.seed(seed)
    
    records = []
    
    generic_fields = vignette.get('generic_fields', {})
    ordinal_fields = vignette.get('ordinal_fields', {})
    horizontal_fields = vignette.get('horizontal_fields', {})
    derived_fields = vignette.get('derived_fields', {})

    generic_keys = list(generic_fields.keys())
    generic_lists = [resolve_field_reference(generic_fields[k]) for k in generic_keys]

    ordinal_keys = list(ordinal_fields.keys())
    ordinal_lists = [list(ordinal_fields[k].keys()) for k in ordinal_keys]

    horizontal_keys = list(horizontal_fields.keys())
    horizontal_lists = [horizontal_fields[k] for k in horizontal_keys]

    # Generate all possible permutations first
    all_permutations = []
    for generic_vals in product(*generic_lists):
        for ordinal_vals in product(*ordinal_lists) if ordinal_lists else [()]:
            for horizontal_vals in product(*horizontal_lists) if horizontal_lists else [()]:
                all_permutations.append((generic_vals, ordinal_vals, horizontal_vals))
    
    # Randomly sample from all permutations
    sample_size = min(num_samples, len(all_permutations))
    sampled_permutations = random.sample(all_permutations, sample_size)
    
    print(f"  Sampling {sample_size} out of {len(all_permutations)} possible permutations")
    
    for generic_vals, ordinal_vals, horizontal_vals in sampled_permutations:
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
                    # Use mapping to get possible country_B values
                    mapping = dspec["mapping"]
                    source_field = dspec["source_field"]
                    if mapping == "systems_to_countries_map":
                        val = sample_values.get(source_field)
                        if val and val in systems_to_countries_map:
                            sample_values["country_B"] = systems_to_countries_map[val][0]  # pick first for determinism
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
        
        # Build record (without model response for now)
        record = {
            'meta_topic': vignette['meta_topic'],
            'topic': vignette['topic'],
            'fields': {k: sample_values.get(k) for k in list(generic_keys) + list(ordinal_keys) + list(horizontal_keys) + list(derived_fields.keys()) if k in sample_values},
            'vignette_text': vignette_text,
            'sample_values': sample_values  # Keep for inference
        }
        
        # Add ordinal values as separate fields
        for k in ordinal_keys:
            if f"{k}__ordinal" in sample_values:
                record[f"fields.{k}__ordinal"] = sample_values[f"{k}__ordinal"]
        
        records.append(record)
    
    return records

def run_subset_inference(model_subdir, vignettes_path, filter_criteria, num_samples=3, 
                        output_path=None, seed=None, dry_run=False, all_vignettes=False, use_hf_hub=False):
    """
    Run inference on a subset of vignettes with random sampling.
    
    Args:
        model_subdir (str): Model subdirectory name or HF Hub model name
        vignettes_path (str): Path to vignettes JSON file
        filter_criteria (dict): Criteria for filtering vignettes
        num_samples (int): Number of random samples per vignette
        output_path (str): Output file path
        seed (int): Random seed for reproducibility
        dry_run (bool): If True, don't run actual inference
        all_vignettes (bool): If True, process all vignettes (skip filtering)
        use_hf_hub (bool): If True, load model from Hugging Face Hub
    
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
    
    # Generate random samples
    print(f"\nGenerating {num_samples} random samples per vignette...")
    all_samples = []
    
    for vignette in filtered_vignettes:
        print(f"Processing: {vignette['topic']}")
        samples = generate_random_permutations(vignette, num_samples, seed)
        all_samples.extend(samples)
    
    print(f"\nTotal samples to process: {len(all_samples)}")
    
    if dry_run:
        print("\n[DRY RUN] Would run inference on above samples")
        return all_samples
    
    # Run inference
    print(f"\nLoading model and running inference...")
    from inference_pipeline import InferencePipeline
    
    pipeline = InferencePipeline(model_subdir, use_hf_hub=use_hf_hub)
    
    # Prepare batch inference
    print(f"  Running batch inference on {len(all_samples)} samples...")
    vignette_texts = [sample['vignette_text'] for sample in all_samples]
    
    # Run batch inference with optimized batch size for A100s
    batch_size = 16  # Optimized batch size for A100s with 80GB memory
    model_responses = pipeline.run_inference_auto(vignette_texts, batch_size=batch_size)
    
    # Build final records
    print(f"  Building result records...")
    inference_results = []
    for i, (sample, model_response) in enumerate(zip(all_samples, model_responses)):
        # Create prompt for record (not for inference)
        prompt = pipeline._create_prompt(sample['vignette_text'])
        
        # Build final record
        result = {
            'meta_topic': sample['meta_topic'],
            'topic': sample['topic'],
            'fields': sample['fields'],
            'vignette_text': sample['vignette_text'],
            'model_response': model_response,
            'prompt_used': prompt,
            'inference_timestamp': datetime.now().isoformat(),
            'sampling_info': {
                'num_samples_requested': num_samples,
                'random_seed': seed
            }
        }
        
        # Add ordinal fields
        for key, value in sample.items():
            if key.startswith('fields.') and key.endswith('__ordinal'):
                result[key] = value
        
        inference_results.append(result)
    
    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(inference_results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to: {output_path}")
    
    return inference_results

def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a subset of vignettes with random sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 3 random samples from vignettes containing "settlement" in topic (local model)
  python run_subset_inference.py my_model --topic-keywords settlement --samples 3
  
  # Run 5 samples from specific topics (local model)
  python run_subset_inference.py my_model --topics "Firm settlement" "Safe third country" --samples 5
  
  # Run samples from HF Hub model
  python run_subset_inference.py "meta-llama/Meta-Llama-3-8B-Instruct" --use-hf-hub --topic-keywords persecution --samples 3
  
  # Run samples from specific meta topic
  python run_subset_inference.py my_model --meta-topics "National security vs. human rights" --samples 2
  
  # Run samples from all vignettes
  python run_subset_inference.py my_model --all-vignettes --samples 9
  
  # Dry run to see what would be processed
  python run_subset_inference.py my_model --topic-keywords persecution --dry-run
        """
    )
    
    parser.add_argument("model_subdir", help="Model subdirectory under models/")
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
        output_filename = f"subset_inference_{args.model_subdir}_{timestamp}.json"
        args.output = os.path.join("inference", "results", output_filename)
    
    # Run inference
    try:
        results = run_subset_inference(
            model_subdir=args.model_subdir,
            vignettes_path=args.vignettes,
            filter_criteria=filter_criteria,
            num_samples=args.samples,
            output_path=args.output,
            seed=args.seed,
            dry_run=args.dry_run,
            all_vignettes=args.all_vignettes,
            use_hf_hub=args.use_hf_hub
        )
        
        if not args.dry_run:
            print(f"\n✓ Subset inference completed!")
            print(f"  Total samples processed: {len(results)}")
            print(f"  Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"\nError during subset inference: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 