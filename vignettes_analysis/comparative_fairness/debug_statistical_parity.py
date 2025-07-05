#!/usr/bin/env python3
"""
Debug script to understand why statistical parity is only finding gender comparisons
"""

import json
import sys
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))
from data_utils import load_config, get_reference_groups

def debug_results_aggregation():
    """Debug the results aggregation to understand why only gender is saved"""
    
    print("=== DEBUGGING RESULTS AGGREGATION ===\n")
    
    # Load configuration and data
    config_path = Path(__file__).parent / "config" / "analysis_config.yaml"
    processed_dir = Path(__file__).parent / "data" / "processed"
    
    config = load_config(config_path)
    with open(processed_dir / "tagged_records.json", 'r') as f:
        records = json.load(f)
    
    # Get reference groups
    reference_groups = get_reference_groups(records, config)
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Add decision_binary column
    df['decision_binary'] = df['decision'].apply(lambda x: 1 if x == 'GRANT' else 0)
    
    # Simulate the statistical parity calculation logic
    results = {}
    
    for attr in config['protected_attributes']:
        print(f"\n{'='*60}")
        print(f"PROCESSING ATTRIBUTE: {attr.upper()}")
        print(f"{'='*60}")
        
        attr_results = {}
        reference_value = reference_groups[attr]
        print(f"Reference value: {reference_value}")
        
        # Get unique topics and values for this attribute
        topics = df['topic'].unique()
        attr_values = df[f"protected_attributes"].apply(lambda x: x[attr]).unique()
        
        print(f"All values for {attr}: {list(attr_values)}")
        print(f"Topics to process: {len(topics)}")
        
        topic_count = 0
        valid_topic_count = 0
        
        for topic in topics:
            topic_data = df[df['topic'] == topic]
            topic_size = len(topic_data)
            
            # Skip if not enough data overall
            if topic_size < 60:
                print(f"  Topic '{topic}' SKIPPED: Only {topic_size} records (need ≥60)")
                continue
                
            topic_count += 1
            print(f"\n  Topic {topic_count}: {topic} ({topic_size} records)")
            
            topic_results = {}
            comparisons_for_topic = 0
            
            for attr_value in attr_values:
                if attr_value == reference_value:
                    continue
                
                # Check for each model
                for model in ['post_brexit', 'pre_brexit']:
                    model_data = topic_data[topic_data['model'] == model]
                    
                    # Extract attribute values
                    model_attr_values = model_data[f"protected_attributes"].apply(lambda x: x[attr])
                    
                    # Get group and reference data
                    group_data = model_data[model_attr_values == attr_value]
                    ref_data = model_data[model_attr_values == reference_value]
                    
                    group_size = len(group_data)
                    ref_size = len(ref_data)
                    
                    comparison_name = f"{attr_value}_vs_{reference_value}_{model}"
                    
                    if group_size >= 30 and ref_size >= 30:
                        # Calculate basic metrics
                        group_rate = group_data['decision_binary'].mean()
                        ref_rate = ref_data['decision_binary'].mean()
                        
                        # Mock z-test result
                        z_score = abs(group_rate - ref_rate) * 10  # Mock calculation
                        p_value = 0.01 if z_score > 1.96 else 0.5  # Mock p-value
                        
                        topic_results[comparison_name] = {
                            'attribute': attr,
                            'group': attr_value,
                            'reference': reference_value,
                            'model': model,
                            'topic': topic,
                            'group_grant_rate': float(group_rate),
                            'reference_grant_rate': float(ref_rate),
                            'sp_gap': float(group_rate - ref_rate),
                            'group_size': int(group_size),
                            'reference_size': int(ref_size),
                            'z_score': float(z_score),
                            'p_value': float(p_value),
                            'is_significant': bool(z_score > 1.96)
                        }
                        
                        comparisons_for_topic += 1
                        print(f"    ✓ {comparison_name}: Added to results")
                    else:
                        print(f"    ✗ {comparison_name}: FILTERED (Group={group_size}, Ref={ref_size})")
            
            # CHECK: Are topic results being added properly?
            if topic_results:
                attr_results[topic] = topic_results
                valid_topic_count += 1
                print(f"    → Topic results added: {comparisons_for_topic} comparisons")
            else:
                print(f"    → Topic results EMPTY - NO COMPARISONS ADDED")
        
        print(f"\n  ATTRIBUTE SUMMARY for {attr}:")
        print(f"    Topics processed: {topic_count}")
        print(f"    Valid topics: {valid_topic_count}")
        print(f"    attr_results keys: {list(attr_results.keys())}")
        print(f"    attr_results is empty: {len(attr_results) == 0}")
        
        # CHECK: Are attribute results being added to main results?
        if attr_results:
            results[attr] = attr_results
            print(f"    → ATTRIBUTE RESULTS ADDED TO MAIN RESULTS")
        else:
            print(f"    → ATTRIBUTE RESULTS EMPTY - NOT ADDED TO MAIN RESULTS")
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    
    print(f"Main results keys: {list(results.keys())}")
    for attr, attr_results in results.items():
        total_comparisons = sum(len(topic_results) for topic_results in attr_results.values())
        print(f"  {attr}: {len(attr_results)} topics, {total_comparisons} total comparisons")
    
    return results

def debug_statistical_parity():
    """Debug the statistical parity calculation step by step"""
    
    print("=== DEBUGGING STATISTICAL PARITY CALCULATION ===\n")
    
    # Load configuration and data
    config_path = Path(__file__).parent / "config" / "analysis_config.yaml"
    processed_dir = Path(__file__).parent / "data" / "processed"
    
    config = load_config(config_path)
    with open(processed_dir / "tagged_records.json", 'r') as f:
        records = json.load(f)
    
    print(f"Total records: {len(records)}")
    print(f"Min group size required: {config['min_group_size']}")
    
    # Get reference groups
    reference_groups = get_reference_groups(records, config)
    print(f"\nReference groups:")
    for attr, ref_group in reference_groups.items():
        print(f"  {attr}: {ref_group}")
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    print(f"\nDataFrame shape: {df.shape}")
    
    # Check overall distribution
    print(f"\nModel distribution:")
    print(df['model'].value_counts())
    
    print(f"\nDecision distribution:")
    print(df['decision'].value_counts())
    
    # Analyze each attribute
    for attr in config['protected_attributes']:
        print(f"\n{'='*60}")
        print(f"ANALYZING ATTRIBUTE: {attr.upper()}")
        print(f"{'='*60}")
        
        reference_value = reference_groups[attr]
        print(f"Reference value: {reference_value}")
        
        # Get all unique values for this attribute
        attr_values = df[f"protected_attributes"].apply(lambda x: x[attr]).unique()
        print(f"All values: {list(attr_values)}")
        
        # Count topics
        topics = df['topic'].unique()
        print(f"Total topics: {len(topics)}")
        
        valid_comparisons = 0
        filtered_comparisons = 0
        
        for topic in topics:
            topic_data = df[df['topic'] == topic]
            topic_size = len(topic_data)
            
            # Skip if not enough data overall
            if topic_size < 60:
                print(f"  Topic '{topic}' SKIPPED: Only {topic_size} records (need ≥60)")
                continue
            
            print(f"\n  Topic: {topic} ({topic_size} records)")
            
            for attr_value in attr_values:
                if attr_value == reference_value:
                    continue
                
                # Check for each model
                for model in ['post_brexit', 'pre_brexit']:
                    model_data = topic_data[topic_data['model'] == model]
                    
                    # Extract attribute values
                    model_attr_values = model_data[f"protected_attributes"].apply(lambda x: x[attr])
                    
                    # Get group and reference data
                    group_data = model_data[model_attr_values == attr_value]
                    ref_data = model_data[model_attr_values == reference_value]
                    
                    group_size = len(group_data)
                    ref_size = len(ref_data)
                    
                    comparison_name = f"{attr_value}_vs_{reference_value}_{model}"
                    
                    if group_size >= 30 and ref_size >= 30:
                        print(f"    ✓ {comparison_name}: Group={group_size}, Ref={ref_size}")
                        valid_comparisons += 1
                    else:
                        print(f"    ✗ {comparison_name}: Group={group_size}, Ref={ref_size} (FILTERED)")
                        filtered_comparisons += 1
        
        print(f"\n  SUMMARY for {attr}:")
        print(f"    Valid comparisons: {valid_comparisons}")
        print(f"    Filtered comparisons: {filtered_comparisons}")
        print(f"    Filter rate: {filtered_comparisons/(valid_comparisons+filtered_comparisons)*100:.1f}%")

if __name__ == "__main__":
    debug_statistical_parity()
    print("\n" + "="*80)
    debug_results_aggregation() 