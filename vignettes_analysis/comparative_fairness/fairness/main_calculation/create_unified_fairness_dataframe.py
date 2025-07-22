#!/usr/bin/env python3
"""
Create Unified Fairness DataFrame
Generates the exact format requested by user:
- pre_brexit_model_statistical_parity
- post_brexit_model_statistical_parity
- equal_opportunity_models_gap

With metadata for group sizes, protected attributes, and significance.
Updated to work with ALL PAIRS analysis results.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def load_metric_data() -> Tuple[Dict, Dict]:
    """Load all required metric data files"""
    
    metrics_dir = Path("../../../outputs/metrics")
    
    # Load statistical parity results (ALL PAIRS)
    sp_file = metrics_dir / "statistical_parity_all_pairs_results.json"
    if not sp_file.exists():
        print(f"Warning: {sp_file} not found, using empty data")
        sp_data = {}
    else:
        with open(sp_file, 'r') as f:
            sp_data = json.load(f)
    
    # Load error-based metrics results (ALL PAIRS)
    error_file = metrics_dir / "error_based_metrics_all_pairs_results.json"
    if not error_file.exists():
        print(f"Warning: {error_file} not found, using empty data")
        error_data = {}
    else:
        with open(error_file, 'r') as f:
            error_data = json.load(f)
    
    return sp_data, error_data

# Function removed - no longer needed with topic-granular approach

def create_unified_fairness_dataframe(sp_data: Dict, error_data: Dict) -> pd.DataFrame:
    """Create the unified fairness dataframe with all required columns - TOPIC GRANULAR"""
    
    rows = []
    
    # Process each protected attribute
    for attribute in sp_data.get('individual_attributes', {}):
        print(f"Processing attribute: {attribute}")
        
        # Get data for this attribute
        sp_attr_data = sp_data['individual_attributes'].get(attribute, {})
        error_attr_data = error_data.get('individual_attributes', {}).get(attribute, {})
        
        # Process each TOPIC separately to maintain granularity
        for topic in sp_attr_data:
            
            print(f"  Processing topic: {topic}")
            
            topic_sp_data = sp_attr_data[topic]
            topic_error_data = error_attr_data.get(topic, {})
            
            # Extract all comparisons for this topic
            topic_comparisons = set()
            
            # Get comparisons from statistical parity data
            for key in topic_sp_data:
                if key.endswith('_pre_brexit'):
                    base_comp = key[:-len('_pre_brexit')]
                    topic_comparisons.add(base_comp)
                elif key.endswith('_post_brexit'):
                    base_comp = key[:-len('_post_brexit')]
                    topic_comparisons.add(base_comp)
            
            # Get comparisons from error data
            for key in topic_error_data:
                if '_vs_' in key:
                    topic_comparisons.add(key)
            
            # Process each comparison for this topic
            for comparison in topic_comparisons:
                print(f"    Processing comparison: {comparison}")
                
                # Initialize row data
                row_data = {
                    'group_comparison': comparison,
                    'protected_attribute': attribute,
                    'topic': topic,
                    'pre_brexit_model_statistical_parity': None,
                    'post_brexit_model_statistical_parity': None,
                    'equal_opportunity_models_gap': None,
                    'pre_brexit_group_size': None,
                    'post_brexit_group_size': None,
                    'pre_brexit_reference_size': None,
                    'post_brexit_reference_size': None,
                    'pre_brexit_sp_significance': None,
                    'post_brexit_sp_significance': None,
                    'pre_brexit_sp_p_value': None,
                    'post_brexit_sp_p_value': None,
                    'equal_opportunity_sample_size': None,
                    'group_name': None,
                    'reference_group_name': None
                }
                
                # Extract group and reference names
                if '_vs_' in comparison:
                    group_name, reference_name = comparison.split('_vs_')
                    row_data['group_name'] = group_name
                    row_data['reference_group_name'] = reference_name
                
                # Find pre-Brexit statistical parity data for this specific topic
                pre_sp_key = f"{comparison}_pre_brexit"
                if pre_sp_key in topic_sp_data:
                    pre_sp_data = topic_sp_data[pre_sp_key]
                    row_data['pre_brexit_model_statistical_parity'] = pre_sp_data['sp_gap']
                    row_data['pre_brexit_group_size'] = pre_sp_data['group_size']
                    row_data['pre_brexit_reference_size'] = pre_sp_data['reference_size']
                    row_data['pre_brexit_sp_significance'] = pre_sp_data['is_significant']
                    row_data['pre_brexit_sp_p_value'] = pre_sp_data['p_value']
                
                # Find post-Brexit statistical parity data for this specific topic
                post_sp_key = f"{comparison}_post_brexit"
                if post_sp_key in topic_sp_data:
                    post_sp_data = topic_sp_data[post_sp_key]
                    row_data['post_brexit_model_statistical_parity'] = post_sp_data['sp_gap']
                    row_data['post_brexit_group_size'] = post_sp_data['group_size']
                    row_data['post_brexit_reference_size'] = post_sp_data['reference_size']
                    row_data['post_brexit_sp_significance'] = post_sp_data['is_significant']
                    row_data['post_brexit_sp_p_value'] = post_sp_data['p_value']
                
                # Find equal opportunity data for this specific topic
                if comparison in topic_error_data:
                    eo_data = topic_error_data[comparison]
                    row_data['equal_opportunity_models_gap'] = eo_data['equal_opportunity_gap']
                    row_data['equal_opportunity_sample_size'] = eo_data['total_sample_size']
                
                # Only add row if we have at least some data
                if any(row_data[col] is not None for col in ['pre_brexit_model_statistical_parity', 
                                                            'post_brexit_model_statistical_parity',
                                                            'equal_opportunity_models_gap']):
                    rows.append(row_data)
    
    return pd.DataFrame(rows)

def find_statistical_parity_data(attr_data: Dict, comparison: str, model: str) -> Optional[Dict]:
    """Find statistical parity data for a specific comparison and model - DEPRECATED"""
    # This function is now deprecated as we handle topic granularity in main function
    return None

def find_equal_opportunity_data(attr_data: Dict, comparison: str) -> Optional[Dict]:
    """Find equal opportunity data for a specific comparison - DEPRECATED"""
    # This function is now deprecated as we handle topic granularity in main function
    return None

def add_comparative_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Add comparative analysis columns"""
    
    # Add model comparison
    df['models_sp_difference'] = df['post_brexit_model_statistical_parity'] - df['pre_brexit_model_statistical_parity']
    
    # Add significance comparison
    df['pre_brexit_significant'] = df['pre_brexit_sp_significance'].fillna(False)
    df['post_brexit_significant'] = df['post_brexit_sp_significance'].fillna(False)
    df['both_models_significant'] = df['pre_brexit_significant'] & df['post_brexit_significant']
    df['significance_change'] = df['pre_brexit_significant'] != df['post_brexit_significant']
    
    # Add magnitude categories
    df['sp_difference_magnitude'] = pd.cut(
        df['models_sp_difference'].abs(),
        bins=[0, 0.05, 0.1, 0.2, 1.0],
        labels=['Small', 'Medium', 'Large', 'Very Large'],
        include_lowest=True
    )
    
    return df

def create_summary_statistics(df: pd.DataFrame) -> Dict:
    """Create summary statistics for the unified dataframe"""
    
    summary = {
        'total_comparisons': len(df),
        'protected_attributes': df['protected_attribute'].nunique(),
        'unique_topics': df['topic'].nunique(),
        'attributes_breakdown': df['protected_attribute'].value_counts().to_dict(),
        'topics_breakdown': df['topic'].value_counts().to_dict(),
        'pre_brexit_significant_count': df['pre_brexit_significant'].sum(),
        'post_brexit_significant_count': df['post_brexit_significant'].sum(),
        'both_significant_count': df['both_models_significant'].sum(),
        'significance_change_count': df['significance_change'].sum(),
        'mean_sp_difference': df['models_sp_difference'].mean(),
        'median_sp_difference': df['models_sp_difference'].median(),
        'sp_difference_std': df['models_sp_difference'].std(),
        'equal_opportunity_coverage': df['equal_opportunity_models_gap'].notna().sum(),
        'complete_records': df[['pre_brexit_model_statistical_parity', 
                               'post_brexit_model_statistical_parity', 
                               'equal_opportunity_models_gap']].notna().all(axis=1).sum()
    }
    
    return summary

def main():
    """Main function to create unified fairness dataframe"""
    
    print("=" * 80)
    print("CREATING UNIFIED FAIRNESS DATAFRAME (TOPIC GRANULAR)")
    print("=" * 80)
    
    # Load data
    print("Loading metric data...")
    sp_data, error_data = load_metric_data()
    
    # Check what data we have
    print(f"Statistical parity data loaded: {'individual_attributes' in sp_data}")
    print(f"Error-based metrics data loaded: {'individual_attributes' in error_data}")
    
    if 'individual_attributes' in sp_data:
        print(f"SP attributes: {list(sp_data['individual_attributes'].keys())}")
    if 'individual_attributes' in error_data:
        print(f"Error attributes: {list(error_data['individual_attributes'].keys())}")
    
    # Create unified dataframe
    print("Creating unified dataframe...")
    df = create_unified_fairness_dataframe(sp_data, error_data)
    
    # Add comparative analysis
    print("Adding comparative analysis...")
    df = add_comparative_analysis(df)
    
    # Create summary statistics
    print("Creating summary statistics...")
    summary = create_summary_statistics(df)
    
    # Save results
    output_dir = Path("../../../outputs/unified_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Save dataframe
    df.to_csv(output_dir / "unified_fairness_dataframe_topic_granular.csv", index=False)
    df.to_json(output_dir / "unified_fairness_dataframe_topic_granular.json", orient='records', indent=2)
    
    # Save summary
    with open(output_dir / "unified_analysis_summary_topic_granular.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print("UNIFIED DATAFRAME SUMMARY (TOPIC GRANULAR)")
    print("=" * 80)
    
    print(f"Total comparisons: {len(df)}")
    print(f"Protected attributes: {df['protected_attribute'].nunique()}")
    print(f"Unique topics: {df['topic'].nunique()}")
    print(f"Attributes: {list(df['protected_attribute'].unique())}")
    print(f"Topics: {list(df['topic'].unique())}")
    print(f"Complete records (all 3 metrics): {summary['complete_records']}")
    
    print("\n" + "=" * 80)
    print("DATAFRAME STRUCTURE")
    print("=" * 80)
    
    print("Columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    print("\n" + "=" * 80)
    print("SAMPLE DATA")
    print("=" * 80)
    
    # Show core columns
    core_columns = ['group_comparison', 'protected_attribute', 'topic',
                   'pre_brexit_model_statistical_parity', 
                   'post_brexit_model_statistical_parity',
                   'equal_opportunity_models_gap']
    
    print("Core columns sample:")
    print(df[core_columns].head(10))
    
    print("\n" + "=" * 80)
    print("SIGNIFICANCE ANALYSIS")
    print("=" * 80)
    
    print(f"Pre-Brexit significant: {summary['pre_brexit_significant_count']}")
    print(f"Post-Brexit significant: {summary['post_brexit_significant_count']}")
    print(f"Both models significant: {summary['both_significant_count']}")
    print(f"Significance pattern changed: {summary['significance_change_count']}")
    
    print("\n" + "=" * 80)
    print("STATISTICAL PARITY DIFFERENCE ANALYSIS")
    print("=" * 80)
    
    print(f"Mean SP difference: {summary['mean_sp_difference']:.4f}")
    print(f"Median SP difference: {summary['median_sp_difference']:.4f}")
    print(f"Standard deviation: {summary['sp_difference_std']:.4f}")
    
    print("\nMagnitude distribution:")
    print(df['sp_difference_magnitude'].value_counts())
    
    print("\n" + "=" * 80)
    print("FILES SAVED")
    print("=" * 80)
    
    print("✅ outputs/unified_analysis/unified_fairness_dataframe_topic_granular.csv")
    print("✅ outputs/unified_analysis/unified_fairness_dataframe_topic_granular.json")
    print("✅ outputs/unified_analysis/unified_analysis_summary_topic_granular.json")
    
    return df, summary

if __name__ == "__main__":
    df, summary = main() 