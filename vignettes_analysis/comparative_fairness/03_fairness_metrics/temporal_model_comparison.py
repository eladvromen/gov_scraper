"""
Temporal Model Comparison - Testing significance of differences between models
This addresses the key question: Are there significant differences between 
Pre-Brexit and Post-Brexit models for the same groups?
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import numba
from numba import jit

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_utils import load_config, setup_logging

@jit(nopython=True, parallel=True)
def vectorized_model_comparison_test(post_rates: np.ndarray, post_sizes: np.ndarray,
                                    pre_rates: np.ndarray, pre_sizes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Test for significant differences between models for the same groups
    This is the proper test for temporal bias detection
    
    Args:
        post_rates: Post-Brexit model grant rates for each group
        post_sizes: Post-Brexit model sample sizes
        pre_rates: Pre-Brexit model grant rates for each group  
        pre_sizes: Pre-Brexit model sample sizes
        
    Returns:
        Tuple of (z_scores, p_values) for model differences
    """
    n = len(post_rates)
    z_scores = np.full(n, np.nan)
    p_values = np.full(n, np.nan)
    
    for i in numba.prange(n):
        if post_sizes[i] >= 30 and pre_sizes[i] >= 30:
            # Calculate pooled standard error for difference of proportions
            p1, n1 = post_rates[i], post_sizes[i]
            p2, n2 = pre_rates[i], pre_sizes[i]
            
            # Standard error for difference of proportions
            se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
            
            if se > 0:
                # Z-score for difference
                z_scores[i] = (p1 - p2) / se
                
                # P-value (two-tailed)
                z_abs = abs(z_scores[i])
                if z_abs > 8:
                    p_values[i] = 0.0
                else:
                    # Normal CDF approximation
                    p_values[i] = 2 * (1 - 0.5 * (1 + np.tanh(z_abs * np.sqrt(2/np.pi))))
    
    return z_scores, p_values

def calculate_temporal_bias_analysis(df: pd.DataFrame, config: Dict) -> Dict:
    """
    Calculate temporal bias by comparing models for the same groups
    This is the proper approach for detecting temporal bias
    """
    
    results = {}
    
    # Process each protected attribute
    for attr in config['protected_attributes']:
        attr_results = {}
        
        topics = df['topic'].unique()
        attr_values = df[attr].unique()
        
        for topic in topics:
            topic_data = df[df['topic'] == topic]
            
            if len(topic_data) < 60:  # Need both models
                continue
            
            topic_results = {}
            
            # For each group value, compare Post-Brexit vs Pre-Brexit
            for attr_value in attr_values:
                group_data = topic_data[topic_data[attr] == attr_value]
                
                # Get data for both models
                post_data = group_data[group_data['model'] == 'post_brexit']
                pre_data = group_data[group_data['model'] == 'pre_brexit']
                
                if len(post_data) >= 30 and len(pre_data) >= 30:
                    # Calculate grant rates for each model
                    post_rate = post_data['decision_binary'].mean()
                    pre_rate = pre_data['decision_binary'].mean()
                    
                    # Test for significant difference between models
                    z_scores, p_values = vectorized_model_comparison_test(
                        np.array([post_rate]), np.array([len(post_data)]),
                        np.array([pre_rate]), np.array([len(pre_data)])
                    )
                    
                    temporal_change = post_rate - pre_rate
                    
                    key = f"{attr_value}_temporal_change"
                    topic_results[key] = {
                        'attribute': attr,
                        'group': attr_value,
                        'topic': topic,
                        'post_brexit_rate': float(post_rate),
                        'pre_brexit_rate': float(pre_rate),
                        'temporal_change': float(temporal_change),
                        'temporal_change_pct': float(temporal_change * 100),
                        'post_brexit_size': int(len(post_data)),
                        'pre_brexit_size': int(len(pre_data)),
                        'z_score': float(z_scores[0]) if not np.isnan(z_scores[0]) else None,
                        'p_value': float(p_values[0]) if not np.isnan(p_values[0]) else None,
                        'is_significant': bool(abs(z_scores[0]) > 1.96) if not np.isnan(z_scores[0]) else False,
                        'change_direction': 'increase' if temporal_change > 0 else 'decrease'
                    }
            
            if topic_results:
                attr_results[topic] = topic_results
        
        if attr_results:
            results[attr] = attr_results
    
    return results

def calculate_intersectional_temporal_bias(df: pd.DataFrame, config: Dict) -> Dict:
    """Calculate temporal bias for intersectional groups"""
    
    # Add intersectional columns
    df['gender_x_religion'] = df['gender'] + '_x_' + df['religion']
    df['gender_x_country'] = df['gender'] + '_x_' + df['country']
    df['religion_x_country'] = df['religion'] + '_x_' + df['country']
    df['age_x_gender'] = df['age'].astype(str) + '_x_' + df['gender']
    df['religion_x_country_x_gender'] = df['religion'] + '_x_' + df['country'] + '_x_' + df['gender']
    
    results = {}
    
    for intersection in config['intersections']:
        intersection_results = {}
        
        topics = df['topic'].unique()
        intersection_values = df[intersection].unique()
        
        for topic in topics:
            topic_data = df[df['topic'] == topic]
            
            if len(topic_data) < 60:
                continue
            
            topic_results = {}
            
            # For each intersection value, compare models
            for intersection_value in intersection_values:
                group_data = topic_data[topic_data[intersection] == intersection_value]
                
                post_data = group_data[group_data['model'] == 'post_brexit']
                pre_data = group_data[group_data['model'] == 'pre_brexit']
                
                if len(post_data) >= 30 and len(pre_data) >= 30:
                    post_rate = post_data['decision_binary'].mean()
                    pre_rate = pre_data['decision_binary'].mean()
                    
                    # Test for significant difference
                    z_scores, p_values = vectorized_model_comparison_test(
                        np.array([post_rate]), np.array([len(post_data)]),
                        np.array([pre_rate]), np.array([len(pre_data)])
                    )
                    
                    temporal_change = post_rate - pre_rate
                    
                    key = f"{intersection_value}_temporal_change"
                    topic_results[key] = {
                        'intersection': intersection,
                        'group': intersection_value,
                        'topic': topic,
                        'post_brexit_rate': float(post_rate),
                        'pre_brexit_rate': float(pre_rate),
                        'temporal_change': float(temporal_change),
                        'temporal_change_pct': float(temporal_change * 100),
                        'post_brexit_size': int(len(post_data)),
                        'pre_brexit_size': int(len(pre_data)),
                        'z_score': float(z_scores[0]) if not np.isnan(z_scores[0]) else None,
                        'p_value': float(p_values[0]) if not np.isnan(p_values[0]) else None,
                        'is_significant': bool(abs(z_scores[0]) > 1.96) if not np.isnan(z_scores[0]) else False,
                        'change_direction': 'increase' if temporal_change > 0 else 'decrease'
                    }
            
            if topic_results:
                intersection_results[topic] = topic_results
        
        if intersection_results:
            results[intersection] = intersection_results
    
    return results

def calculate_differential_temporal_bias(df: pd.DataFrame, config: Dict) -> Dict:
    """
    Calculate differential temporal bias - are some groups changing more than others?
    This tests for interaction effects between time and protected attributes
    """
    
    results = {}
    
    for attr in config['protected_attributes']:
        reference_value = config['reference_groups'][attr]
        attr_results = {}
        
        topics = df['topic'].unique()
        attr_values = df[attr].unique()
        
        for topic in topics:
            topic_data = df[df['topic'] == topic]
            
            if len(topic_data) < 120:  # Need enough for all comparisons
                continue
            
            topic_results = {}
            
            # Calculate reference group temporal change
            ref_group = topic_data[topic_data[attr] == reference_value]
            ref_post = ref_group[ref_group['model'] == 'post_brexit']
            ref_pre = ref_group[ref_group['model'] == 'pre_brexit']
            
            if len(ref_post) >= 30 and len(ref_pre) >= 30:
                ref_temporal_change = ref_post['decision_binary'].mean() - ref_pre['decision_binary'].mean()
                
                # Compare each group's temporal change to reference
                for attr_value in attr_values:
                    if attr_value == reference_value:
                        continue
                    
                    group_data = topic_data[topic_data[attr] == attr_value]
                    group_post = group_data[group_data['model'] == 'post_brexit']
                    group_pre = group_data[group_data['model'] == 'pre_brexit']
                    
                    if len(group_post) >= 30 and len(group_pre) >= 30:
                        group_temporal_change = group_post['decision_binary'].mean() - group_pre['decision_binary'].mean()
                        
                        # Differential temporal bias = difference in temporal changes
                        differential_change = group_temporal_change - ref_temporal_change
                        
                        key = f"{attr_value}_vs_{reference_value}_differential"
                        topic_results[key] = {
                            'attribute': attr,
                            'group': attr_value,
                            'reference': reference_value,
                            'topic': topic,
                            'group_temporal_change': float(group_temporal_change),
                            'reference_temporal_change': float(ref_temporal_change),
                            'differential_temporal_bias': float(differential_change),
                            'differential_bias_pct': float(differential_change * 100),
                            'group_post_rate': float(group_post['decision_binary'].mean()),
                            'group_pre_rate': float(group_pre['decision_binary'].mean()),
                            'ref_post_rate': float(ref_post['decision_binary'].mean()),
                            'ref_pre_rate': float(ref_pre['decision_binary'].mean())
                        }
            
            if topic_results:
                attr_results[topic] = topic_results
        
        if attr_results:
            results[attr] = attr_results
    
    return results

def main():
    """Main function for temporal model comparison analysis"""
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / "config" / "analysis_config.yaml"
    processed_dir = base_dir / "data" / "processed"
    output_dir = base_dir / "outputs" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = base_dir / "outputs" / "logs" / "temporal_model_comparison.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(log_file))
    
    logger.info("Starting temporal model comparison analysis")
    
    # Load configuration and data
    config = load_config(config_path)
    
    with open(processed_dir / "tagged_records.json", 'r') as f:
        records = json.load(f)
    
    logger.info(f"Loaded {len(records)} records for temporal analysis")
    
    # Convert to DataFrame
    from statistical_parity import prepare_data_arrays
    df = prepare_data_arrays(records)
    
    logger.info("Calculating temporal bias analysis...")
    temporal_analysis = calculate_temporal_bias_analysis(df, config)
    
    # Count temporal comparisons
    total_temporal_comparisons = 0
    significant_temporal = 0
    for attr, attr_results in temporal_analysis.items():
        for topic, topic_results in attr_results.items():
            for comparison, result in topic_results.items():
                total_temporal_comparisons += 1
                if result.get('is_significant', False):
                    significant_temporal += 1
        logger.info(f"  {attr}: {sum(len(topic_results) for topic_results in attr_results.values())} temporal comparisons")
    
    logger.info("Calculating intersectional temporal bias...")
    intersectional_temporal = calculate_intersectional_temporal_bias(df, config)
    
    # Count intersectional temporal comparisons
    total_intersectional_temporal = 0
    significant_intersectional_temporal = 0
    for intersection, inter_results in intersectional_temporal.items():
        for topic, topic_results in inter_results.items():
            for comparison, result in topic_results.items():
                total_intersectional_temporal += 1
                if result.get('is_significant', False):
                    significant_intersectional_temporal += 1
        logger.info(f"  {intersection}: {sum(len(topic_results) for topic_results in inter_results.values())} intersectional temporal comparisons")
    
    logger.info("Calculating differential temporal bias...")
    differential_temporal = calculate_differential_temporal_bias(df, config)
    
    # Count differential comparisons
    total_differential = 0
    for attr, attr_results in differential_temporal.items():
        for topic, topic_results in attr_results.items():
            total_differential += len(topic_results)
    
    logger.info("Temporal Model Comparison Summary:")
    logger.info(f"  Individual temporal comparisons: {total_temporal_comparisons} ({significant_temporal} significant)")
    logger.info(f"  Intersectional temporal comparisons: {total_intersectional_temporal} ({significant_intersectional_temporal} significant)")
    logger.info(f"  Differential temporal bias comparisons: {total_differential}")
    
    # Save results
    temporal_results = {
        'individual_temporal_analysis': temporal_analysis,
        'intersectional_temporal_analysis': intersectional_temporal,
        'differential_temporal_analysis': differential_temporal,
        'summary': {
            'total_temporal_comparisons': total_temporal_comparisons,
            'significant_temporal_changes': significant_temporal,
            'temporal_significance_rate': significant_temporal / total_temporal_comparisons if total_temporal_comparisons > 0 else 0,
            'intersectional_temporal_comparisons': total_intersectional_temporal,
            'significant_intersectional_temporal': significant_intersectional_temporal,
            'differential_temporal_comparisons': total_differential
        }
    }
    
    temporal_file = output_dir / "temporal_model_comparison_results.json"
    with open(temporal_file, 'w') as f:
        json.dump(temporal_results, f, indent=2)
    
    logger.info(f"Saved temporal model comparison results to {temporal_file}")
    logger.info("Temporal model comparison analysis completed successfully!")
    
    return temporal_results

if __name__ == "__main__":
    results = main() 