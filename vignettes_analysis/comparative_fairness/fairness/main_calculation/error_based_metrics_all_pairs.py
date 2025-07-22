
#!/usr/bin/env python3
"""
Error-Based Fairness Metrics with ALL PAIRWISE COMPARISONS
Includes Equal Opportunity, False Positive Rate, Equalized Odds L2, and Agreement Rate
Using ALL possible group pairs instead of just reference groups.
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import numba
from numba import jit
from collections import defaultdict
import itertools

def json_serial(obj):
    """JSON serializer for numpy data types"""
    if isinstance(obj, (np.integer, np.floating, np.ndarray)):
        return obj.tolist() if hasattr(obj, 'tolist') else int(obj)
    raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_utils import load_config, setup_logging

@jit(nopython=True, parallel=True)
def vectorized_conditional_metrics(post_decisions: np.ndarray, pre_decisions: np.ndarray,
                                   condition_mask: np.ndarray, group_indices: np.ndarray,
                                   group1_idx: int, group2_idx: int) -> Tuple[float, float, int, int]:
    """
    Calculate conditional metrics (TPR, FPR) efficiently using numba for two specific groups
    
    Args:
        post_decisions: Post-Brexit model decisions (1=GRANT, 0=DENY)
        pre_decisions: Pre-Brexit model decisions (1=GRANT, 0=DENY) - used as ground truth
        condition_mask: Boolean mask for condition (e.g., pre_decisions==1 for TPR)
        group_indices: Group membership indices
        group1_idx: First group index
        group2_idx: Second group index
    
    Returns:
        Tuple of (group1_rate, group2_rate, group1_size, group2_size)
    """
    
    # Group 1 calculations
    group1_mask = (group_indices == group1_idx) & condition_mask
    group1_size = np.sum(group1_mask)
    group1_rate = np.nan
    if group1_size > 0:
        group1_rate = np.mean(post_decisions[group1_mask])
    
    # Group 2 calculations
    group2_mask = (group_indices == group2_idx) & condition_mask
    group2_size = np.sum(group2_mask)
    group2_rate = np.nan
    if group2_size > 0:
        group2_rate = np.mean(post_decisions[group2_mask])
    
    return group1_rate, group2_rate, group1_size, group2_size

@jit(nopython=True)
def calculate_agreement_rate(post_decisions: np.ndarray, pre_decisions: np.ndarray,
                            group_indices: np.ndarray, group1_idx: int, group2_idx: int) -> Tuple[float, float, int, int]:
    """
    Calculate agreement rate between models for two specific groups
    
    Args:
        post_decisions: Post-Brexit model decisions
        pre_decisions: Pre-Brexit model decisions
        group_indices: Group membership indices
        group1_idx: First group index
        group2_idx: Second group index
    
    Returns:
        Tuple of (group1_agreement, group2_agreement, group1_size, group2_size)
    """
    
    # Group 1 calculations
    group1_mask = group_indices == group1_idx
    group1_size = np.sum(group1_mask)
    group1_agreement = np.nan
    if group1_size > 0:
        group1_agreement = np.mean(post_decisions[group1_mask] == pre_decisions[group1_mask])
    
    # Group 2 calculations
    group2_mask = group_indices == group2_idx
    group2_size = np.sum(group2_mask)
    group2_agreement = np.nan
    if group2_size > 0:
        group2_agreement = np.mean(post_decisions[group2_mask] == pre_decisions[group2_mask])
    
    return group1_agreement, group2_agreement, group1_size, group2_size

def prepare_model_comparison_data(records: List[Dict]) -> pd.DataFrame:
    """Prepare data for model comparison analysis"""
    
    # Separate by model and create matched pairs
    post_records = [r for r in records if r['model'] == 'post_brexit']
    pre_records = [r for r in records if r['model'] == 'pre_brexit']
    
    # Create matched pairs by sample_id and topic
    matched_data = []
    
    # Create lookup for pre-Brexit records
    pre_lookup = {}
    for record in pre_records:
        key = (record['sample_id'], record['topic'])
        pre_lookup[key] = record
    
    # Match with post-Brexit records
    for post_record in post_records:
        key = (post_record['sample_id'], post_record['topic'])
        if key in pre_lookup:
            pre_record = pre_lookup[key]
            
            matched_data.append({
                'topic': post_record['topic'],
                'sample_id': post_record['sample_id'],
                'post_decision': 1 if post_record['decision'] == 'GRANT' else 0,
                'pre_decision': 1 if pre_record['decision'] == 'GRANT' else 0,
                'country': post_record['protected_attributes']['country'],
                'age': post_record['protected_attributes']['age'],
                'religion': post_record['protected_attributes']['religion'],
                'gender': post_record['protected_attributes']['gender']
            })
    
    df = pd.DataFrame(matched_data)
    
    # Create intersectional columns
    df['gender_x_religion'] = df['gender'] + '_x_' + df['religion']
    df['gender_x_country'] = df['gender'] + '_x_' + df['country']
    df['religion_x_country'] = df['religion'] + '_x_' + df['country']
    df['age_x_gender'] = df['age'].astype(str) + '_x_' + df['gender']
    df['religion_x_country_x_gender'] = df['religion'] + '_x_' + df['country'] + '_x_' + df['gender']
    
    return df

def calculate_error_based_metrics_all_pairs(df: pd.DataFrame, config: Dict) -> Dict:
    """Calculate error-based fairness metrics using ALL PAIRWISE COMPARISONS"""
    
    results = {}
    
    # Process individual protected attributes
    for attr in config['protected_attributes']:
        print(f"Processing {attr} attribute...")
        attr_results = {}
        
        topics = df['topic'].unique()
        attr_values = df[attr].unique()
        
        # Use all attribute values - filter at comparison level instead
        valid_attr_values = list(attr_values)
        
        print(f"  Processing {attr} values: {len(valid_attr_values)}")
        
        for topic in topics:
            topic_data = df[df['topic'] == topic]
            
            if len(topic_data) < 60:  # Need enough matched pairs for reliable TPR/FPR calculations
                continue
            
            topic_results = {}
            
            # Convert to numpy arrays for efficiency
            post_decisions = topic_data['post_decision'].values
            pre_decisions = topic_data['pre_decision'].values
            group_values = topic_data[attr].values
            
            # Create group indices
            unique_groups = np.unique(group_values)
            group_indices = np.zeros(len(group_values), dtype=np.int32)
            group_name_map = {}
            
            for i, group in enumerate(unique_groups):
                mask = group_values == group
                group_indices[mask] = i
                group_name_map[i] = group
            
            # Generate ALL PAIRWISE COMBINATIONS
            for group1_idx, group2_idx in itertools.combinations(range(len(unique_groups)), 2):
                group1_name = group_name_map[group1_idx]
                group2_name = group_name_map[group2_idx]
                
                # Skip if either group is not in valid list
                if group1_name not in valid_attr_values or group2_name not in valid_attr_values:
                    continue
                
                # Calculate Equal Opportunity (TPR given pre_decision=1)
                tpr_condition = pre_decisions == 1
                tpr1_rate, tpr2_rate, tpr1_size, tpr2_size = vectorized_conditional_metrics(
                    post_decisions, pre_decisions, tpr_condition, group_indices, group1_idx, group2_idx
                )
                
                # Calculate False Positive Rate (FPR given pre_decision=0)
                fpr_condition = pre_decisions == 0
                fpr1_rate, fpr2_rate, fpr1_size, fpr2_size = vectorized_conditional_metrics(
                    post_decisions, pre_decisions, fpr_condition, group_indices, group1_idx, group2_idx
                )
                
                # Calculate Agreement Rate
                agr1_rate, agr2_rate, agr1_size, agr2_size = calculate_agreement_rate(
                    post_decisions, pre_decisions, group_indices, group1_idx, group2_idx
                )
                
                # Check minimum sample sizes - need enough for each metric calculation
                # TPR: need 30+ samples where pre_decision=1 for each group
                # FPR: need 30+ samples where pre_decision=0 for each group  
                # This ensures reliable statistics for each comparison
                if (tpr1_size >= 30 and tpr2_size >= 30 and 
                    fpr1_size >= 30 and fpr2_size >= 30):
                    
                    # Calculate metrics (group1 vs group2)
                    eo_gap = tpr1_rate - tpr2_rate
                    fpr_gap = fpr1_rate - fpr2_rate
                    eol2_gap = np.sqrt(eo_gap**2 + fpr_gap**2)
                    
                    key = f"{group1_name}_vs_{group2_name}"
                    topic_results[key] = {
                        'attribute': attr,
                        'group': group1_name,
                        'reference': group2_name,
                        'topic': topic,
                        'equal_opportunity_gap': float(eo_gap),
                        'false_positive_rate_gap': float(fpr_gap),
                        'equalized_odds_l2_gap': float(eol2_gap),
                        'group_tpr': float(tpr1_rate),
                        'reference_tpr': float(tpr2_rate),
                        'group_fpr': float(fpr1_rate),
                        'reference_fpr': float(fpr2_rate),
                        'agreement_rate': float(agr1_rate),
                        'tpr_sample_size': int(tpr1_size),
                        'fpr_sample_size': int(fpr1_size),
                        'total_sample_size': int(agr1_size)
                    }
                    
                    # Also create the reverse comparison (group2 vs group1)
                    reverse_key = f"{group2_name}_vs_{group1_name}"
                    topic_results[reverse_key] = {
                        'attribute': attr,
                        'group': group2_name,
                        'reference': group1_name,
                        'topic': topic,
                        'equal_opportunity_gap': float(-eo_gap),
                        'false_positive_rate_gap': float(-fpr_gap),
                        'equalized_odds_l2_gap': float(eol2_gap),
                        'group_tpr': float(tpr2_rate),
                        'reference_tpr': float(tpr1_rate),
                        'group_fpr': float(fpr2_rate),
                        'reference_fpr': float(fpr1_rate),
                        'agreement_rate': float(agr2_rate),
                        'tpr_sample_size': int(tpr2_size),
                        'fpr_sample_size': int(fpr2_size),
                        'total_sample_size': int(agr2_size)
                    }
            
            if topic_results:
                attr_results[topic] = topic_results
                print(f"    Topic '{topic}': {len(topic_results)} comparisons")
        
        if attr_results:
            results[attr] = attr_results
            total_comparisons = sum(len(topic_data) for topic_data in attr_results.values())
            print(f"  Total {attr} comparisons: {total_comparisons}")
    
    return results

def calculate_intersectional_error_metrics_all_pairs(df: pd.DataFrame, config: Dict) -> Dict:
    """Calculate error-based metrics for intersectional groups using ALL PAIRWISE COMPARISONS"""
    
    results = {}
    
    for intersection in config['intersections']:
        print(f"Processing {intersection} intersection...")
        intersection_results = {}
        
        topics = df['topic'].unique()
        intersection_values = df[intersection].unique()
        
        # Use all intersection values - filter at comparison level instead
        valid_intersection_values = list(intersection_values)
        
        print(f"  Processing {intersection} values: {len(valid_intersection_values)}")
        
        for topic in topics:
            topic_data = df[df['topic'] == topic]
            
            if len(topic_data) < 60:  # Need enough matched pairs for reliable TPR/FPR calculations
                continue
            
            topic_results = {}
            
            # Convert to numpy arrays
            post_decisions = topic_data['post_decision'].values
            pre_decisions = topic_data['pre_decision'].values
            group_values = topic_data[intersection].values
            
            # Create group indices
            unique_groups = np.unique(group_values)
            group_indices = np.zeros(len(group_values), dtype=np.int32)
            group_name_map = {}
            
            for i, group in enumerate(unique_groups):
                mask = group_values == group
                group_indices[mask] = i
                group_name_map[i] = group
            
            # Generate ALL PAIRWISE COMBINATIONS
            for group1_idx, group2_idx in itertools.combinations(range(len(unique_groups)), 2):
                group1_name = group_name_map[group1_idx]
                group2_name = group_name_map[group2_idx]
                
                # Skip if either group is not in valid list
                if group1_name not in valid_intersection_values or group2_name not in valid_intersection_values:
                    continue
                
                # Calculate metrics similar to individual attributes
                tpr_condition = pre_decisions == 1
                tpr1_rate, tpr2_rate, tpr1_size, tpr2_size = vectorized_conditional_metrics(
                    post_decisions, pre_decisions, tpr_condition, group_indices, group1_idx, group2_idx
                )
                
                fpr_condition = pre_decisions == 0
                fpr1_rate, fpr2_rate, fpr1_size, fpr2_size = vectorized_conditional_metrics(
                    post_decisions, pre_decisions, fpr_condition, group_indices, group1_idx, group2_idx
                )
                
                agr1_rate, agr2_rate, agr1_size, agr2_size = calculate_agreement_rate(
                    post_decisions, pre_decisions, group_indices, group1_idx, group2_idx
                )
                
                # Check minimum sample sizes - need enough for each metric calculation
                # TPR: need 30+ samples where pre_decision=1 for each group
                # FPR: need 30+ samples where pre_decision=0 for each group  
                # This ensures reliable statistics for each comparison
                if (tpr1_size >= 30 and tpr2_size >= 30 and 
                    fpr1_size >= 30 and fpr2_size >= 30):
                    
                    eo_gap = tpr1_rate - tpr2_rate
                    fpr_gap = fpr1_rate - fpr2_rate
                    eol2_gap = np.sqrt(eo_gap**2 + fpr_gap**2)
                    
                    key = f"{group1_name}_vs_{group2_name}"
                    topic_results[key] = {
                        'intersection': intersection,
                        'group': group1_name,
                        'reference': group2_name,
                        'topic': topic,
                        'equal_opportunity_gap': float(eo_gap),
                        'false_positive_rate_gap': float(fpr_gap),
                        'equalized_odds_l2_gap': float(eol2_gap),
                        'group_tpr': float(tpr1_rate),
                        'reference_tpr': float(tpr2_rate),
                        'group_fpr': float(fpr1_rate),
                        'reference_fpr': float(fpr2_rate),
                        'agreement_rate': float(agr1_rate),
                        'tpr_sample_size': int(tpr1_size),
                        'fpr_sample_size': int(fpr1_size),
                        'total_sample_size': int(agr1_size)
                    }
                    
                    # Also create the reverse comparison
                    reverse_key = f"{group2_name}_vs_{group1_name}"
                    topic_results[reverse_key] = {
                        'intersection': intersection,
                        'group': group2_name,
                        'reference': group1_name,
                        'topic': topic,
                        'equal_opportunity_gap': float(-eo_gap),
                        'false_positive_rate_gap': float(-fpr_gap),
                        'equalized_odds_l2_gap': float(eol2_gap),
                        'group_tpr': float(tpr2_rate),
                        'reference_tpr': float(tpr1_rate),
                        'group_fpr': float(fpr2_rate),
                        'reference_fpr': float(fpr1_rate),
                        'agreement_rate': float(agr2_rate),
                        'tpr_sample_size': int(tpr2_size),
                        'fpr_sample_size': int(fpr2_size),
                        'total_sample_size': int(agr2_size)
                    }
            
            if topic_results:
                intersection_results[topic] = topic_results
                print(f"    Topic '{topic}': {len(topic_results)} comparisons")
        
        if intersection_results:
            results[intersection] = intersection_results
            total_comparisons = sum(len(topic_data) for topic_data in intersection_results.values())
            print(f"  Total {intersection} comparisons: {total_comparisons}")
    
    return results

def calculate_overall_agreement_rate(df: pd.DataFrame) -> float:
    """Calculate overall agreement rate across all matched pairs"""
    return (df['post_decision'] == df['pre_decision']).mean()

def main():
    """Main function for ALL PAIRWISE error-based metrics calculation"""
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / "config" / "analysis_config.yaml"
    processed_dir = base_dir / "data" / "processed"
    output_dir = base_dir / "outputs" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = base_dir / "outputs" / "logs" / "error_based_metrics_all_pairs.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(log_file))
    
    logger.info("Starting ALL PAIRWISE error-based metrics calculation")
    print("=" * 80)
    print("ALL PAIRWISE ERROR-BASED METRICS CALCULATION")
    print("=" * 80)
    
    # Load configuration and data
    config = load_config(config_path)
    
    with open(processed_dir / "tagged_records.json", 'r') as f:
        records = json.load(f)
    
    logger.info(f"Loaded {len(records)} records for error-based metrics analysis")
    print(f"Loaded {len(records)} records")
    
    # Prepare matched model comparison data
    logger.info("Preparing matched model comparison data...")
    print("Preparing matched model comparison data...")
    df = prepare_model_comparison_data(records)
    logger.info(f"Created {len(df)} matched pairs for comparison")
    print(f"Created {len(df)} matched pairs")
    
    # Calculate individual attribute error metrics with all pairs
    logger.info("Calculating individual attribute error-based metrics (ALL PAIRS)...")
    print("\nCalculating individual attribute error-based metrics (ALL PAIRS)...")
    individual_error = calculate_error_based_metrics_all_pairs(df, config)
    
    # Calculate intersectional error metrics with all pairs
    logger.info("Calculating intersectional error-based metrics (ALL PAIRS)...")
    print("\nCalculating intersectional error-based metrics (ALL PAIRS)...")
    intersectional_error = calculate_intersectional_error_metrics_all_pairs(df, config)
    
    # Calculate overall agreement rate
    overall_agreement = calculate_overall_agreement_rate(df)
    
    # Log results summary
    total_individual_comparisons = 0
    for attr_data in individual_error.values():
        for topic_data in attr_data.values():
            total_individual_comparisons += len(topic_data)
    
    total_intersectional_comparisons = 0
    for intersection_data in intersectional_error.values():
        for topic_data in intersection_data.values():
            total_intersectional_comparisons += len(topic_data)
    
    total_comparisons = total_individual_comparisons + total_intersectional_comparisons
    
    logger.info(f"Error-based Metrics Results Summary:")
    logger.info(f"  Total comparisons: {total_comparisons}")
    logger.info(f"  Individual comparisons: {total_individual_comparisons}")
    logger.info(f"  Intersectional comparisons: {total_intersectional_comparisons}")
    logger.info(f"  Overall agreement rate: {overall_agreement:.3f}")
    
    print(f"\nResults Summary:")
    print(f"  Total comparisons: {total_comparisons}")
    print(f"  Individual comparisons: {total_individual_comparisons}")
    print(f"  Intersectional comparisons: {total_intersectional_comparisons}")
    print(f"  Overall agreement rate: {overall_agreement:.3f}")
    
    # Save results
    error_results = {
        'individual_attributes': individual_error,
        'intersectional_groups': intersectional_error,
        'summary': {
            'total_comparisons': total_comparisons,
            'individual_comparisons': total_individual_comparisons,
            'intersectional_comparisons': total_intersectional_comparisons,
            'matched_pairs': len(df),
            'overall_agreement_rate': overall_agreement
        },
        'configuration': {
            'min_group_size': config['min_group_size'],
            'comparison_method': 'all_pairwise',
            'methodology': 'Pre-Brexit decisions used as ground truth baseline'
        }
    }
    
    # Save comprehensive results
    error_file = output_dir / "error_based_metrics_all_pairs_results.json"
    with open(error_file, 'w') as f:
        json.dump(error_results, f, indent=2, default=json_serial)
    
    logger.info(f"Saved error-based metrics results to {error_file}")
    print(f"\nSaved results to: {error_file}")
    print("ALL PAIRWISE error-based metrics calculation completed successfully!")
    
    return error_results

if __name__ == "__main__":
    main() 