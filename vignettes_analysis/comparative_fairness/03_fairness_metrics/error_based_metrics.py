"""
Optimized Error-Based Fairness Metrics using numpy vectorization and numba
Includes Equal Opportunity, False Positive Rate, Equalized Odds L2, and Agreement Rate
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

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_utils import load_config, setup_logging

@jit(nopython=True, parallel=True)
def vectorized_conditional_metrics(post_decisions: np.ndarray, pre_decisions: np.ndarray,
                                   condition_mask: np.ndarray, group_indices: np.ndarray,
                                   ref_group_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate conditional metrics (TPR, FPR) efficiently using numba
    
    Args:
        post_decisions: Post-Brexit model decisions (1=GRANT, 0=DENY)
        pre_decisions: Pre-Brexit model decisions (1=GRANT, 0=DENY) - used as ground truth
        condition_mask: Boolean mask for condition (e.g., pre_decisions==1 for TPR)
        group_indices: Group membership indices
        ref_group_idx: Reference group index
    
    Returns:
        Tuple of (group_rates, ref_rates, sample_sizes)
    """
    unique_groups = np.unique(group_indices)
    n_groups = len(unique_groups)
    
    group_rates = np.full(n_groups, np.nan)
    ref_rates = np.full(n_groups, np.nan)
    sample_sizes = np.zeros(n_groups, dtype=np.int32)
    
    # Calculate reference group rate first
    ref_mask = (group_indices == ref_group_idx) & condition_mask
    if np.sum(ref_mask) > 0:
        ref_rate = np.mean(post_decisions[ref_mask])
    else:
        ref_rate = np.nan
    
    for i in numba.prange(n_groups):
        group_idx = unique_groups[i]
        group_mask = (group_indices == group_idx) & condition_mask
        
        sample_sizes[i] = np.sum(group_mask)
        
        if sample_sizes[i] >= 30:  # Minimum sample size
            group_rates[i] = np.mean(post_decisions[group_mask])
            ref_rates[i] = ref_rate
    
    return group_rates, ref_rates, sample_sizes

@jit(nopython=True)
def calculate_agreement_rate(post_decisions: np.ndarray, pre_decisions: np.ndarray,
                            group_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate agreement rate between models efficiently using numba
    
    Args:
        post_decisions: Post-Brexit model decisions
        pre_decisions: Pre-Brexit model decisions  
        group_indices: Group membership indices
    
    Returns:
        Tuple of (agreement_rates, sample_sizes)
    """
    unique_groups = np.unique(group_indices)
    n_groups = len(unique_groups)
    
    agreement_rates = np.zeros(n_groups)
    sample_sizes = np.zeros(n_groups, dtype=np.int32)
    
    for i in range(n_groups):
        group_mask = group_indices == unique_groups[i]
        
        post_group = post_decisions[group_mask]
        pre_group = pre_decisions[group_mask]
        
        sample_sizes[i] = len(post_group)
        if sample_sizes[i] > 0:
            agreement_rates[i] = np.mean(post_group == pre_group)
    
    return agreement_rates, sample_sizes

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

def calculate_error_based_metrics_vectorized(df: pd.DataFrame, config: Dict) -> Dict:
    """Calculate error-based fairness metrics using vectorized operations"""
    
    results = {}
    reference_groups = config['reference_groups']
    
    # Process individual protected attributes
    for attr in config['protected_attributes']:
        attr_results = {}
        reference_value = reference_groups[attr]
        
        topics = df['topic'].unique()
        
        for topic in topics:
            topic_data = df[df['topic'] == topic]
            
            if len(topic_data) < 60:
                continue
            
            topic_results = {}
            
            # Convert to numpy arrays for efficiency
            post_decisions = topic_data['post_decision'].values
            pre_decisions = topic_data['pre_decision'].values
            group_values = topic_data[attr].values
            
            # Create group indices
            unique_groups = np.unique(group_values)
            group_indices = np.zeros(len(group_values), dtype=np.int32)
            ref_group_idx = -1
            
            for i, group in enumerate(unique_groups):
                mask = group_values == group
                group_indices[mask] = i
                if group == reference_value:
                    ref_group_idx = i
            
            if ref_group_idx == -1:
                continue  # No reference group in this topic
            
            # Calculate Equal Opportunity (TPR given pre_decision=1)
            tpr_condition = pre_decisions == 1
            if np.sum(tpr_condition) >= 60:  # Enough positive cases
                tpr_rates, tpr_ref_rates, tpr_sizes = vectorized_conditional_metrics(
                    post_decisions, pre_decisions, tpr_condition, group_indices, ref_group_idx
                )
                
                # Calculate False Positive Rate (FPR given pre_decision=0)
                fpr_condition = pre_decisions == 0
                if np.sum(fpr_condition) >= 60:  # Enough negative cases
                    fpr_rates, fpr_ref_rates, fpr_sizes = vectorized_conditional_metrics(
                        post_decisions, pre_decisions, fpr_condition, group_indices, ref_group_idx
                    )
                    
                    # Calculate Agreement Rate
                    agr_rates, agr_sizes = calculate_agreement_rate(
                        post_decisions, pre_decisions, group_indices
                    )
                    
                    # Process results for each group
                    for i, group in enumerate(unique_groups):
                        if group == reference_value or np.isnan(tpr_rates[i]) or np.isnan(fpr_rates[i]):
                            continue
                        
                        if tpr_sizes[i] >= 30 and fpr_sizes[i] >= 30 and agr_sizes[i] >= 30:
                            # Calculate metrics
                            eo_gap = tpr_rates[i] - tpr_ref_rates[i]
                            fpr_gap = fpr_rates[i] - fpr_ref_rates[i]
                            eol2_gap = np.sqrt(eo_gap**2 + fpr_gap**2)
                            
                            key = f"{group}_vs_{reference_value}"
                            topic_results[key] = {
                                'attribute': attr,
                                'group': group,
                                'reference': reference_value,
                                'topic': topic,
                                'equal_opportunity_gap': float(eo_gap),
                                'false_positive_rate_gap': float(fpr_gap),
                                'equalized_odds_l2_gap': float(eol2_gap),
                                'group_tpr': float(tpr_rates[i]),
                                'reference_tpr': float(tpr_ref_rates[i]),
                                'group_fpr': float(fpr_rates[i]),
                                'reference_fpr': float(fpr_ref_rates[i]),
                                'agreement_rate': float(agr_rates[i]),
                                'tpr_sample_size': int(tpr_sizes[i]),
                                'fpr_sample_size': int(fpr_sizes[i]),
                                'total_sample_size': int(agr_sizes[i])
                            }
            
            if topic_results:
                attr_results[topic] = topic_results
        
        if attr_results:
            results[attr] = attr_results
    
    return results

def calculate_intersectional_error_metrics(df: pd.DataFrame, config: Dict) -> Dict:
    """Calculate error-based metrics for intersectional groups"""
    
    results = {}
    
    # Define reference groups for intersections
    intersection_refs = {
        'gender_x_religion': f"{config['reference_groups']['gender']}_x_{config['reference_groups']['religion']}",
        'gender_x_country': f"{config['reference_groups']['gender']}_x_{config['reference_groups']['country']}",
        'religion_x_country': f"{config['reference_groups']['religion']}_x_{config['reference_groups']['country']}",
        'age_x_gender': f"{config['reference_groups']['age']}_x_{config['reference_groups']['gender']}",
        'religion_x_country_x_gender': f"{config['reference_groups']['religion']}_x_{config['reference_groups']['country']}_x_{config['reference_groups']['gender']}"
    }
    
    for intersection in config['intersections']:
        intersection_results = {}
        reference_value = intersection_refs[intersection]
        
        topics = df['topic'].unique()
        
        for topic in topics:
            topic_data = df[df['topic'] == topic]
            
            if len(topic_data) < 60:
                continue
            
            topic_results = {}
            
            # Convert to numpy arrays
            post_decisions = topic_data['post_decision'].values
            pre_decisions = topic_data['pre_decision'].values
            group_values = topic_data[intersection].values
            
            # Create group indices
            unique_groups = np.unique(group_values)
            group_indices = np.zeros(len(group_values), dtype=np.int32)
            ref_group_idx = -1
            
            for i, group in enumerate(unique_groups):
                mask = group_values == group
                group_indices[mask] = i
                if group == reference_value:
                    ref_group_idx = i
            
            if ref_group_idx == -1:
                continue
            
            # Calculate metrics similar to individual attributes
            tpr_condition = pre_decisions == 1
            fpr_condition = pre_decisions == 0
            
            if np.sum(tpr_condition) >= 60 and np.sum(fpr_condition) >= 60:
                tpr_rates, tpr_ref_rates, tpr_sizes = vectorized_conditional_metrics(
                    post_decisions, pre_decisions, tpr_condition, group_indices, ref_group_idx
                )
                
                fpr_rates, fpr_ref_rates, fpr_sizes = vectorized_conditional_metrics(
                    post_decisions, pre_decisions, fpr_condition, group_indices, ref_group_idx
                )
                
                agr_rates, agr_sizes = calculate_agreement_rate(
                    post_decisions, pre_decisions, group_indices
                )
                
                for i, group in enumerate(unique_groups):
                    if group == reference_value or np.isnan(tpr_rates[i]) or np.isnan(fpr_rates[i]):
                        continue
                    
                    if tpr_sizes[i] >= 30 and fpr_sizes[i] >= 30 and agr_sizes[i] >= 30:
                        eo_gap = tpr_rates[i] - tpr_ref_rates[i]
                        fpr_gap = fpr_rates[i] - fpr_ref_rates[i]
                        eol2_gap = np.sqrt(eo_gap**2 + fpr_gap**2)
                        
                        key = f"{group}_vs_{reference_value}"
                        topic_results[key] = {
                            'intersection': intersection,
                            'group': group,
                            'reference': reference_value,
                            'topic': topic,
                            'equal_opportunity_gap': float(eo_gap),
                            'false_positive_rate_gap': float(fpr_gap),
                            'equalized_odds_l2_gap': float(eol2_gap),
                            'group_tpr': float(tpr_rates[i]),
                            'reference_tpr': float(tpr_ref_rates[i]),
                            'group_fpr': float(fpr_rates[i]),
                            'reference_fpr': float(fpr_ref_rates[i]),
                            'agreement_rate': float(agr_rates[i]),
                            'tpr_sample_size': int(tpr_sizes[i]),
                            'fpr_sample_size': int(fpr_sizes[i]),
                            'total_sample_size': int(agr_sizes[i])
                        }
            
            if topic_results:
                intersection_results[topic] = topic_results
        
        if intersection_results:
            results[intersection] = intersection_results
    
    return results

def main():
    """Main function for optimized error-based metrics calculation"""
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / "config" / "analysis_config.yaml"
    processed_dir = base_dir / "data" / "processed"
    output_dir = base_dir / "outputs" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = base_dir / "outputs" / "logs" / "error_based_metrics.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(log_file))
    
    logger.info("Starting optimized error-based metrics calculation")
    
    # Load configuration and data
    config = load_config(config_path)
    
    with open(processed_dir / "tagged_records.json", 'r') as f:
        records = json.load(f)
    
    logger.info(f"Loaded {len(records)} records for error-based metrics analysis")
    
    # Prepare matched model comparison data
    logger.info("Preparing matched model comparison data...")
    df = prepare_model_comparison_data(records)
    logger.info(f"Created {len(df)} matched pairs for comparison")
    
    # Calculate individual attribute error-based metrics
    logger.info("Calculating individual attribute error-based metrics...")
    individual_metrics = calculate_error_based_metrics_vectorized(df, config)
    
    # Log individual results summary
    total_individual_comparisons = 0
    for attr, attr_results in individual_metrics.items():
        attr_comparisons = sum(len(topic_results) for topic_results in attr_results.values())
        total_individual_comparisons += attr_comparisons
        logger.info(f"  {attr}: {attr_comparisons} comparisons")
    
    # Calculate intersectional error-based metrics
    logger.info("Calculating intersectional error-based metrics...")
    intersectional_metrics = calculate_intersectional_error_metrics(df, config)
    
    # Log intersectional results summary
    total_intersectional_comparisons = 0
    for intersection, inter_results in intersectional_metrics.items():
        inter_comparisons = sum(len(topic_results) for topic_results in inter_results.values())
        total_intersectional_comparisons += inter_comparisons
        logger.info(f"  {intersection}: {inter_comparisons} comparisons")
    
    # Overall summary
    total_comparisons = total_individual_comparisons + total_intersectional_comparisons
    
    logger.info("Error-Based Metrics Analysis Summary:")
    logger.info(f"  Total comparisons: {total_comparisons}")
    logger.info(f"  Individual attribute comparisons: {total_individual_comparisons}")
    logger.info(f"  Intersectional comparisons: {total_intersectional_comparisons}")
    
    # Save results
    error_metrics_results = {
        'individual_attributes': individual_metrics,
        'intersectional_groups': intersectional_metrics,
        'summary': {
            'total_comparisons': total_comparisons,
            'individual_comparisons': total_individual_comparisons,
            'intersectional_comparisons': total_intersectional_comparisons,
            'matched_pairs': len(df)
        },
        'configuration': {
            'min_group_size': config['min_group_size'],
            'reference_groups': config['reference_groups']
        }
    }
    
    # Save comprehensive results
    metrics_file = output_dir / "error_based_metrics_results.json"
    with open(metrics_file, 'w') as f:
        json.dump(error_metrics_results, f, indent=2)
    
    logger.info(f"Saved error-based metrics results to {metrics_file}")
    logger.info("Error-based metrics calculation completed successfully!")
    
    return error_metrics_results

if __name__ == "__main__":
    results = main() 