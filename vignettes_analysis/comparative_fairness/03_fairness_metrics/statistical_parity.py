"""
Optimized Statistical Parity calculation using numpy vectorization and numba
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

def json_serial(obj):
    """JSON serializer for numpy data types"""
    if isinstance(obj, (np.integer, np.floating, np.ndarray)):
        return obj.tolist() if hasattr(obj, 'tolist') else int(obj)
    raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_utils import load_config, setup_logging, get_reference_groups

@jit(nopython=True, parallel=True)
def vectorized_z_test(p1_array: np.ndarray, n1_array: np.ndarray, 
                      p2_array: np.ndarray, n2_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized z-test calculation using numba for speed
    
    Args:
        p1_array: Array of group proportions
        n1_array: Array of group sample sizes
        p2_array: Array of reference proportions  
        n2_array: Array of reference sample sizes
    
    Returns:
        Tuple of (z_scores, p_values)
    """
    n = len(p1_array)
    z_scores = np.full(n, np.nan)
    p_values = np.full(n, np.nan)
    
    for i in numba.prange(n):
        if n1_array[i] >= 30 and n2_array[i] >= 30:
            # Pooled proportion
            p_pooled = (p1_array[i] * n1_array[i] + p2_array[i] * n2_array[i]) / (n1_array[i] + n2_array[i])
            
            # Standard error
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1_array[i] + 1/n2_array[i]))
            
            if se > 0:
                # Z-score
                z_scores[i] = (p1_array[i] - p2_array[i]) / se
                
                # P-value (two-tailed) - approximation for numba compatibility
                z_abs = abs(z_scores[i])
                if z_abs > 8:  # Very large z-score
                    p_values[i] = 0.0
                else:
                    # Normal CDF approximation for numba
                    p_values[i] = 2 * (1 - 0.5 * (1 + np.tanh(z_abs * np.sqrt(2/np.pi))))
    
    return z_scores, p_values

@jit(nopython=True)
def calculate_grant_rates(decisions: np.ndarray, group_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate grant rates efficiently using numba
    
    Args:
        decisions: Array of binary decisions (1=GRANT, 0=DENY)
        group_indices: Array indicating group membership
    
    Returns:
        Tuple of (grant_rates, sample_sizes)
    """
    unique_groups = np.unique(group_indices)
    n_groups = len(unique_groups)
    
    grant_rates = np.zeros(n_groups)
    sample_sizes = np.zeros(n_groups, dtype=np.int32)
    
    for i in range(n_groups):
        group_mask = group_indices == unique_groups[i]
        group_decisions = decisions[group_mask]
        
        sample_sizes[i] = len(group_decisions)
        if sample_sizes[i] > 0:
            grant_rates[i] = np.mean(group_decisions)
    
    return grant_rates, sample_sizes

def prepare_data_arrays(records: List[Dict]) -> pd.DataFrame:
    """Convert records to optimized pandas DataFrame for vectorized operations"""
    
    # Extract key fields into lists
    data = {
        'topic': [],
        'model': [],
        'decision_binary': [],  # 1 for GRANT, 0 for DENY/INCONCLUSIVE
        'country': [],
        'age': [],
        'religion': [],
        'gender': []
    }
    
    for record in records:
        data['topic'].append(record['topic'])
        data['model'].append(record['model'])
        data['decision_binary'].append(1 if record['decision'] == 'GRANT' else 0)
        
        attrs = record['protected_attributes']
        data['country'].append(attrs['country'])
        data['age'].append(attrs['age'])
        data['religion'].append(attrs['religion'])
        data['gender'].append(attrs['gender'])
    
    df = pd.DataFrame(data)
    
    # Create numeric group encodings for efficiency
    for attr in ['country', 'age', 'religion', 'gender', 'topic', 'model']:
        df[f'{attr}_code'] = pd.Categorical(df[attr]).codes
    
    return df

def calculate_statistical_parity_vectorized(df: pd.DataFrame, config: Dict) -> Dict:
    """Calculate Statistical Parity using vectorized operations"""
    
    results = {}
    reference_groups = config['reference_groups']
    
    # Process each protected attribute
    for attr in config['protected_attributes']:
        attr_results = {}
        reference_value = reference_groups[attr]
        
        # Get unique topics and values for this attribute
        topics = df['topic'].unique()
        attr_values = df[attr].unique()
        
        for topic in topics:
            topic_data = df[df['topic'] == topic]
            
            # Skip if not enough data
            if len(topic_data) < 60:  # At least 30 per model
                continue
            
            topic_results = {}
            
            for attr_value in attr_values:
                if attr_value == reference_value:
                    continue
                
                # Calculate for each model
                for model in ['post_brexit', 'pre_brexit']:
                    model_data = topic_data[topic_data['model'] == model]
                    
                    # Get group and reference data
                    group_data = model_data[model_data[attr] == attr_value]
                    ref_data = model_data[model_data[attr] == reference_value]
                    
                    if len(group_data) >= 30 and len(ref_data) >= 30:
                        # Calculate grant rates
                        group_rate = group_data['decision_binary'].mean()
                        ref_rate = ref_data['decision_binary'].mean()
                        
                        # Calculate z-test using vectorized function
                        z_scores, p_values = vectorized_z_test(
                            np.array([group_rate]), np.array([len(group_data)]),
                            np.array([ref_rate]), np.array([len(ref_data)])
                        )
                        
                        key = f"{attr_value}_vs_{reference_value}_{model}"
                        topic_results[key] = {
                            'attribute': attr,
                            'group': attr_value,
                            'reference': reference_value,
                            'model': model,
                            'topic': topic,
                            'group_grant_rate': float(group_rate),
                            'reference_grant_rate': float(ref_rate),
                            'sp_gap': float(group_rate - ref_rate),
                            'group_size': int(len(group_data)),
                            'reference_size': int(len(ref_data)),
                            'z_score': float(z_scores[0]) if not np.isnan(z_scores[0]) else None,
                            'p_value': float(p_values[0]) if not np.isnan(p_values[0]) else None,
                            'is_significant': bool(abs(z_scores[0]) > 1.96) if not np.isnan(z_scores[0]) else False
                        }
            
            if topic_results:
                attr_results[topic] = topic_results
        
        if attr_results:
            results[attr] = attr_results
    
    return results

def calculate_intersectional_sp_vectorized(df: pd.DataFrame, config: Dict) -> Dict:
    """Calculate Statistical Parity for intersectional groups using vectorized operations"""
    
    # Add intersectional columns to DataFrame
    df['gender_x_religion'] = df['gender'] + '_x_' + df['religion']
    df['gender_x_country'] = df['gender'] + '_x_' + df['country']
    df['religion_x_country'] = df['religion'] + '_x_' + df['country']
    df['age_x_gender'] = df['age'].astype(str) + '_x_' + df['gender']
    df['religion_x_country_x_gender'] = df['religion'] + '_x_' + df['country'] + '_x_' + df['gender']
    
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
        intersection_values = df[intersection].unique()
        
        for topic in topics:
            topic_data = df[df['topic'] == topic]
            
            if len(topic_data) < 60:
                continue
            
            topic_results = {}
            
            # Process in batches for efficiency
            valid_intersections = []
            for intersection_value in intersection_values:
                if intersection_value == reference_value:
                    continue
                
                # Check if we have enough samples in both models
                for model in ['post_brexit', 'pre_brexit']:
                    model_data = topic_data[topic_data['model'] == model]
                    group_size = len(model_data[model_data[intersection] == intersection_value])
                    ref_size = len(model_data[model_data[intersection] == reference_value])
                    
                    if group_size >= 30 and ref_size >= 30:
                        valid_intersections.append((intersection_value, model))
            
            # Batch process valid intersections
            if valid_intersections:
                for intersection_value, model in valid_intersections:
                    model_data = topic_data[topic_data['model'] == model]
                    
                    group_data = model_data[model_data[intersection] == intersection_value]
                    ref_data = model_data[model_data[intersection] == reference_value]
                    
                    group_rate = group_data['decision_binary'].mean()
                    ref_rate = ref_data['decision_binary'].mean()
                    
                    # Calculate z-test
                    z_scores, p_values = vectorized_z_test(
                        np.array([group_rate]), np.array([len(group_data)]),
                        np.array([ref_rate]), np.array([len(ref_data)])
                    )
                    
                    key = f"{intersection_value}_vs_{reference_value}_{model}"
                    topic_results[key] = {
                        'intersection': intersection,
                        'group': intersection_value,
                        'reference': reference_value,
                        'model': model,
                        'topic': topic,
                        'group_grant_rate': float(group_rate),
                        'reference_grant_rate': float(ref_rate),
                        'sp_gap': float(group_rate - ref_rate),
                        'group_size': int(len(group_data)),
                        'reference_size': int(len(ref_data)),
                        'z_score': float(z_scores[0]) if not np.isnan(z_scores[0]) else None,
                        'p_value': float(p_values[0]) if not np.isnan(p_values[0]) else None,
                        'is_significant': bool(abs(z_scores[0]) > 1.96) if not np.isnan(z_scores[0]) else False
                    }
            
            if topic_results:
                intersection_results[topic] = topic_results
        
        if intersection_results:
            results[intersection] = intersection_results
    
    return results

def main():
    """Main function for optimized Statistical Parity calculation"""
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / "config" / "analysis_config.yaml"
    processed_dir = base_dir / "data" / "processed"
    output_dir = base_dir / "outputs" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = base_dir / "outputs" / "logs" / "statistical_parity.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(log_file))
    
    logger.info("Starting optimized Statistical Parity calculation")
    
    # Load configuration and data
    config = load_config(config_path)
    
    with open(processed_dir / "tagged_records.json", 'r') as f:
        records = json.load(f)
    
    logger.info(f"Loaded {len(records)} records for Statistical Parity analysis")
    
    # CRITICAL FIX: Resolve reference groups from "TBD" to actual values
    resolved_reference_groups = get_reference_groups(records, config)
    config['reference_groups'] = resolved_reference_groups
    
    logger.info("Resolved reference groups:")
    for attr, ref_group in resolved_reference_groups.items():
        logger.info(f"  {attr}: {ref_group}")
    
    # Convert to optimized DataFrame
    logger.info("Converting data to optimized format...")
    df = prepare_data_arrays(records)
    logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    
    # Calculate individual attribute SP
    logger.info("Calculating individual attribute Statistical Parity...")
    individual_sp = calculate_statistical_parity_vectorized(df, config)
    
    # Log individual results summary
    total_individual_comparisons = 0
    significant_individual = 0
    for attr, attr_results in individual_sp.items():
        attr_comparisons = sum(len(topic_results) for topic_results in attr_results.values())
        attr_significant = sum(
            sum(1 for result in topic_results.values() if result['is_significant'])
            for topic_results in attr_results.values()
        )
        total_individual_comparisons += attr_comparisons
        significant_individual += attr_significant
        logger.info(f"  {attr}: {attr_comparisons} comparisons, {attr_significant} significant")
    
    # Calculate intersectional SP
    logger.info("Calculating intersectional Statistical Parity...")
    intersectional_sp = calculate_intersectional_sp_vectorized(df, config)
    
    # Log intersectional results summary
    total_intersectional_comparisons = 0
    significant_intersectional = 0
    for intersection, inter_results in intersectional_sp.items():
        inter_comparisons = sum(len(topic_results) for topic_results in inter_results.values())
        inter_significant = sum(
            sum(1 for result in topic_results.values() if result['is_significant'])
            for topic_results in inter_results.values()
        )
        total_intersectional_comparisons += inter_comparisons
        significant_intersectional += inter_significant
        logger.info(f"  {intersection}: {inter_comparisons} comparisons, {inter_significant} significant")
    
    # Overall summary
    total_comparisons = total_individual_comparisons + total_intersectional_comparisons
    total_significant = significant_individual + significant_intersectional
    
    logger.info("Statistical Parity Analysis Summary:")
    logger.info(f"  Total comparisons: {total_comparisons}")
    logger.info(f"  Significant disparities: {total_significant} ({total_significant/total_comparisons*100:.1f}%)")
    logger.info(f"  Individual attribute disparities: {significant_individual}/{total_individual_comparisons}")
    logger.info(f"  Intersectional disparities: {significant_intersectional}/{total_intersectional_comparisons}")
    
    # Save results
    sp_results = {
        'individual_attributes': individual_sp,
        'intersectional_groups': intersectional_sp,
        'summary': {
            'total_comparisons': total_comparisons,
            'total_significant': total_significant,
            'significance_rate': total_significant/total_comparisons if total_comparisons > 0 else 0,
            'individual_comparisons': total_individual_comparisons,
            'individual_significant': significant_individual,
            'intersectional_comparisons': total_intersectional_comparisons,
            'intersectional_significant': significant_intersectional
        },
        'configuration': {
            'significance_threshold': config['significance_threshold'],
            'min_group_size': config['min_group_size'],
            'reference_groups': config['reference_groups']
        }
    }
    
    # Save comprehensive results
    sp_file = output_dir / "statistical_parity_results.json"
    with open(sp_file, 'w') as f:
        json.dump(sp_results, f, indent=2, default=json_serial)
    
    logger.info(f"Saved Statistical Parity results to {sp_file}")
    logger.info("Statistical Parity calculation completed successfully!")
    
    return sp_results

if __name__ == "__main__":
    results = main() 