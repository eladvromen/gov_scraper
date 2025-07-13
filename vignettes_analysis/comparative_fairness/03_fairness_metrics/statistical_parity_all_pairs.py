#!/usr/bin/env python3
"""
Statistical Parity calculation with ALL PAIRWISE COMPARISONS
Instead of only comparing to reference groups, this generates all possible pairs
like Pakistan vs Syria, China vs Nigeria, etc.
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

def calculate_statistical_parity_all_pairs(df: pd.DataFrame, config: Dict) -> Dict:
    """Calculate Statistical Parity using ALL PAIRWISE COMPARISONS"""
    
    results = {}
    
    # Process each protected attribute
    for attr in config['protected_attributes']:
        print(f"Processing {attr} attribute...")
        attr_results = {}
        
        # Get unique topics and values for this attribute
        topics = df['topic'].unique()
        attr_values = df[attr].unique()
        
        # Filter out values that don't have enough data
        valid_attr_values = []
        for attr_value in attr_values:
            total_count = len(df[df[attr] == attr_value])
            if total_count >= 60:  # Ensure we have enough data across models
                valid_attr_values.append(attr_value)
        
        print(f"  Valid {attr} values: {len(valid_attr_values)} out of {len(attr_values)}")
        
        for topic in topics:
            topic_data = df[df['topic'] == topic]
            
            # Skip if not enough data
            if len(topic_data) < 60:  # At least 30 per model
                continue
            
            topic_results = {}
            
            # Generate ALL PAIRWISE COMBINATIONS
            for attr_value1, attr_value2 in itertools.combinations(valid_attr_values, 2):
                # Calculate for each model
                for model in ['post_brexit', 'pre_brexit']:
                    model_data = topic_data[topic_data['model'] == model]
                    
                    # Get group and reference data
                    group_data = model_data[model_data[attr] == attr_value1]
                    ref_data = model_data[model_data[attr] == attr_value2]
                    
                    if len(group_data) >= 30 and len(ref_data) >= 30:
                        # Calculate grant rates
                        group_rate = group_data['decision_binary'].mean()
                        ref_rate = ref_data['decision_binary'].mean()
                        
                        # Calculate z-test using vectorized function
                        z_scores, p_values = vectorized_z_test(
                            np.array([group_rate]), np.array([len(group_data)]),
                            np.array([ref_rate]), np.array([len(ref_data)])
                        )
                        
                        # Create comparison key
                        key = f"{attr_value1}_vs_{attr_value2}_{model}"
                        topic_results[key] = {
                            'attribute': attr,
                            'group': attr_value1,
                            'reference': attr_value2,
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
                        
                        # Also create the reverse comparison
                        reverse_key = f"{attr_value2}_vs_{attr_value1}_{model}"
                        topic_results[reverse_key] = {
                            'attribute': attr,
                            'group': attr_value2,
                            'reference': attr_value1,
                            'model': model,
                            'topic': topic,
                            'group_grant_rate': float(ref_rate),
                            'reference_grant_rate': float(group_rate),
                            'sp_gap': float(ref_rate - group_rate),
                            'group_size': int(len(ref_data)),
                            'reference_size': int(len(group_data)),
                            'z_score': float(-z_scores[0]) if not np.isnan(z_scores[0]) else None,
                            'p_value': float(p_values[0]) if not np.isnan(p_values[0]) else None,
                            'is_significant': bool(abs(z_scores[0]) > 1.96) if not np.isnan(z_scores[0]) else False
                        }
            
            if topic_results:
                attr_results[topic] = topic_results
                print(f"    Topic '{topic}': {len(topic_results)} comparisons")
        
        if attr_results:
            results[attr] = attr_results
            total_comparisons = sum(len(topic_data) for topic_data in attr_results.values())
            print(f"  Total {attr} comparisons: {total_comparisons}")
    
    return results

def calculate_intersectional_sp_all_pairs(df: pd.DataFrame, config: Dict) -> Dict:
    """Calculate Statistical Parity for intersectional groups using ALL PAIRWISE COMPARISONS"""
    
    # Add intersectional columns to DataFrame
    df['gender_x_religion'] = df['gender'] + '_x_' + df['religion']
    df['gender_x_country'] = df['gender'] + '_x_' + df['country']
    df['religion_x_country'] = df['religion'] + '_x_' + df['country']
    df['age_x_gender'] = df['age'].astype(str) + '_x_' + df['gender']
    df['religion_x_country_x_gender'] = df['religion'] + '_x_' + df['country'] + '_x_' + df['gender']
    
    results = {}
    
    for intersection in config['intersections']:
        print(f"Processing {intersection} intersection...")
        intersection_results = {}
        
        topics = df['topic'].unique()
        intersection_values = df[intersection].unique()
        
        # Filter valid intersection values
        valid_intersection_values = []
        for intersection_value in intersection_values:
            total_count = len(df[df[intersection] == intersection_value])
            if total_count >= 60:
                valid_intersection_values.append(intersection_value)
        
        print(f"  Valid {intersection} values: {len(valid_intersection_values)} out of {len(intersection_values)}")
        
        for topic in topics:
            topic_data = df[df['topic'] == topic]
            
            if len(topic_data) < 60:
                continue
            
            topic_results = {}
            
            # Generate ALL PAIRWISE COMBINATIONS for intersections
            for intersection_value1, intersection_value2 in itertools.combinations(valid_intersection_values, 2):
                # Calculate for each model
                for model in ['post_brexit', 'pre_brexit']:
                    model_data = topic_data[topic_data['model'] == model]
                    
                    group_data = model_data[model_data[intersection] == intersection_value1]
                    ref_data = model_data[model_data[intersection] == intersection_value2]
                    
                    if len(group_data) >= 30 and len(ref_data) >= 30:
                        group_rate = group_data['decision_binary'].mean()
                        ref_rate = ref_data['decision_binary'].mean()
                        
                        # Calculate z-test
                        z_scores, p_values = vectorized_z_test(
                            np.array([group_rate]), np.array([len(group_data)]),
                            np.array([ref_rate]), np.array([len(ref_data)])
                        )
                        
                        key = f"{intersection_value1}_vs_{intersection_value2}_{model}"
                        topic_results[key] = {
                            'intersection': intersection,
                            'group': intersection_value1,
                            'reference': intersection_value2,
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
                        
                        # Also create the reverse comparison
                        reverse_key = f"{intersection_value2}_vs_{intersection_value1}_{model}"
                        topic_results[reverse_key] = {
                            'intersection': intersection,
                            'group': intersection_value2,
                            'reference': intersection_value1,
                            'model': model,
                            'topic': topic,
                            'group_grant_rate': float(ref_rate),
                            'reference_grant_rate': float(group_rate),
                            'sp_gap': float(ref_rate - group_rate),
                            'group_size': int(len(ref_data)),
                            'reference_size': int(len(group_data)),
                            'z_score': float(-z_scores[0]) if not np.isnan(z_scores[0]) else None,
                            'p_value': float(p_values[0]) if not np.isnan(p_values[0]) else None,
                            'is_significant': bool(abs(z_scores[0]) > 1.96) if not np.isnan(z_scores[0]) else False
                        }
            
            if topic_results:
                intersection_results[topic] = topic_results
                print(f"    Topic '{topic}': {len(topic_results)} comparisons")
        
        if intersection_results:
            results[intersection] = intersection_results
            total_comparisons = sum(len(topic_data) for topic_data in intersection_results.values())
            print(f"  Total {intersection} comparisons: {total_comparisons}")
    
    return results

def main():
    """Main function for ALL PAIRWISE Statistical Parity calculation"""
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / "config" / "analysis_config.yaml"
    processed_dir = base_dir / "data" / "processed"
    output_dir = base_dir / "outputs" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = base_dir / "outputs" / "logs" / "statistical_parity_all_pairs.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(log_file))
    
    logger.info("Starting ALL PAIRWISE Statistical Parity calculation")
    print("=" * 80)
    print("ALL PAIRWISE STATISTICAL PARITY CALCULATION")
    print("=" * 80)
    
    # Load configuration and data
    config = load_config(config_path)
    
    with open(processed_dir / "tagged_records.json", 'r') as f:
        records = json.load(f)
    
    logger.info(f"Loaded {len(records)} records for Statistical Parity analysis")
    print(f"Loaded {len(records)} records")
    
    # Convert to optimized DataFrame
    logger.info("Converting data to optimized format...")
    print("Converting data to optimized format...")
    df = prepare_data_arrays(records)
    logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    print(f"Created DataFrame with {len(df)} rows")
    
    # Calculate individual attribute SP with all pairs
    logger.info("Calculating individual attribute Statistical Parity (ALL PAIRS)...")
    print("\nCalculating individual attribute Statistical Parity (ALL PAIRS)...")
    individual_sp = calculate_statistical_parity_all_pairs(df, config)
    
    # Calculate intersectional SP with all pairs
    logger.info("Calculating intersectional Statistical Parity (ALL PAIRS)...")
    print("\nCalculating intersectional Statistical Parity (ALL PAIRS)...")
    intersectional_sp = calculate_intersectional_sp_all_pairs(df, config)
    
    # Log results summary
    total_individual_comparisons = 0
    significant_individual = 0
    
    for attr_data in individual_sp.values():
        for topic_data in attr_data.values():
            total_individual_comparisons += len(topic_data)
            significant_individual += sum(1 for result in topic_data.values() if result['is_significant'])
    
    total_intersectional_comparisons = 0
    significant_intersectional = 0
    
    for intersection_data in intersectional_sp.values():
        for topic_data in intersection_data.values():
            total_intersectional_comparisons += len(topic_data)
            significant_intersectional += sum(1 for result in topic_data.values() if result['is_significant'])
    
    total_comparisons = total_individual_comparisons + total_intersectional_comparisons
    total_significant = significant_individual + significant_intersectional
    
    logger.info(f"Statistical Parity Results Summary:")
    logger.info(f"  Total comparisons: {total_comparisons}")
    logger.info(f"  Total significant: {total_significant}")
    logger.info(f"  Significance rate: {total_significant/total_comparisons:.3f}")
    logger.info(f"  Individual disparities: {significant_individual}/{total_individual_comparisons}")
    logger.info(f"  Intersectional disparities: {significant_intersectional}/{total_intersectional_comparisons}")
    
    print(f"\nResults Summary:")
    print(f"  Total comparisons: {total_comparisons}")
    print(f"  Total significant: {total_significant}")
    print(f"  Significance rate: {total_significant/total_comparisons:.3f}")
    print(f"  Individual disparities: {significant_individual}/{total_individual_comparisons}")
    print(f"  Intersectional disparities: {significant_intersectional}/{total_intersectional_comparisons}")
    
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
            'comparison_method': 'all_pairwise'
        }
    }
    
    # Save comprehensive results
    sp_file = output_dir / "statistical_parity_all_pairs_results.json"
    with open(sp_file, 'w') as f:
        json.dump(sp_results, f, indent=2, default=json_serial)
    
    logger.info(f"Saved Statistical Parity results to {sp_file}")
    print(f"\nSaved results to: {sp_file}")
    print("ALL PAIRWISE Statistical Parity calculation completed successfully!")
    
    return sp_results

if __name__ == "__main__":
    main() 