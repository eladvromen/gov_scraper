"""
Optimized Counterfactual Fairness Analysis using numpy vectorization and numba
Identifies counterfactual pairs and calculates flip rates efficiently
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set
import numba
from numba import jit
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_utils import load_config, setup_logging

@jit(nopython=True)
def count_attribute_differences(attrs1: np.ndarray, attrs2: np.ndarray) -> int:
    """Count differences between two attribute arrays using numba"""
    differences = 0
    for i in range(len(attrs1)):
        if attrs1[i] != attrs2[i]:
            differences += 1
    return differences

def find_counterfactual_pairs_optimized(attribute_matrix: np.ndarray, 
                                       topic_codes: np.ndarray,
                                       scenario_keys: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find counterfactual pairs efficiently using optimized numpy operations
    
    Args:
        attribute_matrix: N x 4 matrix of encoded protected attributes
        topic_codes: Array of topic codes for each record
        scenario_keys: Array of scenario keys (non-protected attributes)
    
    Returns:
        List of (index1, index2) pairs that are counterfactuals
    """
    n_records = len(attribute_matrix)
    pairs = []
    
    # Use vectorized operations for efficiency
    for i in range(n_records):
        # Find records with same topic and scenario
        same_context_mask = (topic_codes == topic_codes[i]) & (scenario_keys == scenario_keys[i])
        same_context_indices = np.where(same_context_mask)[0]
        
        # Only check records after current index to avoid duplicates
        later_indices = same_context_indices[same_context_indices > i]
        
        if len(later_indices) > 0:
            # Vectorized difference calculation
            attr_diffs = np.sum(attribute_matrix[later_indices] != attribute_matrix[i], axis=1)
            
            # Find pairs with exactly 1 difference
            counterfactual_mask = attr_diffs == 1
            counterfactual_indices = later_indices[counterfactual_mask]
            
            # Add pairs
            for j in counterfactual_indices:
                pairs.append((i, j))
    
    return pairs

def prepare_counterfactual_data(records: List[Dict]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare data for efficient counterfactual analysis"""
    
    data = []
    
    for i, record in enumerate(records):
        # Create scenario key from non-protected attributes
        metadata = record.get('original_record', {}).get('metadata', {})
        fields = metadata.get('fields', {})
        
        scenario_parts = []
        for field, value in sorted(fields.items()):
            if field not in ['country', 'age', 'religion', 'gender']:
                scenario_parts.append(f"{field}:{value}")
        
        scenario_key = "|".join(scenario_parts)
        
        data.append({
            'index': i,
            'topic': record['topic'],
            'model': record['model'],
            'decision': record['decision'],
            'decision_binary': 1 if record['decision'] == 'GRANT' else 0,
            'country': record['protected_attributes']['country'],
            'age': record['protected_attributes']['age'],
            'religion': record['protected_attributes']['religion'],
            'gender': record['protected_attributes']['gender'],
            'scenario_key': scenario_key
        })
    
    df = pd.DataFrame(data)
    
    # Create encoded arrays for numba
    attribute_encoders = {}
    attribute_matrix = np.zeros((len(df), 4), dtype=np.int32)
    
    for i, attr in enumerate(['country', 'age', 'religion', 'gender']):
        unique_values = df[attr].unique()
        encoder = {val: idx for idx, val in enumerate(unique_values)}
        attribute_encoders[attr] = encoder
        attribute_matrix[:, i] = df[attr].map(encoder).values
    
    # Encode topics and scenarios
    topic_encoder = {val: idx for idx, val in enumerate(df['topic'].unique())}
    topic_codes = df['topic'].map(topic_encoder).values
    
    scenario_encoder = {val: idx for idx, val in enumerate(df['scenario_key'].unique())}
    scenario_codes = df['scenario_key'].map(scenario_encoder).values
    
    return df, attribute_matrix, topic_codes, scenario_codes

def calculate_counterfactual_flip_rates(df: pd.DataFrame, 
                                       counterfactual_pairs: List[Tuple[int, int]],
                                       config: Dict) -> Dict:
    """Calculate counterfactual flip rates for all groups"""
    
    results = {}
    
    # Convert pairs to DataFrame for easier analysis
    pair_data = []
    
    for idx1, idx2 in counterfactual_pairs:
        record1 = df.iloc[idx1]
        record2 = df.iloc[idx2]
        
        # Find which attribute differs
        differing_attr = None
        for attr in ['country', 'age', 'religion', 'gender']:
            if record1[attr] != record2[attr]:
                differing_attr = attr
                break
        
        if differing_attr is None:
            continue
        
        # Both models must have this pair
        model1 = record1['model']
        model2 = record2['model']
        
        if model1 == model2:
            continue  # Skip same model pairs
        
        # Create pair record
        if model1 == 'post_brexit':
            post_record, pre_record = record1, record2
        else:
            post_record, pre_record = record2, record1
        
        decision_flip = post_record['decision_binary'] != pre_record['decision_binary']
        
        pair_data.append({
            'topic': post_record['topic'],
            'differing_attribute': differing_attr,
            'value1': post_record[differing_attr],
            'value2': pre_record[differing_attr],
            'post_decision': post_record['decision_binary'],
            'pre_decision': pre_record['decision_binary'],
            'decision_flip': decision_flip,
            'flip_direction': 'grant_to_deny' if post_record['decision_binary'] < pre_record['decision_binary'] else 'deny_to_grant'
        })
    
    pairs_df = pd.DataFrame(pair_data)
    
    if len(pairs_df) == 0:
        return results
    
    # Calculate flip rates by attribute
    for attr in config['protected_attributes']:
        attr_pairs = pairs_df[pairs_df['differing_attribute'] == attr]
        
        if len(attr_pairs) == 0:
            continue
        
        attr_results = {}
        
        # Group by topic
        for topic in attr_pairs['topic'].unique():
            topic_pairs = attr_pairs[attr_pairs['topic'] == topic]
            
            if len(topic_pairs) < 10:  # Minimum pairs for meaningful analysis
                continue
            
            # Calculate overall flip rate for this topic/attribute
            flip_rate = topic_pairs['decision_flip'].mean()
            
            # Calculate directional flip rates
            grant_to_deny_rate = (topic_pairs['flip_direction'] == 'grant_to_deny').mean()
            deny_to_grant_rate = (topic_pairs['flip_direction'] == 'deny_to_grant').mean()
            
            # Calculate flip rates by value pairs
            value_pair_results = {}
            for _, group in topic_pairs.groupby(['value1', 'value2']):
                if len(group) >= 5:  # Minimum for value pair
                    pair_key = f"{group.iloc[0]['value1']}_vs_{group.iloc[0]['value2']}"
                    value_pair_results[pair_key] = {
                        'flip_rate': group['decision_flip'].mean(),
                        'grant_to_deny_rate': (group['flip_direction'] == 'grant_to_deny').mean(),
                        'deny_to_grant_rate': (group['flip_direction'] == 'deny_to_grant').mean(),
                        'n_pairs': len(group)
                    }
            
            attr_results[topic] = {
                'overall_flip_rate': float(flip_rate),
                'grant_to_deny_rate': float(grant_to_deny_rate),
                'deny_to_grant_rate': float(deny_to_grant_rate),
                'total_pairs': len(topic_pairs),
                'value_pairs': value_pair_results
            }
        
        if attr_results:
            results[attr] = attr_results
    
    return results

def calculate_intersectional_counterfactuals(df: pd.DataFrame,
                                           counterfactual_pairs: List[Tuple[int, int]],
                                           config: Dict) -> Dict:
    """Calculate counterfactual analysis for intersectional groups"""
    
    # Add intersectional columns
    df['gender_x_religion'] = df['gender'] + '_x_' + df['religion']
    df['gender_x_country'] = df['gender'] + '_x_' + df['country']
    df['religion_x_country'] = df['religion'] + '_x_' + df['country']
    df['age_x_gender'] = df['age'].astype(str) + '_x_' + df['gender']
    df['religion_x_country_x_gender'] = df['religion'] + '_x_' + df['country'] + '_x_' + df['gender']
    
    results = {}
    
    # For intersectional analysis, we look for pairs where multiple attributes differ
    # but form meaningful intersectional comparisons
    intersectional_pairs = []
    
    for idx1, idx2 in counterfactual_pairs:
        record1 = df.iloc[idx1]
        record2 = df.iloc[idx2]
        
        # Check if this forms an intersectional comparison
        for intersection in config['intersections']:
            if record1[intersection] != record2[intersection]:
                # Check if only one component of the intersection differs
                intersection_parts = intersection.split('_x_')
                differing_parts = 0
                
                for part in intersection_parts:
                    if record1[part] != record2[part]:
                        differing_parts += 1
                
                # If only one part differs, it's a valid intersectional counterfactual
                if differing_parts == 1:
                    if record1['model'] != record2['model']:
                        if record1['model'] == 'post_brexit':
                            post_record, pre_record = record1, record2
                        else:
                            post_record, pre_record = record2, record1
                        
                        decision_flip = post_record['decision_binary'] != pre_record['decision_binary']
                        
                        intersectional_pairs.append({
                            'topic': post_record['topic'],
                            'intersection': intersection,
                            'value1': post_record[intersection],
                            'value2': pre_record[intersection],
                            'decision_flip': decision_flip,
                            'flip_direction': 'grant_to_deny' if post_record['decision_binary'] < pre_record['decision_binary'] else 'deny_to_grant'
                        })
    
    # Process intersectional pairs
    if intersectional_pairs:
        inter_df = pd.DataFrame(intersectional_pairs)
        
        for intersection in config['intersections']:
            inter_pairs = inter_df[inter_df['intersection'] == intersection]
            
            if len(inter_pairs) == 0:
                continue
            
            intersection_results = {}
            
            for topic in inter_pairs['topic'].unique():
                topic_pairs = inter_pairs[inter_pairs['topic'] == topic]
                
                if len(topic_pairs) >= 5:
                    flip_rate = topic_pairs['decision_flip'].mean()
                    
                    intersection_results[topic] = {
                        'flip_rate': float(flip_rate),
                        'total_pairs': len(topic_pairs),
                        'grant_to_deny_rate': (topic_pairs['flip_direction'] == 'grant_to_deny').mean(),
                        'deny_to_grant_rate': (topic_pairs['flip_direction'] == 'deny_to_grant').mean()
                    }
            
            if intersection_results:
                results[intersection] = intersection_results
    
    return results

def main():
    """Main function for optimized counterfactual analysis"""
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / "config" / "analysis_config.yaml"
    processed_dir = base_dir / "data" / "processed"
    output_dir = base_dir / "outputs" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = base_dir / "outputs" / "logs" / "counterfactual_metrics.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(log_file))
    
    logger.info("Starting optimized counterfactual analysis")
    
    # Load configuration and data
    config = load_config(config_path)
    
    with open(processed_dir / "tagged_records.json", 'r') as f:
        records = json.load(f)
    
    logger.info(f"Loaded {len(records)} records for counterfactual analysis")
    
    # Prepare data for counterfactual analysis
    logger.info("Preparing data for counterfactual pair identification...")
    df, attribute_matrix, topic_codes, scenario_codes = prepare_counterfactual_data(records)
    
    # Find counterfactual pairs using optimized numpy function
    logger.info("Finding counterfactual pairs using optimized algorithm...")
    counterfactual_pairs = find_counterfactual_pairs_optimized(
        attribute_matrix, topic_codes, scenario_codes
    )
    
    logger.info(f"Found {len(counterfactual_pairs)} counterfactual pairs")
    
    # Calculate counterfactual flip rates
    logger.info("Calculating counterfactual flip rates...")
    individual_cf = calculate_counterfactual_flip_rates(df, counterfactual_pairs, config)
    
    # Log individual results
    total_cf_comparisons = 0
    for attr, attr_results in individual_cf.items():
        attr_comparisons = sum(len(topic_results['value_pairs']) for topic_results in attr_results.values())
        total_cf_comparisons += attr_comparisons
        logger.info(f"  {attr}: {attr_comparisons} counterfactual comparisons")
    
    # Calculate intersectional counterfactuals
    logger.info("Calculating intersectional counterfactual analysis...")
    intersectional_cf = calculate_intersectional_counterfactuals(df, counterfactual_pairs, config)
    
    # Log intersectional results
    total_inter_cf = 0
    for intersection, inter_results in intersectional_cf.items():
        inter_comparisons = len(inter_results)
        total_inter_cf += inter_comparisons
        logger.info(f"  {intersection}: {inter_comparisons} intersectional comparisons")
    
    logger.info("Counterfactual Analysis Summary:")
    logger.info(f"  Total counterfactual pairs found: {len(counterfactual_pairs)}")
    logger.info(f"  Individual attribute analyses: {total_cf_comparisons}")
    logger.info(f"  Intersectional analyses: {total_inter_cf}")
    
    # Save results
    cf_results = {
        'individual_attributes': individual_cf,
        'intersectional_groups': intersectional_cf,
        'summary': {
            'total_counterfactual_pairs': len(counterfactual_pairs),
            'individual_comparisons': total_cf_comparisons,
            'intersectional_comparisons': total_inter_cf
        },
        'configuration': {
            'min_pairs_for_analysis': 5,
            'min_pairs_for_topic': 10
        }
    }
    
    # Save comprehensive results
    cf_file = output_dir / "counterfactual_metrics_results.json"
    with open(cf_file, 'w') as f:
        json.dump(cf_results, f, indent=2)
    
    logger.info(f"Saved counterfactual analysis results to {cf_file}")
    
    # Save counterfactual pairs for further analysis
    pairs_df = pd.DataFrame([(i, j) for i, j in counterfactual_pairs], columns=['index1', 'index2'])
    pairs_file = output_dir / "counterfactual_pairs.csv"
    pairs_df.to_csv(pairs_file, index=False)
    
    logger.info(f"Saved counterfactual pairs to {pairs_file}")
    logger.info("Counterfactual analysis completed successfully!")
    
    return cf_results

if __name__ == "__main__":
    results = main() 