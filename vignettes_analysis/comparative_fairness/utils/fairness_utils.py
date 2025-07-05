"""
Fairness calculation utilities for comparative fairness analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import scipy.stats as stats

def create_group_labels(records: List[Dict], protected_attributes: List[str]) -> List[Dict]:
    """Create group labels for all records including intersections"""
    
    labeled_records = []
    
    for record in records:
        labeled_record = record.copy()
        attrs = record['protected_attributes']
        
        # Individual group labels
        for attr in protected_attributes:
            labeled_record[f'{attr}_group'] = attrs[attr]
        
        # Two-way intersections
        labeled_record['gender_x_religion'] = f"{attrs['gender']}_x_{attrs['religion']}"
        labeled_record['gender_x_country'] = f"{attrs['gender']}_x_{attrs['country']}"
        labeled_record['religion_x_country'] = f"{attrs['religion']}_x_{attrs['country']}"
        labeled_record['age_x_gender'] = f"{attrs['age']}_x_{attrs['gender']}"
        
        # Three-way intersection
        labeled_record['religion_x_country_x_gender'] = f"{attrs['religion']}_x_{attrs['country']}_x_{attrs['gender']}"
        
        labeled_records.append(labeled_record)
    
    return labeled_records

def filter_groups_by_size(records: List[Dict], min_size: int = 30) -> Dict[str, Dict]:
    """Filter groups by minimum size per topic per model"""
    
    # Group attributes to check
    group_attrs = [
        'country_group', 'age_group', 'religion_group', 'gender_group',
        'gender_x_religion', 'gender_x_country', 'religion_x_country', 
        'age_x_gender', 'religion_x_country_x_gender'
    ]
    
    # Count samples per topic per model per group
    group_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for record in records:
        topic = record['topic']
        model = record['model']
        
        for attr in group_attrs:
            if attr in record:
                group_value = record[attr]
                group_counts[topic][model][f"{attr}_{group_value}"] += 1
    
    # Identify valid groups (>=min_size in both models)
    valid_groups = defaultdict(lambda: defaultdict(set))
    
    for topic in group_counts:
        for attr in group_attrs:
            # Get all group values for this attribute
            all_groups = set()
            for model in group_counts[topic]:
                for group_key in group_counts[topic][model]:
                    if group_key.startswith(f"{attr}_"):
                        all_groups.add(group_key)
            
            # Check if each group has enough samples in both models
            for group_key in all_groups:
                post_count = group_counts[topic]['post_brexit'].get(group_key, 0)
                pre_count = group_counts[topic]['pre_brexit'].get(group_key, 0)
                
                if post_count >= min_size and pre_count >= min_size:
                    group_value = group_key.replace(f"{attr}_", "")
                    valid_groups[topic][attr].add(group_value)
    
    # Convert to regular dict for JSON serialization
    valid_groups_dict = {}
    for topic in valid_groups:
        valid_groups_dict[topic] = {}
        for attr in valid_groups[topic]:
            valid_groups_dict[topic][attr] = list(valid_groups[topic][attr])
    
    return valid_groups_dict

def calculate_statistical_parity(records: List[Dict], group_attr: str, reference_group: str) -> Dict:
    """Calculate Statistical Parity for a group attribute"""
    
    results = {}
    
    # Group by topic
    topics = set(r['topic'] for r in records)
    
    for topic in topics:
        topic_records = [r for r in records if r['topic'] == topic]
        
        # Get unique groups
        groups = set(r[group_attr] for r in topic_records if group_attr in r)
        
        topic_results = {}
        
        for group in groups:
            if group == reference_group:
                continue
                
            # Calculate grant rates for each model
            for model in ['post_brexit', 'pre_brexit']:
                model_records = [r for r in topic_records if r['model'] == model]
                
                # Group records
                group_records = [r for r in model_records if r.get(group_attr) == group]
                ref_records = [r for r in model_records if r.get(group_attr) == reference_group]
                
                if len(group_records) < 30 or len(ref_records) < 30:
                    continue
                
                # Calculate grant rates
                group_grants = sum(1 for r in group_records if r['decision'] == 'GRANT')
                ref_grants = sum(1 for r in ref_records if r['decision'] == 'GRANT')
                
                group_rate = group_grants / len(group_records)
                ref_rate = ref_grants / len(ref_records)
                
                sp_gap = group_rate - ref_rate
                
                key = f"{group}_vs_{reference_group}_{model}"
                topic_results[key] = {
                    'group': group,
                    'reference': reference_group,
                    'model': model,
                    'group_grant_rate': group_rate,
                    'reference_grant_rate': ref_rate,
                    'sp_gap': sp_gap,
                    'group_size': len(group_records),
                    'reference_size': len(ref_records)
                }
        
        if topic_results:
            results[topic] = topic_results
    
    return results

def calculate_z_test(p1: float, n1: int, p2: float, n2: int) -> Tuple[float, float]:
    """Calculate z-test for two proportions"""
    
    if n1 < 30 or n2 < 30:
        return np.nan, np.nan
    
    # Pooled proportion
    p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    
    # Standard error
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    
    if se == 0:
        return np.nan, np.nan
    
    # Z-score
    z = (p1 - p2) / se
    
    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return z, p_value

def identify_counterfactual_pairs(records: List[Dict], protected_attributes: List[str]) -> List[Tuple[Dict, Dict]]:
    """Identify counterfactual pairs differing by only one protected attribute"""
    
    pairs = []
    
    # Group records by topic and vignette content (excluding protected attributes)
    vignette_groups = defaultdict(list)
    
    for record in records:
        # Create a key excluding protected attributes
        key_parts = []
        key_parts.append(record['topic'])
        
        # Add non-protected fields from metadata if available
        metadata = record.get('original_record', {}).get('metadata', {})
        fields = metadata.get('fields', {})
        
        # Only include non-protected scenario fields
        scenario_fields = []
        for field, value in fields.items():
            if field not in protected_attributes:
                scenario_fields.append(f"{field}:{value}")
        
        key_parts.extend(sorted(scenario_fields))
        vignette_key = "|".join(key_parts)
        
        vignette_groups[vignette_key].append(record)
    
    # Find pairs within each vignette group
    for vignette_key, group_records in vignette_groups.items():
        if len(group_records) < 2:
            continue
            
        # Compare all pairs within the group
        for i in range(len(group_records)):
            for j in range(i + 1, len(group_records)):
                record1 = group_records[i]
                record2 = group_records[j]
                
                # Count differences in protected attributes
                attrs1 = record1['protected_attributes']
                attrs2 = record2['protected_attributes']
                
                differences = 0
                for attr in protected_attributes:
                    if attrs1[attr] != attrs2[attr]:
                        differences += 1
                
                # If exactly one difference, it's a counterfactual pair
                if differences == 1:
                    pairs.append((record1, record2))
    
    return pairs

def calculate_group_distribution_summary(records: List[Dict]) -> Dict:
    """Calculate distribution summary for all groups"""
    
    summary = {}
    
    # Topic distribution
    topic_counts = defaultdict(lambda: defaultdict(int))
    for record in records:
        topic_counts[record['topic']][record['model']] += 1
    
    summary['topic_distribution'] = dict(topic_counts)
    
    # Protected attribute distributions
    for attr in ['country', 'age', 'religion', 'gender']:
        attr_counts = defaultdict(int)
        for record in records:
            value = record['protected_attributes'][attr]
            attr_counts[value] += 1
        summary[f'{attr}_distribution'] = dict(attr_counts)
    
    # Intersection distributions
    intersections = ['gender_x_religion', 'gender_x_country', 'religion_x_country', 
                    'age_x_gender', 'religion_x_country_x_gender']
    
    for intersection in intersections:
        intersection_counts = defaultdict(int)
        for record in records:
            if intersection in record:
                value = record[intersection]
                intersection_counts[value] += 1
        summary[f'{intersection}_distribution'] = dict(intersection_counts)
    
    return summary 