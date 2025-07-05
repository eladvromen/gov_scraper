"""
Comprehensive Counterfactual Analysis using ALL vignette attributes
This properly identifies counterfactual pairs by considering ALL non-generic attributes
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_utils import load_config, setup_logging

def extract_all_vignette_attributes(records: List[Dict]) -> Dict:
    """
    Extract ALL attributes from vignette records for proper counterfactual analysis
    
    Returns:
        Dict with attribute names and their possible values
    """
    
    all_attributes = defaultdict(set)
    
    for record in records:
        # Basic attributes
        all_attributes['topic'].add(record['topic'])
        all_attributes['meta_topic'].add(record['meta_topic'])
        
        # Protected attributes
        for attr, value in record['protected_attributes'].items():
            all_attributes[attr].add(value)
        
        # Additional vignette-specific attributes from metadata
        if 'original_record' in record and 'metadata' in record['original_record']:
            metadata = record['original_record']['metadata']
            
            if 'fields' in metadata:
                for field, value in metadata['fields'].items():
                    if field not in ['age', 'religion', 'gender', 'country']:  # Skip already captured
                        all_attributes[field].add(value)
    
    # Convert sets to lists for JSON serialization
    return {attr: list(values) for attr, values in all_attributes.items()}

def prepare_comprehensive_counterfactual_data(records: List[Dict]) -> pd.DataFrame:
    """
    Prepare data for comprehensive counterfactual analysis using ALL attributes
    """
    
    data = []
    
    for i, record in enumerate(records):
        # Extract all attributes for this record
        record_attributes = {
            'index': i,
            'sample_id': record['sample_id'],
            'topic': record['topic'],
            'meta_topic': record['meta_topic'],
            'model': record['model'],
            'decision': record['decision'],
            'decision_binary': 1 if record['decision'] == 'GRANT' else 0,
        }
        
        # Add protected attributes
        for attr, value in record['protected_attributes'].items():
            record_attributes[attr] = value
        
        # Add vignette-specific attributes
        if 'original_record' in record and 'metadata' in record['original_record']:
            metadata = record['original_record']['metadata']
            
            if 'fields' in metadata:
                for field, value in metadata['fields'].items():
                    if field not in ['age', 'religion', 'gender', 'country']:  # Skip duplicates
                        record_attributes[field] = value
        
        data.append(record_attributes)
    
    return pd.DataFrame(data)

def identify_counterfactual_pairs_comprehensive(df: pd.DataFrame) -> List[Tuple[int, int, str]]:
    """
    Identify counterfactual pairs using ALL attributes
    
    Returns:
        List of (index1, index2, differing_attribute) tuples
    """
    
    # Identify all non-generic attributes (exclude index, model, decision)
    non_generic_attrs = [col for col in df.columns 
                        if col not in ['index', 'model', 'decision', 'decision_binary']]
    
    counterfactual_pairs = []
    
    # Group by sample_id to ensure we're comparing the same vignette
    for sample_id in df['sample_id'].unique():
        sample_data = df[df['sample_id'] == sample_id]
        
        if len(sample_data) < 2:
            continue
            
        # Check all pairs within this sample
        indices = sample_data.index.tolist()
        
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx1, idx2 = indices[i], indices[j]
                record1 = sample_data.loc[idx1]
                record2 = sample_data.loc[idx2]
                
                # Count attribute differences
                differing_attrs = []
                for attr in non_generic_attrs:
                    if record1[attr] != record2[attr]:
                        differing_attrs.append(attr)
                
                # Counterfactual if exactly 1 attribute differs
                if len(differing_attrs) == 1:
                    counterfactual_pairs.append((idx1, idx2, differing_attrs[0]))
    
    return counterfactual_pairs

def analyze_counterfactual_decisions(df: pd.DataFrame, 
                                   counterfactual_pairs: List[Tuple[int, int, str]]) -> Dict:
    """
    Analyze decision differences in counterfactual pairs
    """
    
    results = {
        'by_differing_attribute': defaultdict(list),
        'by_topic': defaultdict(list),
        'by_model_pair': defaultdict(list),
        'flip_analysis': defaultdict(int)
    }
    
    for idx1, idx2, differing_attr in counterfactual_pairs:
        record1 = df.loc[idx1]
        record2 = df.loc[idx2]
        
        # Basic pair info
        pair_info = {
            'sample_id': record1['sample_id'],
            'topic': record1['topic'],
            'differing_attribute': differing_attr,
            'value1': record1[differing_attr],
            'value2': record2[differing_attr],
            'model1': record1['model'],
            'model2': record2['model'],
            'decision1': record1['decision'],
            'decision2': record2['decision'],
            'decision_binary1': record1['decision_binary'],
            'decision_binary2': record2['decision_binary'],
            'decision_flip': record1['decision_binary'] != record2['decision_binary']
        }
        
        # Categorize the pair
        results['by_differing_attribute'][differing_attr].append(pair_info)
        results['by_topic'][record1['topic']].append(pair_info)
        
        # Model comparison categories
        if record1['model'] != record2['model']:
            model_pair = f"{record1['model']}_vs_{record2['model']}"
            results['by_model_pair'][model_pair].append(pair_info)
            
            # Flip analysis for temporal comparison
            if pair_info['decision_flip']:
                if record1['decision_binary'] > record2['decision_binary']:
                    results['flip_analysis'][f"{record1['model']}_favors_{differing_attr}_{record1[differing_attr]}"] += 1
                else:
                    results['flip_analysis'][f"{record2['model']}_favors_{differing_attr}_{record2[differing_attr]}"] += 1
    
    return results

def calculate_counterfactual_bias_metrics(analysis_results: Dict) -> Dict:
    """
    Calculate bias metrics from counterfactual analysis
    """
    
    metrics = {
        'attribute_bias_analysis': {},
        'temporal_bias_analysis': {},
        'topic_bias_analysis': {},
        'overall_summary': {}
    }
    
    # Analyze bias by differing attribute
    for attr, pairs in analysis_results['by_differing_attribute'].items():
        if len(pairs) < 10:  # Minimum pairs for meaningful analysis
            continue
            
        total_pairs = len(pairs)
        flip_pairs = sum(1 for pair in pairs if pair['decision_flip'])
        flip_rate = flip_pairs / total_pairs
        
        # Value-specific analysis
        value_analysis = defaultdict(lambda: {'total': 0, 'flips': 0})
        
        for pair in pairs:
            value1, value2 = pair['value1'], pair['value2']
            
            value_analysis[value1]['total'] += 1
            value_analysis[value2]['total'] += 1
            
            if pair['decision_flip']:
                value_analysis[value1]['flips'] += 1
                value_analysis[value2]['flips'] += 1
        
        metrics['attribute_bias_analysis'][attr] = {
            'total_pairs': total_pairs,
            'flip_pairs': flip_pairs,
            'flip_rate': flip_rate,
            'value_analysis': {
                val: {
                    'total_pairs': info['total'],
                    'flip_pairs': info['flips'],
                    'flip_rate': info['flips'] / info['total'] if info['total'] > 0 else 0
                }
                for val, info in value_analysis.items()
            }
        }
    
    # Analyze temporal bias (model comparisons)
    for model_pair, pairs in analysis_results['by_model_pair'].items():
        if len(pairs) < 10:
            continue
            
        total_pairs = len(pairs)
        flip_pairs = sum(1 for pair in pairs if pair['decision_flip'])
        flip_rate = flip_pairs / total_pairs
        
        # Direction analysis
        direction_analysis = defaultdict(int)
        for pair in pairs:
            if pair['decision_flip']:
                if pair['decision_binary1'] > pair['decision_binary2']:
                    direction_analysis[f"{pair['model1']}_more_lenient"] += 1
                else:
                    direction_analysis[f"{pair['model2']}_more_lenient"] += 1
        
        metrics['temporal_bias_analysis'][model_pair] = {
            'total_pairs': total_pairs,
            'flip_pairs': flip_pairs,
            'flip_rate': flip_rate,
            'direction_analysis': dict(direction_analysis)
        }
    
    # Topic analysis
    for topic, pairs in analysis_results['by_topic'].items():
        if len(pairs) < 5:
            continue
            
        total_pairs = len(pairs)
        flip_pairs = sum(1 for pair in pairs if pair['decision_flip'])
        flip_rate = flip_pairs / total_pairs
        
        # Attribute breakdown for this topic
        attr_breakdown = defaultdict(lambda: {'total': 0, 'flips': 0})
        
        for pair in pairs:
            attr = pair['differing_attribute']
            attr_breakdown[attr]['total'] += 1
            if pair['decision_flip']:
                attr_breakdown[attr]['flips'] += 1
        
        metrics['topic_bias_analysis'][topic] = {
            'total_pairs': total_pairs,
            'flip_pairs': flip_pairs,
            'flip_rate': flip_rate,
            'attribute_breakdown': {
                attr: {
                    'total_pairs': info['total'],
                    'flip_pairs': info['flips'],
                    'flip_rate': info['flips'] / info['total'] if info['total'] > 0 else 0
                }
                for attr, info in attr_breakdown.items()
            }
        }
    
    # Overall summary
    total_pairs = len(analysis_results.get('by_differing_attribute', {}).get('model', []))
    if total_pairs == 0:
        total_pairs = sum(len(pairs) for pairs in analysis_results['by_differing_attribute'].values())
    
    total_flips = sum(
        sum(1 for pair in pairs if pair['decision_flip'])
        for pairs in analysis_results['by_differing_attribute'].values()
    )
    
    metrics['overall_summary'] = {
        'total_counterfactual_pairs': total_pairs,
        'total_decision_flips': total_flips,
        'overall_flip_rate': total_flips / total_pairs if total_pairs > 0 else 0,
        'attributes_analyzed': len(analysis_results['by_differing_attribute']),
        'topics_analyzed': len(analysis_results['by_topic']),
        'flip_pattern_analysis': dict(analysis_results['flip_analysis'])
    }
    
    return metrics

def main():
    """Main function for comprehensive counterfactual analysis"""
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / "config" / "analysis_config.yaml"
    processed_dir = base_dir / "data" / "processed"
    output_dir = base_dir / "outputs" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = base_dir / "outputs" / "logs" / "comprehensive_counterfactual.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(log_file))
    
    logger.info("Starting comprehensive counterfactual analysis")
    
    # Load data
    with open(processed_dir / "tagged_records.json", 'r') as f:
        records = json.load(f)
    
    logger.info(f"Loaded {len(records)} records for comprehensive counterfactual analysis")
    
    # Extract all attributes
    logger.info("Extracting all vignette attributes...")
    all_attributes = extract_all_vignette_attributes(records)
    
    logger.info(f"Found {len(all_attributes)} unique attributes:")
    for attr, values in all_attributes.items():
        logger.info(f"  {attr}: {len(values)} unique values")
    
    # Prepare comprehensive data
    logger.info("Preparing comprehensive counterfactual data...")
    df = prepare_comprehensive_counterfactual_data(records)
    
    logger.info(f"Created DataFrame with {len(df)} records and {len(df.columns)} attributes")
    
    # Identify counterfactual pairs
    logger.info("Identifying counterfactual pairs using ALL attributes...")
    counterfactual_pairs = identify_counterfactual_pairs_comprehensive(df)
    
    logger.info(f"Found {len(counterfactual_pairs)} counterfactual pairs")
    
    # Analyze the pairs
    logger.info("Analyzing counterfactual decisions...")
    analysis_results = analyze_counterfactual_decisions(df, counterfactual_pairs)
    
    # Calculate bias metrics
    logger.info("Calculating comprehensive bias metrics...")
    bias_metrics = calculate_counterfactual_bias_metrics(analysis_results)
    
    # Log summary
    logger.info("Comprehensive Counterfactual Analysis Summary:")
    logger.info(f"  Total counterfactual pairs: {len(counterfactual_pairs)}")
    logger.info(f"  Attributes with counterfactuals: {len(analysis_results['by_differing_attribute'])}")
    logger.info(f"  Topics with counterfactuals: {len(analysis_results['by_topic'])}")
    logger.info(f"  Model pairs analyzed: {len(analysis_results['by_model_pair'])}")
    
    # Log attribute breakdown
    for attr, pairs in analysis_results['by_differing_attribute'].items():
        flip_rate = sum(1 for pair in pairs if pair['decision_flip']) / len(pairs) if pairs else 0
        logger.info(f"  {attr}: {len(pairs)} pairs, {flip_rate:.3f} flip rate")
    
    # Save comprehensive results
    comprehensive_results = {
        'all_attributes_found': all_attributes,
        'counterfactual_pairs': [
            {
                'index1': int(idx1),
                'index2': int(idx2),
                'differing_attribute': attr,
                'sample_id': int(df.loc[idx1, 'sample_id']),
                'topic': df.loc[idx1, 'topic']
            }
            for idx1, idx2, attr in counterfactual_pairs
        ],
        'analysis_results': {
            'by_differing_attribute': {
                attr: pairs for attr, pairs in analysis_results['by_differing_attribute'].items()
            },
            'by_topic': {
                topic: pairs for topic, pairs in analysis_results['by_topic'].items()
            },
            'by_model_pair': {
                pair: pairs for pair, pairs in analysis_results['by_model_pair'].items()
            },
            'flip_analysis': dict(analysis_results['flip_analysis'])
        },
        'bias_metrics': bias_metrics,
        'summary': {
            'total_pairs': len(counterfactual_pairs),
            'attributes_analyzed': len(analysis_results['by_differing_attribute']),
            'topics_analyzed': len(analysis_results['by_topic']),
            'model_pairs_analyzed': len(analysis_results['by_model_pair'])
        }
    }
    
    # Save results
    comprehensive_file = output_dir / "comprehensive_counterfactual_results.json"
    with open(comprehensive_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    logger.info(f"Saved comprehensive counterfactual results to {comprehensive_file}")
    logger.info("Comprehensive counterfactual analysis completed successfully!")
    
    return comprehensive_results

if __name__ == "__main__":
    results = main() 