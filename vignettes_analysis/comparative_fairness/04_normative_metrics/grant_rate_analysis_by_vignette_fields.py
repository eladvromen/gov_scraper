#!/usr/bin/env python3
"""
Grant Rate Analysis by Vignette-Specific Fields
Analyzes how different case characteristics (not demographics) affect grant rates
across Pre-Brexit vs Post-Brexit models.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import logging
from scipy.stats import chi2_contingency

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_vignette_structure(vignette_file: str) -> Dict[str, Dict]:
    """Load vignette structure to understand ordinal and horizontal fields"""
    logger.info(f"Loading vignette structure from {vignette_file}")
    
    with open(vignette_file, 'r') as f:
        vignettes = json.load(f)
    
    # Create topic -> field structure mapping
    topic_fields = {}
    for vignette in vignettes:
        topic = vignette['topic']
        topic_fields[topic] = {
            'ordinal_fields': vignette.get('ordinal_fields', {}),
            'horizontal_fields': vignette.get('horizontal_fields', {}),
            'generic_fields': vignette.get('generic_fields', {})
        }
    
    logger.info(f"Loaded structure for {len(topic_fields)} topics")
    return topic_fields

def load_tagged_records(records_file: str) -> List[Dict]:
    """Load tagged records from JSON file"""
    logger.info(f"Loading tagged records from {records_file}")
    
    with open(records_file, 'r') as f:
        records = json.load(f)
    
    logger.info(f"Loaded {len(records)} tagged records")
    return records

def extract_field_values(record: Dict, topic_fields: Dict) -> Dict[str, Any]:
    """Extract vignette-specific field values from a record"""
    # Get the original record which contains the field values
    original_record = record.get('original_record', {})
    metadata = original_record.get('metadata', {})
    fields = metadata.get('fields', {})
    
    # Extract all non-generic fields
    generic_fields = {'country', 'age', 'religion', 'gender', 'name', 'pronoun', 'country_B'}
    field_values = {}
    
    for field_name, field_value in fields.items():
        if field_name not in generic_fields:
            field_values[field_name] = field_value
    
    return field_values

def calculate_grant_rates(records: List[Dict], topic_fields: Dict) -> Dict:
    """Calculate grant rates for each (topic, field, field_value, model) combination"""
    logger.info("Calculating grant rates for all field combinations")
    
    # Structure: topic -> field -> field_value -> model -> {grants, total}
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'grants': 0, 'total': 0}))))
    
    # Also track overall model statistics
    model_stats = defaultdict(lambda: {'grants': 0, 'total': 0})
    
    for record in records:
        topic = record['topic']
        model = record['model']
        decision = record['decision']
        
        # Skip inconclusive decisions
        if decision == 'INCONCLUSIVE':
            continue
            
        # Update model stats
        model_stats[model]['total'] += 1
        if decision == 'GRANT':
            model_stats[model]['grants'] += 1
        
        # Extract field values for this record
        field_values = extract_field_values(record, topic_fields)
        
        # Update statistics for each field
        for field_name, field_value in field_values.items():
            stats[topic][field_name][field_value][model]['total'] += 1
            if decision == 'GRANT':
                stats[topic][field_name][field_value][model]['grants'] += 1
    
    # Calculate grant rates
    grant_rates = {}
    grant_rates['model_overall'] = {}
    
    for model, model_stat in model_stats.items():
        grant_rates['model_overall'][model] = model_stat['grants'] / model_stat['total'] if model_stat['total'] > 0 else 0
    
    grant_rates['detailed'] = {}
    for topic in stats:
        grant_rates['detailed'][topic] = {}
        for field in stats[topic]:
            grant_rates['detailed'][topic][field] = {}
            for field_value in stats[topic][field]:
                grant_rates['detailed'][topic][field][field_value] = {}
                for model in stats[topic][field][field_value]:
                    model_stat = stats[topic][field][field_value][model]
                    rate = model_stat['grants'] / model_stat['total'] if model_stat['total'] > 0 else 0
                    grant_rates['detailed'][topic][field][field_value][model] = {
                        'grant_rate': rate,
                        'grants': model_stat['grants'],
                        'total': model_stat['total']
                    }
    
    logger.info(f"Calculated grant rates for {len(grant_rates['detailed'])} topics")
    return grant_rates

def normalize_by_model_mean(grant_rates: Dict) -> Dict:
    """Normalize all grant rates by overall model mean"""
    logger.info("Normalizing grant rates by model means")
    
    model_means = grant_rates['model_overall']
    normalized_data = {}
    
    for topic in grant_rates['detailed']:
        normalized_data[topic] = {}
        for field in grant_rates['detailed'][topic]:
            normalized_data[topic][field] = {}
            for field_value in grant_rates['detailed'][topic][field]:
                normalized_data[topic][field][field_value] = {}
                for model in grant_rates['detailed'][topic][field][field_value]:
                    data = grant_rates['detailed'][topic][field][field_value][model]
                    
                    # Skip if insufficient data
                    if data['total'] < 30:  # Minimum sample size
                        continue
                    
                    # Normalize by model mean
                    model_mean = model_means[model]
                    normalized_score = (data['grant_rate'] - model_mean) / model_mean if model_mean > 0 else 0
                    
                    normalized_data[topic][field][field_value][model] = {
                        'normalized_score': normalized_score,
                        'raw_grant_rate': data['grant_rate'],
                        'grants': data['grants'],
                        'total': data['total']
                    }
    
    logger.info("Normalization complete")
    return normalized_data, model_means

def calculate_topic_tendencies(normalized_data: Dict) -> Dict:
    """Calculate aggregated model tendency per topic"""
    logger.info("Calculating topic-level tendencies")
    
    topic_tendencies = {}
    
    for topic in normalized_data:
        topic_tendencies[topic] = {}
        
        # Collect all normalized scores for each model in this topic
        model_scores = defaultdict(list)
        
        for field in normalized_data[topic]:
            for field_value in normalized_data[topic][field]:
                for model in normalized_data[topic][field][field_value]:
                    score = normalized_data[topic][field][field_value][model]['normalized_score']
                    model_scores[model].append(score)
        
        # Calculate average tendency for each model
        for model in model_scores:
            if model_scores[model]:
                topic_tendencies[topic][model] = {
                    'avg_normalized_score': np.mean(model_scores[model]),
                    'std_normalized_score': np.std(model_scores[model]),
                    'num_comparisons': len(model_scores[model])
                }
    
    logger.info(f"Calculated tendencies for {len(topic_tendencies)} topics")
    return topic_tendencies

def calculate_within_topic_preferences(normalized_data: Dict) -> Dict:
    """Calculate relative tendency within topic to different completions"""
    logger.info("Calculating within-topic preferences")
    
    within_topic_prefs = {}
    
    for topic in normalized_data:
        within_topic_prefs[topic] = {}
        
        for field in normalized_data[topic]:
            within_topic_prefs[topic][field] = {}
            
            # Collect all field values and their normalized scores
            field_value_scores = {}
            
            for field_value in normalized_data[topic][field]:
                field_value_scores[field_value] = {}
                for model in normalized_data[topic][field][field_value]:
                    data = normalized_data[topic][field][field_value][model]
                    field_value_scores[field_value][model] = data['normalized_score']
            
            # Calculate rankings and preferences
            for model in ['pre_brexit', 'post_brexit']:
                model_scores = []
                for field_value in field_value_scores:
                    if model in field_value_scores[field_value]:
                        model_scores.append((field_value, field_value_scores[field_value][model]))
                
                # Sort by normalized score (descending)
                model_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Store rankings and preferences
                within_topic_prefs[topic][field][model] = {
                    'rankings': [(fv, score, rank+1) for rank, (fv, score) in enumerate(model_scores)],
                    'preferences': {fv: score for fv, score in model_scores}
                }
    
    logger.info("Within-topic preferences calculated")
    return within_topic_prefs

def create_detailed_results_dataframe(normalized_data: Dict, topic_fields: Dict) -> pd.DataFrame:
    """Create detailed results dataframe"""
    logger.info("Creating detailed results dataframe")
    
    # Create mapping of field names to types based on common patterns
    ordinal_fields = {
        'duration', 'emplacement', 'Working status', 'systems', 'connection to country condition',
        'degree of safety condition', 'asylum seeker circumstances', 'place of interception',
        'financial state', 'work intentions', 'profession', 'education intentions',
        'hardship', 'safety and systems', 'assimilation_prospect'
    }
    
    rows = []
    
    for topic in normalized_data:
        for field in normalized_data[topic]:
            # Determine field type
            field_type = 'ordinal' if field in ordinal_fields else 'horizontal'
            
            for field_value in normalized_data[topic][field]:
                # Check if we have both models for comparison
                models_data = normalized_data[topic][field][field_value]
                
                if 'pre_brexit' in models_data and 'post_brexit' in models_data:
                    pre_data = models_data['pre_brexit']
                    post_data = models_data['post_brexit']
                    
                    # Calculate cross-model difference
                    cross_model_diff = pre_data['normalized_score'] - post_data['normalized_score']
                    
                    # Statistical significance test
                    contingency_table = [
                        [pre_data['grants'], pre_data['total'] - pre_data['grants']],
                        [post_data['grants'], post_data['total'] - post_data['grants']]
                    ]
                    
                    try:
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                        is_significant = p_value < 0.05
                    except:
                        is_significant = False
                        p_value = 1.0
                    
                    rows.append({
                        'topic': topic,
                        'field_name': field,
                        'field_type': field_type,
                        'field_value': field_value,
                        'pre_brexit_normalized': pre_data['normalized_score'],
                        'post_brexit_normalized': post_data['normalized_score'],
                        'cross_model_difference': cross_model_diff,
                        'pre_brexit_raw_rate': pre_data['raw_grant_rate'],
                        'post_brexit_raw_rate': post_data['raw_grant_rate'],
                        'pre_brexit_sample_size': pre_data['total'],
                        'post_brexit_sample_size': post_data['total'],
                        'statistical_significance': is_significant,
                        'p_value': p_value,
                        'favors_model': 'pre_brexit' if cross_model_diff > 0 else 'post_brexit'
                    })
    
    df = pd.DataFrame(rows)
    logger.info(f"Created detailed results dataframe with {len(df)} rows")
    return df

def create_topic_tendencies_dataframe(topic_tendencies: Dict) -> pd.DataFrame:
    """Create topic tendencies dataframe"""
    logger.info("Creating topic tendencies dataframe")
    
    rows = []
    
    for topic in topic_tendencies:
        row = {'topic': topic}
        
        for model in ['pre_brexit', 'post_brexit']:
            if model in topic_tendencies[topic]:
                data = topic_tendencies[topic][model]
                row[f'{model}_topic_tendency'] = data['avg_normalized_score']
                row[f'{model}_tendency_std'] = data['std_normalized_score']
                row[f'{model}_num_comparisons'] = data['num_comparisons']
        
        if 'pre_brexit_topic_tendency' in row and 'post_brexit_topic_tendency' in row:
            row['topic_tendency_difference'] = row['pre_brexit_topic_tendency'] - row['post_brexit_topic_tendency']
            row['favors_model'] = 'pre_brexit' if row['topic_tendency_difference'] > 0 else 'post_brexit'
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    logger.info(f"Created topic tendencies dataframe with {len(df)} rows")
    return df

def main():
    """Main analysis function"""
    logger.info("Starting Grant Rate Analysis by Vignette-Specific Fields")
    
    # File paths
    vignette_file = "/data/shil6369/vignettes/complete_vignettes.json"
    records_file = "../../../data/processed/tagged_records.json"
    output_dir = Path("../../../outputs/grant_rate_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load vignette structure
    topic_fields = load_vignette_structure(vignette_file)
    
    # Step 2: Load tagged records
    records = load_tagged_records(records_file)
    
    # Step 3: Calculate grant rates
    grant_rates = calculate_grant_rates(records, topic_fields)
    
    # Step 4: Normalize by model mean
    normalized_data, model_means = normalize_by_model_mean(grant_rates)
    
    # Step 5: Calculate topic tendencies
    topic_tendencies = calculate_topic_tendencies(normalized_data)
    
    # Step 6: Calculate within-topic preferences
    within_topic_prefs = calculate_within_topic_preferences(normalized_data)
    
    # Step 7: Create output dataframes
    detailed_df = create_detailed_results_dataframe(normalized_data, topic_fields)
    topic_tendencies_df = create_topic_tendencies_dataframe(topic_tendencies)
    
    # Step 8: Save outputs
    detailed_df.to_csv(output_dir / "grant_rate_analysis_by_vignette_fields.csv", index=False)
    topic_tendencies_df.to_csv(output_dir / "topic_tendencies_analysis.csv", index=False)
    
    # Save summary statistics
    summary_stats = {
        'model_overall_grant_rates': model_means,
        'total_records_analyzed': len(records),
        'topics_analyzed': len(topic_fields),
        'total_field_comparisons': len(detailed_df),
        'significant_differences': len(detailed_df[detailed_df['statistical_significance']]),
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(output_dir / "grant_rate_analysis_summary.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    logger.info("Analysis complete!")
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Total comparisons: {len(detailed_df)}")
    logger.info(f"Significant differences: {len(detailed_df[detailed_df['statistical_significance']])}")

if __name__ == "__main__":
    main() 