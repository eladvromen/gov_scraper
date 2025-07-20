#!/usr/bin/env python3
"""
Grant Rate Analysis by Vignette-Specific Fields - Enhanced Version
ENHANCED: Now includes Disclosure and Contradiction vignettes
Analyzes how different case characteristics affect grant rates
across Pre-Brexit vs Post-Brexit models.

NEW FEATURES:
- Handles disclosure/contradiction vignettes (which have empty ordinal/horizontal fields)
- Treats each disclosure/contradiction type as a "field variation"
- Aggregates into broader "Disclosure" and "Contradiction" topic categories
- Adds 2 topics and 10 granular dimensions to the normative vectors
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

def is_disclosure_contradiction_topic(topic: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if topic is disclosure/contradiction and categorize it
    
    Returns:
        (is_special, category, specific_type)
        - is_special: True if disclosure/contradiction
        - category: "Disclosure" or "Contradiction" 
        - specific_type: specific disclosure/contradiction type
    """
    if topic.startswith("Disclosure:"):
        return True, "Disclosure", topic
    elif topic.startswith("Contradiction:"):
        return True, "Contradiction", topic
    else:
        return False, None, None

def extract_field_values(record: Dict, topic_fields: Dict) -> Dict[str, Any]:
    """
    Extract vignette-specific field values from a record
    ENHANCED: Now handles disclosure/contradiction topics specially
    """
    topic = record['topic']
    
    # Check if this is a disclosure/contradiction topic
    is_special, category, specific_type = is_disclosure_contradiction_topic(topic)
    
    if is_special:
        # For disclosure/contradiction topics, the topic itself IS the field value
        return {
            f'{category.lower()}_type': specific_type  # e.g., 'disclosure_type': 'Disclosure: Political persecution & sexual violence'
        }
    else:
        # Regular processing for normal topics
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
    """
    Calculate grant rates for each (topic, field, field_value, model) combination
    ENHANCED: Now includes disclosure/contradiction processing
    """
    logger.info("Calculating grant rates for all field combinations (including disclosure/contradiction)")
    
    # Structure: topic -> field -> field_value -> model -> {grants, total}
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'grants': 0, 'total': 0}))))
    
    # Also track overall model statistics
    model_stats = defaultdict(lambda: {'grants': 0, 'total': 0})
    
    # Track disclosure/contradiction aggregations for topic-level analysis
    disclosure_stats = defaultdict(lambda: {'grants': 0, 'total': 0})
    contradiction_stats = defaultdict(lambda: {'grants': 0, 'total': 0})
    
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
        
        # Check if this is disclosure/contradiction
        is_special, category, specific_type = is_disclosure_contradiction_topic(topic)
        
        if is_special:
            # Handle disclosure/contradiction topics specially
            
            # 1. Update granular stats (each specific type as a field value)
            field_name = f'{category.lower()}_type'
            stats[topic][field_name][specific_type][model]['total'] += 1
            if decision == 'GRANT':
                stats[topic][field_name][specific_type][model]['grants'] += 1
            
            # 2. Update aggregated stats for topic-level analysis
            if category == "Disclosure":
                disclosure_stats[model]['total'] += 1
                if decision == 'GRANT':
                    disclosure_stats[model]['grants'] += 1
            else:  # Contradiction
                contradiction_stats[model]['total'] += 1
                if decision == 'GRANT':
                    contradiction_stats[model]['grants'] += 1
        else:
            # Regular processing for normal topics
            field_values = extract_field_values(record, topic_fields)
            
            # Update statistics for each field
            for field_name, field_value in field_values.items():
                stats[topic][field_name][field_value][model]['total'] += 1
                if decision == 'GRANT':
                    stats[topic][field_name][field_value][model]['grants'] += 1
    
    # Add aggregated disclosure/contradiction as "topics"
    if disclosure_stats:
        for model in disclosure_stats:
            stats['Disclosure']['aggregated_type']['all_disclosures'][model] = disclosure_stats[model]
    
    if contradiction_stats:
        for model in contradiction_stats:
            stats['Contradiction']['aggregated_type']['all_contradictions'][model] = contradiction_stats[model]
    
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
    
    logger.info(f"Calculated grant rates for {len(grant_rates['detailed'])} topics (including disclosure/contradiction)")
    return grant_rates

def normalize_by_model_mean(grant_rates: Dict) -> Tuple[Dict, Dict]:
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
    """
    Calculate topic-level tendencies by averaging field comparisons
    ENHANCED: Now includes Disclosure and Contradiction as topic categories
    """
    logger.info("Calculating topic tendencies (including disclosure/contradiction aggregations)")
    
    topic_tendencies = {}
    
    for topic in normalized_data:
        topic_tendencies[topic] = {}
        
        # Collect all normalized scores for this topic
        for model in ['pre_brexit', 'post_brexit']:
            scores = []
            
            for field in normalized_data[topic]:
                for field_value in normalized_data[topic][field]:
                    if model in normalized_data[topic][field][field_value]:
                        scores.append(normalized_data[topic][field][field_value][model]['normalized_score'])
            
            if scores:
                topic_tendencies[topic][model] = {
                    'avg_normalized_score': np.mean(scores),
                    'std_normalized_score': np.std(scores),
                    'num_comparisons': len(scores)
                }
    
    logger.info(f"Calculated tendencies for {len(topic_tendencies)} topics")
    return topic_tendencies

def calculate_within_topic_preferences(normalized_data: Dict) -> Dict:
    """Calculate preferences within each topic"""
    logger.info("Calculating within-topic model preferences")
    
    preferences = {}
    
    for topic in normalized_data:
        topic_prefs = {'pre_brexit_favored': 0, 'post_brexit_favored': 0, 'neutral': 0}
        
        for field in normalized_data[topic]:
            for field_value in normalized_data[topic][field]:
                if 'pre_brexit' in normalized_data[topic][field][field_value] and 'post_brexit' in normalized_data[topic][field][field_value]:
                    pre_score = normalized_data[topic][field][field_value]['pre_brexit']['normalized_score']
                    post_score = normalized_data[topic][field][field_value]['post_brexit']['normalized_score']
                    
                    if pre_score > post_score:
                        topic_prefs['pre_brexit_favored'] += 1
                    elif post_score > pre_score:
                        topic_prefs['post_brexit_favored'] += 1
                    else:
                        topic_prefs['neutral'] += 1
        
        preferences[topic] = topic_prefs
    
    return preferences

def create_detailed_results_dataframe(normalized_data: Dict, topic_fields: Dict) -> pd.DataFrame:
    """Create detailed results dataframe with all comparisons"""
    logger.info("Creating detailed results dataframe")
    
    rows = []
    
    for topic in normalized_data:
        for field in normalized_data[topic]:
            for field_value in normalized_data[topic][field]:
                
                # Check if we have both models
                if 'pre_brexit' not in normalized_data[topic][field][field_value] or 'post_brexit' not in normalized_data[topic][field][field_value]:
                    continue
                
                pre_data = normalized_data[topic][field][field_value]['pre_brexit']
                post_data = normalized_data[topic][field][field_value]['post_brexit']
                
                # Determine field type
                is_special, category, specific_type = is_disclosure_contradiction_topic(topic)
                if is_special:
                    field_type = f"{category.lower()}_variation"
                else:
                    # Regular field type detection
                    topic_info = topic_fields.get(topic, {})
                    if field in topic_info.get('ordinal_fields', {}):
                        field_type = 'ordinal'
                    elif field in topic_info.get('horizontal_fields', {}):
                        field_type = 'horizontal'
                    else:
                        field_type = 'unknown'
                
                # Statistical significance test
                # Create contingency table: [[pre_grants, pre_rejects], [post_grants, post_rejects]]
                contingency = [
                    [pre_data['grants'], pre_data['total'] - pre_data['grants']],
                    [post_data['grants'], post_data['total'] - post_data['grants']]
                ]
                
                try:
                    chi2, p_value, dof, expected = chi2_contingency(contingency)
                    is_significant = p_value < 0.05
                except:
                    p_value = 1.0
                    is_significant = False
                
                # Determine which model is favored
                favors_model = 'pre_brexit' if pre_data['normalized_score'] > post_data['normalized_score'] else 'post_brexit'
                
                row = {
                    'topic': topic,
                    'field_name': field,
                    'field_type': field_type,
                    'field_value': field_value,
                    'pre_brexit_normalized': pre_data['normalized_score'],
                    'post_brexit_normalized': post_data['normalized_score'],
                    'cross_model_difference': pre_data['normalized_score'] - post_data['normalized_score'],
                    'pre_brexit_raw_rate': pre_data['raw_grant_rate'],
                    'post_brexit_raw_rate': post_data['raw_grant_rate'],
                    'pre_brexit_sample_size': pre_data['total'],
                    'post_brexit_sample_size': post_data['total'],
                    'statistical_significance': is_significant,
                    'p_value': p_value,
                    'favors_model': favors_model
                }
                
                rows.append(row)
    
    df = pd.DataFrame(rows)
    logger.info(f"Created detailed dataframe with {len(df)} comparisons")
    return df

def create_topic_tendencies_dataframe(topic_tendencies: Dict) -> pd.DataFrame:
    """Create topic tendencies summary dataframe"""
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
    logger.info("Starting ENHANCED Grant Rate Analysis (with Disclosure/Contradiction support)")
    
    # File paths
    vignette_file = "/data/shil6369/vignettes/complete_vignettes.json"
    records_file = "/data/shil6369/gov_scraper/vignettes_analysis/comparative_fairness/data/processed/tagged_records.json"
    output_dir = Path("../../../outputs/grant_rate_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load vignette structure
    topic_fields = load_vignette_structure(vignette_file)
    
    # Step 2: Load tagged records
    records = load_tagged_records(records_file)
    
    # Step 3: Calculate grant rates (including disclosure/contradiction)
    grant_rates = calculate_grant_rates(records, topic_fields)
    
    # Step 4: Normalize by model mean
    normalized_data, model_means = normalize_by_model_mean(grant_rates)
    
    # Step 5: Calculate topic tendencies (including aggregated disclosure/contradiction)
    topic_tendencies = calculate_topic_tendencies(normalized_data)
    
    # Step 6: Calculate within-topic preferences
    within_topic_prefs = calculate_within_topic_preferences(normalized_data)
    
    # Step 7: Create output dataframes
    detailed_df = create_detailed_results_dataframe(normalized_data, topic_fields)
    topic_tendencies_df = create_topic_tendencies_dataframe(topic_tendencies)
    
    # Step 8: Save enhanced outputs to correct directory
    correct_output_dir = Path("../outputs/grant_rate_analysis")
    correct_output_dir.mkdir(parents=True, exist_ok=True)
    
    detailed_df.to_csv(correct_output_dir / "grant_rate_analysis_by_vignette_fields_enhanced.csv", index=False)
    topic_tendencies_df.to_csv(correct_output_dir / "topic_tendencies_analysis_enhanced.csv", index=False)
    
    # Also save to original location for compatibility
    detailed_df.to_csv(output_dir / "grant_rate_analysis_by_vignette_fields_enhanced.csv", index=False)
    topic_tendencies_df.to_csv(output_dir / "topic_tendencies_analysis_enhanced.csv", index=False)
    
    # Save enhanced summary statistics
    summary_stats = {
        'model_overall_grant_rates': model_means,
        'total_records_analyzed': len(records),
        'topics_analyzed': len(topic_fields),
        'total_field_comparisons': len(detailed_df),
        'significant_differences': len(detailed_df[detailed_df['statistical_significance']]),
        'disclosure_topics_included': len([t for t in topic_tendencies if t.startswith('Disclosure:') or t == 'Disclosure']),
        'contradiction_topics_included': len([t for t in topic_tendencies if t.startswith('Contradiction:') or t == 'Contradiction']),
        'enhanced_features': {
            'disclosure_contradiction_support': True,
            'aggregated_topic_categories': ['Disclosure', 'Contradiction'],
            'granular_disclosure_contradiction_variations': 10
        },
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(correct_output_dir / "grant_rate_analysis_summary_enhanced.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Also save to original location for compatibility
    with open(output_dir / "grant_rate_analysis_summary_enhanced.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    logger.info("ENHANCED Analysis complete!")
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Total comparisons: {len(detailed_df)}")
    logger.info(f"Significant differences: {len(detailed_df[detailed_df['statistical_significance']])}")
    logger.info(f"Topics analyzed: {len(topic_tendencies)} (including Disclosure/Contradiction aggregations)")

if __name__ == "__main__":
    main() 