    #!/usr/bin/env python3
"""
Grant Rate Analysis by Vignette-Specific Fields - Fixed Statistical Testing
FIXED: Now uses proper McNemar's test for paired binary data
ENHANCED: Includes Disclosure and Contradiction vignettes
BACKWARD COMPATIBLE: Output format identical to original

NEW STATISTICAL APPROACH:
- Collects paired decisions during aggregation
- Uses McNemar's test for proper paired statistical testing
- Maintains identical output columns for backward compatibility
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import logging
from scipy.stats import chi2_contingency, chi2
import warnings

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

def is_disclosure_contradiction_topic(topic: str) -> Tuple[bool, str, str]:
    """Determine if topic is disclosure/contradiction and extract details"""
    if topic.startswith("Disclosure:"):
        return True, "Disclosure", topic
    elif topic.startswith("Contradiction:"):
        return True, "Contradiction", topic
    else:
        return False, "", ""

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
            f'{category.lower()}_type': specific_type
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

def create_vignette_identifier(record: Dict, field_name: str, field_value: Any) -> str:
    """Create unique identifier for a vignette to enable proper pairing"""
    original_record = record.get('original_record', {})
    metadata = original_record.get('metadata', {})
    fields = metadata.get('fields', {})
    
    # Create identifier based on all field values to ensure same vignette
    # Sort fields for consistent ordering
    sorted_fields = sorted(fields.items())
    field_string = "|".join(f"{k}:{v}" for k, v in sorted_fields)
    
    return f"{record['topic']}|{field_name}|{field_value}|{field_string}"

def calculate_grant_rates(records: List[Dict], topic_fields: Dict) -> Tuple[Dict, Dict]:
    """
    Calculate grant rates for each (topic, field, field_value, model) combination
    ENHANCED: Now includes disclosure/contradiction processing
    FIXED: Also collects paired decisions for proper statistical testing
    """
    logger.info("Calculating grant rates and collecting paired decisions for statistical testing")
    
    # Structure: topic -> field -> field_value -> model -> {grants, total}
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'grants': 0, 'total': 0}))))
    
    # NEW: Collect individual decisions for paired testing
    # Structure: topic -> field -> field_value -> vignette_id -> {pre_decision, post_decision}
    paired_decisions = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    
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
            
            # 2. Collect paired decisions
            vignette_id = create_vignette_identifier(record, field_name, specific_type)
            paired_decisions[topic][field_name][specific_type][vignette_id][model] = decision
            
            # 3. Update aggregated stats for topic-level analysis
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
                
                # Collect paired decisions
                vignette_id = create_vignette_identifier(record, field_name, field_value)
                paired_decisions[topic][field_name][field_value][vignette_id][model] = decision
    
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
    logger.info(f"Collected paired decisions for proper statistical testing")
    return grant_rates, paired_decisions

def mcnemar_test_fixed(paired_decisions_data: Dict[str, Dict[str, str]]) -> Tuple[bool, float]:
    """
    Perform McNemar's test on paired binary decisions
    Returns: (is_significant, p_value)
    """
    
    # Count agreement and disagreement patterns
    both_grant = 0      # a: both models grant
    pre_grant_post_reject = 0  # b: pre grants, post rejects  
    pre_reject_post_grant = 0  # c: pre rejects, post grants
    both_reject = 0     # d: both models reject
    
    for vignette_id, decisions in paired_decisions_data.items():
        # Only include vignettes with decisions from both models
        if 'pre_brexit' in decisions and 'post_brexit' in decisions:
            pre_decision = decisions['pre_brexit']
            post_decision = decisions['post_brexit']
            
            if pre_decision == 'GRANT' and post_decision == 'GRANT':
                both_grant += 1
            elif pre_decision == 'GRANT' and post_decision == 'DENY':
                pre_grant_post_reject += 1
            elif pre_decision == 'DENY' and post_decision == 'GRANT':
                pre_reject_post_grant += 1
            elif pre_decision == 'DENY' and post_decision == 'DENY':
                both_reject += 1
    
    # Create McNemar's 2x2 table
    # [[both_grant, pre_grant_post_reject], [pre_reject_post_grant, both_reject]]
    mcnemar_table = np.array([
        [both_grant, pre_grant_post_reject],
        [pre_reject_post_grant, both_reject]
    ])
    
    # Check if we have enough disagreements for McNemar's test
    total_disagreements = pre_grant_post_reject + pre_reject_post_grant
    
    if total_disagreements < 10:
        # Use exact binomial test for small samples
        from scipy.stats import binom_test
        if total_disagreements == 0:
            # No disagreements = no difference
            return False, 1.0
        else:
            # Test if disagreements are balanced (p=0.5)
            p_value = binom_test(min(pre_grant_post_reject, pre_reject_post_grant), 
                                total_disagreements, 0.5, alternative='two-sided')
            return p_value < 0.05, p_value
    else:
        # Use standard McNemar's test (implemented manually)
        try:
            # McNemar's test statistic with continuity correction
            # Formula: ((|b - c| - 0.5)^2) / (b + c)
            # where b = pre_grant_post_reject, c = pre_reject_post_grant
            
            b = pre_grant_post_reject
            c = pre_reject_post_grant
            
            if (b + c) == 0:
                # No disagreements = no difference
                return False, 1.0
            
            # McNemar's test statistic with continuity correction
            test_statistic = ((abs(b - c) - 0.5) ** 2) / (b + c)
            
            # p-value from chi-square distribution with 1 degree of freedom
            p_value = 1 - chi2.cdf(test_statistic, df=1)
            
            return p_value < 0.05, p_value
        except Exception as e:
            logger.warning(f"McNemar's test failed: {e}. Falling back to chi-square.")
            # Fallback to chi-square if McNemar's fails
            try:
                chi2_stat, p_value, dof, expected = chi2_contingency(mcnemar_table)
                return p_value < 0.05, p_value
            except:
                return False, 1.0

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
    """Calculate overall topic tendency for each model"""
    logger.info("Calculating topic tendencies")
    
    topic_tendencies = {}
    
    for topic in normalized_data:
        model_scores = defaultdict(list)
        
        for field in normalized_data[topic]:
            for field_value in normalized_data[topic][field]:
                for model in normalized_data[topic][field][field_value]:
                    data = normalized_data[topic][field][field_value][model]
                    model_scores[model].append(data['normalized_score'])
        
        # Calculate mean tendency for each model
        for model in model_scores:
            if model_scores[model]:
                topic_tendencies[f"{topic}_{model}"] = np.mean(model_scores[model])
    
    logger.info("Topic tendencies calculated")
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

def create_topic_tendencies_dataframe(topic_tendencies: Dict) -> pd.DataFrame:
    """Create topic tendencies dataframe with enhanced disclosure/contradiction"""
    logger.info("Creating topic tendencies dataframe")
    
    # Group by topic
    topic_data = {}
    for key, tendency in topic_tendencies.items():
        if key.endswith('_pre_brexit'):
            topic = key.replace('_pre_brexit', '')
            if topic not in topic_data:
                topic_data[topic] = {}
            topic_data[topic]['pre_brexit'] = tendency
        elif key.endswith('_post_brexit'):
            topic = key.replace('_post_brexit', '')
            if topic not in topic_data:
                topic_data[topic] = {}
            topic_data[topic]['post_brexit'] = tendency
    
    rows = []
    for topic, data in topic_data.items():
        if 'pre_brexit' in data and 'post_brexit' in data:
            row = {
                'topic': topic,
                'pre_brexit_topic_tendency': data['pre_brexit'],
                'post_brexit_topic_tendency': data['post_brexit'],
                'topic_tendency_difference': data['pre_brexit'] - data['post_brexit']
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    logger.info(f"Created topic tendencies dataframe with {len(df)} topics")
    return df

def create_detailed_results_dataframe(normalized_data: Dict, topic_fields: Dict, paired_decisions: Dict) -> pd.DataFrame:
    """
    Create detailed results dataframe with all comparisons
    FIXED: Now uses proper McNemar's test for statistical significance
    BACKWARD COMPATIBLE: Output format identical to original
    """
    logger.info("Creating detailed results dataframe with fixed statistical testing")
    
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
                
                # FIXED: Statistical significance test using McNemar's test
                try:
                    # Get paired decisions for this field comparison
                    field_paired_data = paired_decisions.get(topic, {}).get(field, {}).get(field_value, {})
                    
                    if field_paired_data:
                        # Use proper McNemar's test for paired data
                        is_significant, p_value = mcnemar_test_fixed(field_paired_data)
                    else:
                        # Fallback: no paired data available
                        logger.warning(f"No paired data for {topic}|{field}|{field_value}. Using old method.")
                        contingency = [
                            [pre_data['grants'], pre_data['total'] - pre_data['grants']],
                            [post_data['grants'], post_data['total'] - post_data['grants']]
                        ]
                        chi2, p_value, dof, expected = chi2_contingency(contingency)
                        is_significant = p_value < 0.05
                except Exception as e:
                    logger.warning(f"Statistical test failed for {topic}|{field}|{field_value}: {e}")
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
                    'statistical_significance': is_significant,  # SAME OUTPUT FORMAT
                    'p_value': p_value,                         # SAME OUTPUT FORMAT
                    'favors_model': favors_model
                }
                
                rows.append(row)
    
    df = pd.DataFrame(rows)
    logger.info(f"Created detailed dataframe with {len(df)} comparisons using FIXED statistical testing")
    logger.info(f"Significant differences: {df['statistical_significance'].sum()}")
    return df

def main():
    """Main analysis function with FIXED statistical testing"""
    logger.info("Starting FIXED Grant Rate Analysis (proper McNemar's test + Disclosure/Contradiction support)")
    
    # File paths
    vignette_file = "/data/shil6369/vignettes/complete_vignettes.json"
    records_file = "/data/shil6369/gov_scraper/vignettes_analysis/comparative_fairness/data/processed/tagged_records.json"
    output_dir = Path("../outputs/grant_rate_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load vignette structure
    topic_fields = load_vignette_structure(vignette_file)
    
    # Step 2: Load tagged records
    records = load_tagged_records(records_file)
    
    # Step 3: Calculate grant rates AND collect paired decisions (FIXED)
    grant_rates, paired_decisions = calculate_grant_rates(records, topic_fields)
    
    # Step 4: Normalize by model mean
    normalized_data, model_means = normalize_by_model_mean(grant_rates)
    
    # Step 5: Calculate topic tendencies (including aggregated disclosure/contradiction)
    topic_tendencies = calculate_topic_tendencies(normalized_data)
    
    # Step 6: Calculate within-topic preferences
    within_topic_prefs = calculate_within_topic_preferences(normalized_data)
    
    # Step 7: Create output dataframes with FIXED statistical testing
    detailed_df = create_detailed_results_dataframe(normalized_data, topic_fields, paired_decisions)
    topic_tendencies_df = create_topic_tendencies_dataframe(topic_tendencies)
    
    # Step 8: Save FIXED outputs (backward compatible filenames)
    detailed_df.to_csv(output_dir / "grant_rate_analysis_by_vignette_fields_enhanced_FIXED.csv", index=False)
    topic_tendencies_df.to_csv(output_dir / "topic_tendencies_analysis_enhanced_FIXED.csv", index=False)
    
    # Save FIXED summary statistics
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
        'statistical_testing': {
            'method': 'McNemar_test',
            'description': 'Proper paired statistical testing for binary outcomes',
            'fallback': 'Exact_binomial_test_for_small_samples',
            'previous_method': 'Chi_square_independence (INCORRECT for paired data)'
        },
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(output_dir / "grant_rate_analysis_summary_enhanced_FIXED.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    logger.info("FIXED Analysis complete!")
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Total comparisons: {len(detailed_df)}")
    logger.info(f"Significant differences (CORRECTED): {len(detailed_df[detailed_df['statistical_significance']])}")
    logger.info(f"Statistical method: McNemar's test (proper paired testing)")
    
    # Show comparison with old results if available
    try:
        old_df = pd.read_csv(output_dir / "grant_rate_analysis_by_vignette_fields_enhanced.csv")
        old_sig_count = old_df['statistical_significance'].sum()
        new_sig_count = detailed_df['statistical_significance'].sum()
        
        logger.info(f"COMPARISON:")
        logger.info(f"  Old method (chi-square): {old_sig_count} significant")
        logger.info(f"  New method (McNemar's):  {new_sig_count} significant")
        logger.info(f"  Difference: {new_sig_count - old_sig_count:+d}")
        
    except FileNotFoundError:
        logger.info("No old results to compare against")

if __name__ == "__main__":
    main() 