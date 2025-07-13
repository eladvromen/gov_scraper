"""
Load and tag records from both models for comparative fairness analysis
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_utils import (
    load_config, load_inference_data, tag_record, 
    validate_data_quality, get_reference_groups, setup_logging
)

def main():
    """Main function to load and tag records from both models"""
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / "config" / "analysis_config.yaml"
    output_dir = base_dir / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = base_dir / "outputs" / "logs" / "data_preprocessing.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(log_file))
    
    logger.info("Starting data loading and tagging process")
    
    # Load configuration
    config = load_config(config_path)
    logger.info(f"Loaded configuration with {len(config['protected_attributes'])} protected attributes")
    
    # Load both models' data
    logger.info("Loading Post-Brexit model data...")
    post_brexit_data = load_inference_data(config['data_paths']['post_brexit'])
    post_brexit_records = post_brexit_data['processed_results']
    logger.info(f"Loaded {len(post_brexit_records)} Post-Brexit records")
    
    logger.info("Loading Pre-Brexit model data...")
    pre_brexit_data = load_inference_data(config['data_paths']['pre_brexit'])
    pre_brexit_records = pre_brexit_data['processed_results']
    logger.info(f"Loaded {len(pre_brexit_records)} Pre-Brexit records")
    
    # Tag records
    logger.info("Tagging records...")
    tagged_records = []
    
    # Tag Post-Brexit records
    for record in post_brexit_records:
        tagged_record = tag_record(record, 'post_brexit')
        tagged_records.append(tagged_record)
    
    # Tag Pre-Brexit records
    for record in pre_brexit_records:
        tagged_record = tag_record(record, 'pre_brexit')
        tagged_records.append(tagged_record)
    
    logger.info(f"Tagged {len(tagged_records)} total records")
    
    # Validate data quality
    logger.info("Validating data quality...")
    quality_report = validate_data_quality(tagged_records)
    
    logger.info("Data Quality Report:")
    logger.info(f"  Total records: {quality_report['total_records']}")
    logger.info(f"  Missing decisions: {quality_report['missing_decisions']} ({quality_report['missing_decisions_pct']:.2f}%)")
    logger.info(f"  Missing topics: {quality_report['missing_topics']} ({quality_report['missing_topics_pct']:.2f}%)")
    for attr, pct in quality_report['missing_attributes_pct'].items():
        logger.info(f"  Missing {attr}: {quality_report['missing_attributes'][attr]} ({pct:.2f}%)")
    
    # Determine reference groups
    logger.info("Determining reference groups...")
    reference_groups = get_reference_groups(tagged_records, config)
    logger.info("Reference groups:")
    for attr, ref_group in reference_groups.items():
        logger.info(f"  {attr}: {ref_group}")
    
    # Update config with determined reference groups
    config['reference_groups'] = reference_groups
    
    # Get topic and model distribution
    logger.info("Analyzing data distribution...")
    
    # Topic distribution
    topic_counts = {}
    model_counts = {'post_brexit': 0, 'pre_brexit': 0}
    
    for record in tagged_records:
        topic = record['topic']
        model = record['model']
        
        if topic not in topic_counts:
            topic_counts[topic] = {'post_brexit': 0, 'pre_brexit': 0, 'total': 0}
        
        topic_counts[topic][model] += 1
        topic_counts[topic]['total'] += 1
        model_counts[model] += 1
    
    logger.info(f"Model distribution: Post-Brexit: {model_counts['post_brexit']}, Pre-Brexit: {model_counts['pre_brexit']}")
    logger.info(f"Found {len(topic_counts)} unique topics")
    
    # Log topics with counts
    for topic, counts in sorted(topic_counts.items()):
        logger.info(f"  {topic}: {counts['total']} total ({counts['post_brexit']} post, {counts['pre_brexit']} pre)")
    
    # Protected attribute distribution
    logger.info("Protected attribute distributions:")
    for attr in config['protected_attributes']:
        attr_counts = {}
        for record in tagged_records:
            value = record['protected_attributes'][attr]
            attr_counts[value] = attr_counts.get(value, 0) + 1
        
        logger.info(f"  {attr}: {len(attr_counts)} unique values")
        # Log top 5 most frequent values
        top_values = sorted(attr_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for value, count in top_values:
            logger.info(f"    {value}: {count} ({count/len(tagged_records)*100:.1f}%)")
    
    # Save results
    logger.info("Saving processed data...")
    
    # Save tagged records
    tagged_records_file = output_dir / "tagged_records.json"
    with open(tagged_records_file, 'w') as f:
        json.dump(tagged_records, f, indent=2)
    logger.info(f"Saved tagged records to {tagged_records_file}")
    
    # Save quality report
    quality_report_file = output_dir / "data_quality_report.json"
    with open(quality_report_file, 'w') as f:
        json.dump(quality_report, f, indent=2)
    logger.info(f"Saved quality report to {quality_report_file}")
    
    # Save updated config
    config_file = output_dir / "analysis_config_updated.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved updated config to {config_file}")
    
    # Save distribution summary
    distribution_summary = {
        'total_records': len(tagged_records),
        'model_counts': model_counts,
        'topic_counts': topic_counts,
        'reference_groups': reference_groups,
        'protected_attributes': {
            attr: {
                'unique_values': len(set(r['protected_attributes'][attr] for r in tagged_records)),
                'most_frequent': reference_groups[attr]
            }
            for attr in config['protected_attributes']
        }
    }
    
    distribution_file = output_dir / "data_distribution_summary.json"
    with open(distribution_file, 'w') as f:
        json.dump(distribution_summary, f, indent=2)
    logger.info(f"Saved distribution summary to {distribution_file}")
    
    logger.info("Data loading and tagging completed successfully!")
    
    return {
        'tagged_records': tagged_records,
        'quality_report': quality_report,
        'config': config,
        'distribution_summary': distribution_summary
    }

if __name__ == "__main__":
    results = main() 