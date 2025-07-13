"""
Detailed data quality check with focus on gender distribution across topics
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
from collections import defaultdict, Counter

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_utils import load_config, setup_logging

def analyze_gender_distribution_by_topic(records: List[Dict]) -> Dict:
    """Analyze gender distribution across all topics"""
    
    topic_gender_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for record in records:
        topic = record['topic']
        model = record['model']
        gender = record['protected_attributes']['gender']
        
        topic_gender_counts[topic][model][gender] += 1
    
    # Convert to regular dict and calculate percentages
    analysis = {}
    
    for topic in topic_gender_counts:
        topic_analysis = {
            'post_brexit': {},
            'pre_brexit': {},
            'combined': {}
        }
        
        # Per model analysis
        for model in ['post_brexit', 'pre_brexit']:
            model_counts = topic_gender_counts[topic][model]
            total_model = sum(model_counts.values())
            
            topic_analysis[model] = {
                'total': total_model,
                'gender_counts': dict(model_counts),
                'gender_percentages': {
                    gender: (count/total_model*100) if total_model > 0 else 0
                    for gender, count in model_counts.items()
                }
            }
        
        # Combined analysis
        combined_counts = defaultdict(int)
        for model in ['post_brexit', 'pre_brexit']:
            for gender, count in topic_gender_counts[topic][model].items():
                combined_counts[gender] += count
        
        total_combined = sum(combined_counts.values())
        topic_analysis['combined'] = {
            'total': total_combined,
            'gender_counts': dict(combined_counts),
            'gender_percentages': {
                gender: (count/total_combined*100) if total_combined > 0 else 0
                for gender, count in combined_counts.items()
            }
        }
        
        analysis[topic] = topic_analysis
    
    return analysis

def check_intersectional_sample_sizes(records: List[Dict], min_size: int = 30) -> Dict:
    """Check sample sizes for all intersectional groups"""
    
    # Create group labels first
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.fairness_utils import create_group_labels
    
    labeled_records = create_group_labels(records, ['country', 'age', 'religion', 'gender'])
    
    intersections = [
        'gender_x_religion', 'gender_x_country', 'religion_x_country', 
        'age_x_gender', 'religion_x_country_x_gender'
    ]
    
    analysis = {}
    
    for intersection in intersections:
        intersection_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        for record in labeled_records:
            topic = record['topic']
            model = record['model']
            intersection_value = record[intersection]
            
            intersection_counts[topic][model][intersection_value] += 1
        
        # Check which groups meet minimum size requirements
        valid_groups = defaultdict(lambda: defaultdict(list))
        insufficient_groups = defaultdict(lambda: defaultdict(list))
        
        for topic in intersection_counts:
            for intersection_value in set().union(
                intersection_counts[topic]['post_brexit'].keys(),
                intersection_counts[topic]['pre_brexit'].keys()
            ):
                post_count = intersection_counts[topic]['post_brexit'].get(intersection_value, 0)
                pre_count = intersection_counts[topic]['pre_brexit'].get(intersection_value, 0)
                
                group_info = {
                    'intersection_value': intersection_value,
                    'post_brexit_count': post_count,
                    'pre_brexit_count': pre_count,
                    'total_count': post_count + pre_count
                }
                
                if post_count >= min_size and pre_count >= min_size:
                    valid_groups[topic][intersection].append(group_info)
                else:
                    insufficient_groups[topic][intersection].append(group_info)
        
        analysis[intersection] = {
            'valid_groups': dict(valid_groups),
            'insufficient_groups': dict(insufficient_groups)
        }
    
    return analysis

def identify_topics_without_nonbinary(records: List[Dict]) -> List[str]:
    """Identify topics that don't have non-binary gender representation"""
    
    topics_without_nonbinary = []
    topic_genders = defaultdict(set)
    
    for record in records:
        topic = record['topic']
        gender = record['protected_attributes']['gender']
        topic_genders[topic].add(gender)
    
    for topic, genders in topic_genders.items():
        if 'Non-binary' not in genders:
            topics_without_nonbinary.append(topic)
    
    return topics_without_nonbinary

def main():
    """Main function for detailed data quality check"""
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / "data" / "processed"
    output_dir = base_dir / "outputs" / "quality_checks"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = base_dir / "outputs" / "logs" / "data_quality_check.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(log_file))
    
    logger.info("Starting detailed data quality check")
    
    # Load processed data
    tagged_records_file = processed_dir / "tagged_records.json"
    with open(tagged_records_file, 'r') as f:
        records = json.load(f)
    
    logger.info(f"Loaded {len(records)} records for quality check")
    
    # 1. Analyze gender distribution by topic
    logger.info("Analyzing gender distribution by topic...")
    gender_analysis = analyze_gender_distribution_by_topic(records)
    
    # Log summary
    topics_without_nonbinary = identify_topics_without_nonbinary(records)
    
    if topics_without_nonbinary:
        logger.warning(f"Found {len(topics_without_nonbinary)} topics without non-binary representation:")
        for topic in topics_without_nonbinary:
            logger.warning(f"  - {topic}")
    else:
        logger.info("All topics have non-binary representation")
    
    # 2. Check intersectional sample sizes
    logger.info("Checking intersectional sample sizes...")
    intersectional_analysis = check_intersectional_sample_sizes(records)
    
    # Log intersectional summary
    for intersection, analysis in intersectional_analysis.items():
        valid_topics = len(analysis['valid_groups'])
        insufficient_topics = len(analysis['insufficient_groups'])
        
        logger.info(f"{intersection}:")
        logger.info(f"  Topics with sufficient samples: {valid_topics}")
        logger.info(f"  Topics with insufficient samples: {insufficient_topics}")
        
        # Count total valid groups
        total_valid_groups = sum(
            len(groups) for groups in analysis['valid_groups'].values()
        )
        logger.info(f"  Total valid groups: {total_valid_groups}")
    
    # 3. Detailed gender breakdown
    logger.info("Detailed gender breakdown by topic:")
    for topic, analysis in gender_analysis.items():
        combined = analysis['combined']
        genders = combined['gender_counts']
        
        logger.info(f"  {topic} (Total: {combined['total']}):")
        for gender, count in genders.items():
            pct = combined['gender_percentages'][gender]
            logger.info(f"    {gender}: {count} ({pct:.1f}%)")
    
    # Save detailed results
    detailed_results = {
        'gender_distribution_by_topic': gender_analysis,
        'intersectional_sample_sizes': intersectional_analysis,
        'topics_without_nonbinary': topics_without_nonbinary,
        'summary': {
            'total_topics': len(gender_analysis),
            'topics_without_nonbinary': len(topics_without_nonbinary),
            'intersectional_coverage': {
                intersection: {
                    'topics_with_valid_groups': len(analysis['valid_groups']),
                    'topics_with_insufficient_groups': len(analysis['insufficient_groups']),
                    'total_valid_groups': sum(
                        len(groups) for groups in analysis['valid_groups'].values()
                    )
                }
                for intersection, analysis in intersectional_analysis.items()
            }
        }
    }
    
    # Save results
    results_file = output_dir / "detailed_quality_check.json"
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    logger.info(f"Saved detailed quality check results to {results_file}")
    
    # Create summary report
    summary_file = output_dir / "quality_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=== Data Quality Check Summary ===\n\n")
        f.write(f"Total Records: {len(records)}\n")
        f.write(f"Total Topics: {len(gender_analysis)}\n\n")
        
        f.write("=== Gender Distribution Issues ===\n")
        if topics_without_nonbinary:
            f.write(f"Topics without non-binary: {len(topics_without_nonbinary)}\n")
            for topic in topics_without_nonbinary:
                f.write(f"  - {topic}\n")
        else:
            f.write("All topics have non-binary representation\n")
        
        f.write("\n=== Intersectional Analysis Coverage ===\n")
        for intersection, summary in detailed_results['summary']['intersectional_coverage'].items():
            f.write(f"{intersection}:\n")
            f.write(f"  Topics with valid groups: {summary['topics_with_valid_groups']}\n")
            f.write(f"  Total valid groups: {summary['total_valid_groups']}\n")
        
        f.write("\n=== Recommendations ===\n")
        if topics_without_nonbinary:
            f.write("- Consider gender-specific analysis for topics without non-binary representation\n")
        
        f.write("- Use the intersectional coverage data to prioritize analysis focus\n")
        f.write("- Consider relaxing minimum sample size for three-way intersections if needed\n")
    
    logger.info(f"Saved summary report to {summary_file}")
    logger.info("Data quality check completed successfully!")
    
    return detailed_results

if __name__ == "__main__":
    results = main() 