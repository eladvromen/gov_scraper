"""
Comprehensive Fairness Metrics Aggregator
Combines Statistical Parity, Error-Based Metrics, and Counterfactual Analysis
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import subprocess
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_utils import load_config, setup_logging

def run_all_metric_calculations() -> Dict[str, Dict]:
    """Run all fairness metric calculations in sequence"""
    
    base_dir = Path(__file__).parent.parent
    metrics_dir = Path(__file__).parent
    
    results = {}
    
    # Run Statistical Parity
    print("Running Statistical Parity calculation...")
    sp_result = subprocess.run(
        ["python", str(metrics_dir / "statistical_parity.py")],
        cwd=str(base_dir),
        capture_output=True,
        text=True
    )
    
    if sp_result.returncode == 0:
        with open(base_dir / "outputs" / "metrics" / "statistical_parity_results.json", 'r') as f:
            results['statistical_parity'] = json.load(f)
        print("✅ Statistical Parity completed successfully")
    else:
        print(f"❌ Statistical Parity failed: {sp_result.stderr}")
    
    # Run Error-Based Metrics
    print("Running Error-Based Metrics calculation...")
    error_result = subprocess.run(
        ["python", str(metrics_dir / "error_based_metrics.py")],
        cwd=str(base_dir),
        capture_output=True,
        text=True
    )
    
    if error_result.returncode == 0:
        with open(base_dir / "outputs" / "metrics" / "error_based_metrics_results.json", 'r') as f:
            results['error_based_metrics'] = json.load(f)
        print("✅ Error-Based Metrics completed successfully")
    else:
        print(f"❌ Error-Based Metrics failed: {error_result.stderr}")
    
    # Run Counterfactual Analysis
    print("Running Counterfactual Analysis...")
    cf_result = subprocess.run(
        ["python", str(metrics_dir / "counterfactual_metrics.py")],
        cwd=str(base_dir),
        capture_output=True,
        text=True
    )
    
    if cf_result.returncode == 0:
        with open(base_dir / "outputs" / "metrics" / "counterfactual_metrics_results.json", 'r') as f:
            results['counterfactual_metrics'] = json.load(f)
        print("✅ Counterfactual Analysis completed successfully")
    else:
        print(f"❌ Counterfactual Analysis failed: {cf_result.stderr}")
    
    return results

def aggregate_bias_findings(all_results: Dict[str, Dict]) -> Dict:
    """Aggregate bias findings across all metrics"""
    
    bias_summary = {
        'temporal_bias_evidence': {},
        'intersectional_disparities': {},
        'most_biased_topics': {},
        'protected_attribute_impacts': {},
        'overall_bias_assessment': {}
    }
    
    # Analyze Statistical Parity results
    if 'statistical_parity' in all_results:
        sp_results = all_results['statistical_parity']
        
        # Count significant disparities by attribute
        sp_disparities = {}
        for attr, attr_data in sp_results.get('individual_attributes', {}).items():
            total_comparisons = 0
            significant_disparities = 0
            
            for topic, topic_data in attr_data.items():
                for comparison, result in topic_data.items():
                    total_comparisons += 1
                    if result.get('is_significant', False):
                        significant_disparities += 1
            
            sp_disparities[attr] = {
                'total_comparisons': total_comparisons,
                'significant_disparities': significant_disparities,
                'bias_rate': significant_disparities / total_comparisons if total_comparisons > 0 else 0
            }
        
        bias_summary['protected_attribute_impacts']['statistical_parity'] = sp_disparities
    
    # Analyze Error-Based Metrics results
    if 'error_based_metrics' in all_results:
        error_results = all_results['error_based_metrics']
        
        # Analyze Equal Opportunity and FPR gaps
        error_disparities = {}
        for attr, attr_data in error_results.get('individual_attributes', {}).items():
            high_eo_gaps = 0
            high_fpr_gaps = 0
            total_comparisons = 0
            
            for topic, topic_data in attr_data.items():
                for comparison, result in topic_data.items():
                    total_comparisons += 1
                    
                    eo_gap = abs(result.get('equal_opportunity_gap', 0))
                    fpr_gap = abs(result.get('false_positive_rate_gap', 0))
                    
                    if eo_gap > 0.1:  # 10% threshold
                        high_eo_gaps += 1
                    if fpr_gap > 0.1:
                        high_fpr_gaps += 1
            
            error_disparities[attr] = {
                'total_comparisons': total_comparisons,
                'high_eo_gaps': high_eo_gaps,
                'high_fpr_gaps': high_fpr_gaps,
                'eo_bias_rate': high_eo_gaps / total_comparisons if total_comparisons > 0 else 0,
                'fpr_bias_rate': high_fpr_gaps / total_comparisons if total_comparisons > 0 else 0
            }
        
        bias_summary['protected_attribute_impacts']['error_based'] = error_disparities
    
    # Analyze Counterfactual results
    if 'counterfactual_metrics' in all_results:
        cf_results = all_results['counterfactual_metrics']
        
        cf_analysis = {}
        for attr, attr_data in cf_results.get('individual_attributes', {}).items():
            high_flip_rates = 0
            total_topics = 0
            
            for topic, topic_data in attr_data.items():
                total_topics += 1
                flip_rate = topic_data.get('overall_flip_rate', 0)
                
                if flip_rate > 0.2:  # 20% threshold
                    high_flip_rates += 1
            
            cf_analysis[attr] = {
                'total_topics': total_topics,
                'high_flip_rate_topics': high_flip_rates,
                'flip_bias_rate': high_flip_rates / total_topics if total_topics > 0 else 0
            }
        
        bias_summary['protected_attribute_impacts']['counterfactual'] = cf_analysis
    
    # Identify most problematic topics
    topic_bias_scores = {}
    
    # Aggregate across all metrics
    for metric_type, metric_data in all_results.items():
        if metric_type == 'statistical_parity':
            for attr, attr_data in metric_data.get('individual_attributes', {}).items():
                for topic, topic_data in attr_data.items():
                    if topic not in topic_bias_scores:
                        topic_bias_scores[topic] = {'sp_issues': 0, 'error_issues': 0, 'cf_issues': 0}
                    
                    significant_count = sum(1 for result in topic_data.values() 
                                          if result.get('is_significant', False))
                    topic_bias_scores[topic]['sp_issues'] += significant_count
        
        elif metric_type == 'error_based_metrics':
            for attr, attr_data in metric_data.get('individual_attributes', {}).items():
                for topic, topic_data in attr_data.items():
                    if topic not in topic_bias_scores:
                        topic_bias_scores[topic] = {'sp_issues': 0, 'error_issues': 0, 'cf_issues': 0}
                    
                    high_gap_count = sum(1 for result in topic_data.values()
                                       if abs(result.get('equal_opportunity_gap', 0)) > 0.1 or 
                                          abs(result.get('false_positive_rate_gap', 0)) > 0.1)
                    topic_bias_scores[topic]['error_issues'] += high_gap_count
        
        elif metric_type == 'counterfactual_metrics':
            for attr, attr_data in metric_data.get('individual_attributes', {}).items():
                for topic, topic_data in attr_data.items():
                    if topic not in topic_bias_scores:
                        topic_bias_scores[topic] = {'sp_issues': 0, 'error_issues': 0, 'cf_issues': 0}
                    
                    if topic_data.get('overall_flip_rate', 0) > 0.2:
                        topic_bias_scores[topic]['cf_issues'] += 1
    
    # Rank topics by total bias issues
    topic_rankings = []
    for topic, scores in topic_bias_scores.items():
        total_issues = scores['sp_issues'] + scores['error_issues'] + scores['cf_issues']
        topic_rankings.append({
            'topic': topic,
            'total_bias_issues': total_issues,
            'statistical_parity_issues': scores['sp_issues'],
            'error_based_issues': scores['error_issues'],
            'counterfactual_issues': scores['cf_issues']
        })
    
    topic_rankings.sort(key=lambda x: x['total_bias_issues'], reverse=True)
    bias_summary['most_biased_topics'] = topic_rankings[:10]  # Top 10
    
    # Overall bias assessment
    total_sp_significant = sum(
        result.get('summary', {}).get('total_significant', 0)
        for result in all_results.values()
        if 'summary' in result and 'total_significant' in result['summary']
    )
    
    total_sp_comparisons = sum(
        result.get('summary', {}).get('total_comparisons', 0)
        for result in all_results.values()
        if 'summary' in result and 'total_comparisons' in result['summary']
    )
    
    bias_summary['overall_bias_assessment'] = {
        'overall_significance_rate': total_sp_significant / total_sp_comparisons if total_sp_comparisons > 0 else 0,
        'total_statistical_tests': total_sp_comparisons,
        'significant_disparities': total_sp_significant,
        'analysis_timestamp': datetime.now().isoformat()
    }
    
    return bias_summary

def generate_executive_summary(all_results: Dict[str, Dict], bias_analysis: Dict) -> str:
    """Generate executive summary of fairness analysis"""
    
    summary = []
    summary.append("=" * 80)
    summary.append("COMPARATIVE FAIRNESS ANALYSIS - EXECUTIVE SUMMARY")
    summary.append("=" * 80)
    summary.append("")
    
    # Overall assessment
    overall = bias_analysis.get('overall_bias_assessment', {})
    sig_rate = overall.get('overall_significance_rate', 0) * 100
    
    summary.append(f"OVERALL BIAS ASSESSMENT:")
    summary.append(f"  Significance Rate: {sig_rate:.1f}% of statistical tests show significant bias")
    summary.append(f"  Total Statistical Tests: {overall.get('total_statistical_tests', 0)}")
    summary.append(f"  Significant Disparities Found: {overall.get('significant_disparities', 0)}")
    summary.append("")
    
    # Protected attribute impacts
    summary.append("PROTECTED ATTRIBUTE BIAS ANALYSIS:")
    
    sp_impacts = bias_analysis.get('protected_attribute_impacts', {}).get('statistical_parity', {})
    for attr, impact in sp_impacts.items():
        bias_rate = impact.get('bias_rate', 0) * 100
        summary.append(f"  {attr.upper()}: {bias_rate:.1f}% bias rate ({impact.get('significant_disparities', 0)}/{impact.get('total_comparisons', 0)} tests)")
    
    summary.append("")
    
    # Most biased topics
    summary.append("MOST PROBLEMATIC TOPICS:")
    top_topics = bias_analysis.get('most_biased_topics', [])[:5]
    for i, topic_info in enumerate(top_topics, 1):
        summary.append(f"  {i}. {topic_info['topic']}: {topic_info['total_bias_issues']} bias issues")
        summary.append(f"     SP: {topic_info['statistical_parity_issues']}, Error: {topic_info['error_based_issues']}, CF: {topic_info['counterfactual_issues']}")
    
    summary.append("")
    
    # Temporal bias evidence
    if 'statistical_parity' in all_results:
        sp_summary = all_results['statistical_parity'].get('summary', {})
        summary.append(f"TEMPORAL MODEL COMPARISON:")
        summary.append(f"  Statistical Parity Tests: {sp_summary.get('total_comparisons', 0)}")
        summary.append(f"  Significant Differences: {sp_summary.get('total_significant', 0)}")
        summary.append(f"  Evidence of Temporal Bias: {'STRONG' if sig_rate > 50 else 'MODERATE' if sig_rate > 25 else 'WEAK'}")
    
    summary.append("")
    
    # Counterfactual evidence
    if 'counterfactual_metrics' in all_results:
        cf_summary = all_results['counterfactual_metrics'].get('summary', {})
        summary.append(f"COUNTERFACTUAL ANALYSIS:")
        summary.append(f"  Counterfactual Pairs Found: {cf_summary.get('total_counterfactual_pairs', 0)}")
        summary.append(f"  Individual Comparisons: {cf_summary.get('individual_comparisons', 0)}")
        summary.append(f"  Intersectional Comparisons: {cf_summary.get('intersectional_comparisons', 0)}")
    
    summary.append("")
    summary.append("RECOMMENDATIONS:")
    
    if sig_rate > 50:
        summary.append("  🚨 HIGH BIAS DETECTED - Immediate intervention required")
        summary.append("  - Audit model training data for temporal bias")
        summary.append("  - Implement bias mitigation techniques")
        summary.append("  - Consider model retraining with balanced data")
    elif sig_rate > 25:
        summary.append("  ⚠️  MODERATE BIAS DETECTED - Monitoring and mitigation advised")
        summary.append("  - Implement ongoing bias monitoring")
        summary.append("  - Review decision patterns for most biased topics")
        summary.append("  - Consider post-processing fairness corrections")
    else:
        summary.append("  ✅ LOW BIAS DETECTED - Continue monitoring")
        summary.append("  - Maintain regular bias assessments")
        summary.append("  - Monitor for emerging bias patterns")
    
    summary.append("")
    summary.append("=" * 80)
    
    return "\n".join(summary)

def main():
    """Main function for comprehensive fairness metrics aggregation"""
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "outputs" / "metrics"
    reports_dir = base_dir / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = base_dir / "outputs" / "logs" / "metrics_aggregator.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(log_file))
    
    logger.info("Starting comprehensive fairness metrics aggregation")
    
    # Run all metric calculations
    print("🚀 Running Comprehensive Fairness Analysis...")
    print("This may take several minutes due to computational complexity.")
    print()
    
    all_results = run_all_metric_calculations()
    
    logger.info(f"Completed {len(all_results)} metric calculations")
    
    # Aggregate bias findings
    logger.info("Aggregating bias findings across all metrics...")
    bias_analysis = aggregate_bias_findings(all_results)
    
    # Generate executive summary
    logger.info("Generating executive summary...")
    executive_summary = generate_executive_summary(all_results, bias_analysis)
    
    # Save comprehensive results
    comprehensive_results = {
        'analysis_metadata': {
            'timestamp': datetime.now().isoformat(),
            'metrics_calculated': list(all_results.keys()),
            'total_groups_analyzed': sum(
                result.get('summary', {}).get('total_comparisons', 0)
                for result in all_results.values()
            )
        },
        'individual_metrics': all_results,
        'bias_analysis': bias_analysis,
        'executive_summary_text': executive_summary
    }
    
    # Save to JSON
    comprehensive_file = output_dir / "comprehensive_fairness_analysis.json"
    with open(comprehensive_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    # Save executive summary as text
    summary_file = reports_dir / "executive_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(executive_summary)
    
    # Print executive summary
    print()
    print(executive_summary)
    
    logger.info(f"Saved comprehensive analysis to {comprehensive_file}")
    logger.info(f"Saved executive summary to {summary_file}")
    logger.info("Comprehensive fairness analysis completed successfully!")
    
    return comprehensive_results

if __name__ == "__main__":
    results = main() 