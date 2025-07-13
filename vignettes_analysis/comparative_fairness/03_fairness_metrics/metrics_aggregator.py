"""
Comprehensive Fairness Metrics Aggregator
Combines Statistical Parity, Counterfactual Analysis, and Temporal Comparison
WITHOUT statistical significance testing (moved to 04_significance_testing)
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
    
    # Check if Statistical Parity results exist
    sp_file = base_dir / "outputs" / "metrics" / "statistical_parity_results.json"
    if sp_file.exists():
        print("✅ Using existing Statistical Parity results...")
        with open(sp_file, 'r') as f:
            results['statistical_parity'] = json.load(f)
    else:
        print("Running Statistical Parity calculation...")
        sp_result = subprocess.run(
            ["python", str(metrics_dir / "statistical_parity.py")],
            cwd=str(base_dir),
            capture_output=True,
            text=True
        )
        
        if sp_result.returncode == 0:
            with open(sp_file, 'r') as f:
                results['statistical_parity'] = json.load(f)
            print("✅ Statistical Parity completed successfully")
        else:
            print(f"❌ Statistical Parity failed: {sp_result.stderr}")
    
    # Check if Counterfactual Analysis results exist
    cf_file = base_dir / "outputs" / "metrics" / "counterfactual_metrics_results.json"
    if cf_file.exists():
        print("✅ Using existing Counterfactual Analysis results...")
        with open(cf_file, 'r') as f:
            results['counterfactual_metrics'] = json.load(f)
    else:
        print("Running Counterfactual Analysis...")
        cf_result = subprocess.run(
            ["python", str(metrics_dir / "counterfactual_metrics.py")],
            cwd=str(base_dir),
            capture_output=True,
            text=True
        )
        
        if cf_result.returncode == 0:
            with open(cf_file, 'r') as f:
                results['counterfactual_metrics'] = json.load(f)
            print("✅ Counterfactual Analysis completed successfully")
        else:
            print(f"❌ Counterfactual Analysis failed: {cf_result.stderr}")
    
    # Check if Error-Based Metrics results exist
    eb_file = base_dir / "outputs" / "metrics" / "error_based_metrics_results.json"
    if eb_file.exists():
        print("✅ Using existing Error-Based Metrics results...")
        with open(eb_file, 'r') as f:
            results['error_based_metrics'] = json.load(f)
    else:
        print("Running Error-Based Metrics calculation...")
        eb_result = subprocess.run(
            ["python", str(metrics_dir / "error_based_metrics.py")],
            cwd=str(base_dir),
            capture_output=True,
            text=True
        )
        
        if eb_result.returncode == 0:
            with open(eb_file, 'r') as f:
                results['error_based_metrics'] = json.load(f)
            print("✅ Error-Based Metrics completed successfully")
        else:
            print(f"❌ Error-Based Metrics failed: {eb_result.stderr}")
    
    # Check if Temporal Model Comparison results exist
    temp_file = base_dir / "outputs" / "metrics" / "temporal_comparison_results.json"
    if temp_file.exists():
        print("✅ Using existing Temporal Model Comparison results...")
        with open(temp_file, 'r') as f:
            results['temporal_comparison'] = json.load(f)
    else:
        print("Running Temporal Model Comparison...")
        temp_result = subprocess.run(
            ["python", str(metrics_dir / "temporal_model_comparison.py")],
            cwd=str(base_dir),
            capture_output=True,
            text=True
        )
        
        if temp_result.returncode == 0:
            with open(temp_file, 'r') as f:
                results['temporal_comparison'] = json.load(f)
            print("✅ Temporal Model Comparison completed successfully")
        else:
            print(f"❌ Temporal Model Comparison failed: {temp_result.stderr}")
    
    return results

def calculate_relative_metrics(all_results: Dict[str, Dict]) -> Dict:
    """Calculate relative metrics between pre and post Brexit models"""
    
    relative_metrics = {
        'statistical_parity_relative': {},
        'counterfactual_relative': {},
        'error_based_relative': {},
        'summary': {}
    }
    
    # Calculate Statistical Parity relative metrics
    if 'statistical_parity' in all_results:
        sp_results = all_results['statistical_parity']
        sp_relative = {}
        
        for attr, attr_data in sp_results.get('individual_attributes', {}).items():
            attr_relative = {}
            
            for topic, topic_data in attr_data.items():
                topic_relative = {}
                
                # Group comparisons by base comparison (ignoring model suffix)
                comparisons = {}
                for key, result in topic_data.items():
                    if key.endswith('_post_brexit') or key.endswith('_pre_brexit'):
                        base_key = key.rsplit('_', 2)[0]  # Remove model suffix
                        model = 'post_brexit' if key.endswith('_post_brexit') else 'pre_brexit'
                        
                        if base_key not in comparisons:
                            comparisons[base_key] = {}
                        comparisons[base_key][model] = result
                
                # Calculate relative metrics for each comparison
                for base_key, models in comparisons.items():
                    if 'post_brexit' in models and 'pre_brexit' in models:
                        post_result = models['post_brexit']
                        pre_result = models['pre_brexit']
                        
                        # Calculate relative metrics
                        sp_gap_change = post_result['sp_gap'] - pre_result['sp_gap']
                        grant_rate_change = post_result['group_grant_rate'] - pre_result['group_grant_rate']
                        
                        topic_relative[f"{base_key}_relative"] = {
                            'attribute': post_result['attribute'],
                            'group': post_result['group'],
                            'reference': post_result['reference'],
                            'topic': topic,
                            'post_brexit_sp_gap': post_result['sp_gap'],
                            'pre_brexit_sp_gap': pre_result['sp_gap'],
                            'sp_gap_change': sp_gap_change,
                            'post_brexit_grant_rate': post_result['group_grant_rate'],
                            'pre_brexit_grant_rate': pre_result['group_grant_rate'],
                            'grant_rate_change': grant_rate_change,
                            'post_brexit_sample_size': post_result['group_size'],
                            'pre_brexit_sample_size': pre_result['group_size'],
                            'bias_direction': 'worsening' if sp_gap_change < 0 else 'improving' if sp_gap_change > 0 else 'stable'
                        }
                
                if topic_relative:
                    attr_relative[topic] = topic_relative
            
            if attr_relative:
                sp_relative[attr] = attr_relative
        
        relative_metrics['statistical_parity_relative'] = sp_relative
    
    # Calculate Counterfactual relative metrics (use built-in relative from new structure)
    if 'counterfactual_metrics' in all_results:
        cf_results = all_results['counterfactual_metrics']
        
        # Use the relative metrics that were already calculated
        if 'counterfactual_relative' in cf_results:
            relative_metrics['counterfactual_relative'] = cf_results['counterfactual_relative']
        else:
            # Fallback for backward compatibility
            cf_relative = {}
            for attr, attr_data in cf_results.get('individual_attributes', {}).items():
                attr_relative = {}
                
                for topic, topic_data in attr_data.items():
                    attr_relative[topic] = {
                        'attribute': attr,
                        'topic': topic,
                        'overall_flip_rate': topic_data.get('overall_flip_rate', 0),
                        'grant_to_deny_rate': topic_data.get('grant_to_deny_rate', 0),
                        'deny_to_grant_rate': topic_data.get('deny_to_grant_rate', 0),
                        'total_pairs': topic_data.get('total_pairs', 0),
                        'counterfactual_instability': 'high' if topic_data.get('overall_flip_rate', 0) > 0.3 else 'moderate' if topic_data.get('overall_flip_rate', 0) > 0.15 else 'low'
                    }
                
                if attr_relative:
                    cf_relative[attr] = attr_relative
            
            relative_metrics['counterfactual_relative'] = cf_relative
    
    # Calculate Error-Based relative metrics
    if 'error_based_metrics' in all_results:
        eb_results = all_results['error_based_metrics']
        eb_relative = {}
        
        for attr, attr_data in eb_results.get('individual_attributes', {}).items():
            attr_relative = {}
            
            for topic, topic_data in attr_data.items():
                topic_relative = {}
                
                # Group comparisons by base comparison (ignoring model suffix)
                comparisons = {}
                for key, result in topic_data.items():
                    if key.endswith('_post_brexit') or key.endswith('_pre_brexit'):
                        base_key = key.rsplit('_', 2)[0]  # Remove model suffix
                        model = 'post_brexit' if key.endswith('_post_brexit') else 'pre_brexit'
                        
                        if base_key not in comparisons:
                            comparisons[base_key] = {}
                        comparisons[base_key][model] = result
                
                # Calculate relative metrics for each comparison
                for base_key, models in comparisons.items():
                    if 'post_brexit' in models and 'pre_brexit' in models:
                        post_result = models['post_brexit']
                        pre_result = models['pre_brexit']
                        
                        # Calculate relative metrics
                        tpr_change = post_result['group_tpr'] - pre_result['group_tpr']
                        fpr_change = post_result['group_fpr'] - pre_result['group_fpr']
                        agreement_change = post_result['agreement_rate'] - pre_result['agreement_rate']
                        
                        topic_relative[f"{base_key}_relative"] = {
                            'attribute': post_result['attribute'],
                            'group': post_result['group'],
                            'reference': post_result['reference'],
                            'topic': topic,
                            'post_brexit_tpr': post_result['group_tpr'],
                            'pre_brexit_tpr': pre_result['group_tpr'],
                            'tpr_change': tpr_change,
                            'post_brexit_fpr': post_result['group_fpr'],
                            'pre_brexit_fpr': pre_result['group_fpr'],
                            'fpr_change': fpr_change,
                            'post_brexit_agreement': post_result['agreement_rate'],
                            'pre_brexit_agreement': pre_result['agreement_rate'],
                            'agreement_change': agreement_change,
                            'post_brexit_sample_size': post_result['total_sample_size'],
                            'pre_brexit_sample_size': pre_result['total_sample_size'],
                            'error_direction': 'worsening' if abs(tpr_change) + abs(fpr_change) > 0.05 else 'improving' if abs(tpr_change) + abs(fpr_change) < -0.05 else 'stable'
                        }
                
                if topic_relative:
                    attr_relative[topic] = topic_relative
            
            if attr_relative:
                eb_relative[attr] = attr_relative
        
        relative_metrics['error_based_relative'] = eb_relative
    
    # Add temporal comparison results if available
    if 'temporal_comparison' in all_results:
        relative_metrics['temporal_comparison'] = all_results['temporal_comparison']
    
    return relative_metrics

def structure_final_output(all_results: Dict[str, Dict], relative_metrics: Dict) -> Dict:
    """Structure the final output according to user requirements"""
    
    structured_output = {
        'individual_model_metrics': {
            'statistical_parity': {
                'pre_brexit': {},
                'post_brexit': {}
            },
            'counterfactual': {
                'pre_brexit': {},
                'post_brexit': {}
            },
            'error_based': {
                'pre_brexit': {},
                'post_brexit': {}
            }
        },
        'relative_metrics': relative_metrics,
        'metadata': {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_topics_analyzed': 0,
            'total_comparisons': 0,
            'metrics_included': []
        }
    }
    
    # Extract individual model metrics from statistical parity
    if 'statistical_parity' in all_results:
        sp_results = all_results['statistical_parity']
        structured_output['metadata']['metrics_included'].append('statistical_parity')
        
        for attr, attr_data in sp_results.get('individual_attributes', {}).items():
            for topic, topic_data in attr_data.items():
                for key, result in topic_data.items():
                    if key.endswith('_post_brexit'):
                        base_key = key.rsplit('_', 2)[0]
                        if attr not in structured_output['individual_model_metrics']['statistical_parity']['post_brexit']:
                            structured_output['individual_model_metrics']['statistical_parity']['post_brexit'][attr] = {}
                        if topic not in structured_output['individual_model_metrics']['statistical_parity']['post_brexit'][attr]:
                            structured_output['individual_model_metrics']['statistical_parity']['post_brexit'][attr][topic] = {}
                        structured_output['individual_model_metrics']['statistical_parity']['post_brexit'][attr][topic][base_key] = result
                    
                    elif key.endswith('_pre_brexit'):
                        base_key = key.rsplit('_', 2)[0]
                        if attr not in structured_output['individual_model_metrics']['statistical_parity']['pre_brexit']:
                            structured_output['individual_model_metrics']['statistical_parity']['pre_brexit'][attr] = {}
                        if topic not in structured_output['individual_model_metrics']['statistical_parity']['pre_brexit'][attr]:
                            structured_output['individual_model_metrics']['statistical_parity']['pre_brexit'][attr][topic] = {}
                        structured_output['individual_model_metrics']['statistical_parity']['pre_brexit'][attr][topic][base_key] = result
    
    # Extract individual model metrics from counterfactual analysis
    if 'counterfactual_metrics' in all_results:
        cf_results = all_results['counterfactual_metrics']
        structured_output['metadata']['metrics_included'].append('counterfactual')
        
        # Handle new structure with separated models - with error handling
        if 'pre_brexit_model' in cf_results and cf_results['pre_brexit_model']:
            pre_brexit_data = cf_results['pre_brexit_model'].get('individual_attributes', {})
            if pre_brexit_data:
                structured_output['individual_model_metrics']['counterfactual']['pre_brexit'] = pre_brexit_data
        
        if 'post_brexit_model' in cf_results and cf_results['post_brexit_model']:
            post_brexit_data = cf_results['post_brexit_model'].get('individual_attributes', {})
            if post_brexit_data:
                structured_output['individual_model_metrics']['counterfactual']['post_brexit'] = post_brexit_data
        
        # Fallback to individual_attributes if model separation didn't work
        if not structured_output['individual_model_metrics']['counterfactual']['pre_brexit'] and \
           not structured_output['individual_model_metrics']['counterfactual']['post_brexit']:
            if 'individual_attributes' in cf_results and cf_results['individual_attributes']:
                # Use the existing structure as fallback
                structured_output['individual_model_metrics']['counterfactual']['combined'] = cf_results['individual_attributes']
    
    # Extract individual model metrics from error-based analysis
    if 'error_based_metrics' in all_results:
        eb_results = all_results['error_based_metrics']
        structured_output['metadata']['metrics_included'].append('error_based')
        
        for attr, attr_data in eb_results.get('individual_attributes', {}).items():
            for topic, topic_data in attr_data.items():
                for key, result in topic_data.items():
                    if key.endswith('_post_brexit'):
                        base_key = key.rsplit('_', 2)[0]
                        if attr not in structured_output['individual_model_metrics']['error_based']['post_brexit']:
                            structured_output['individual_model_metrics']['error_based']['post_brexit'][attr] = {}
                        if topic not in structured_output['individual_model_metrics']['error_based']['post_brexit'][attr]:
                            structured_output['individual_model_metrics']['error_based']['post_brexit'][attr][topic] = {}
                        structured_output['individual_model_metrics']['error_based']['post_brexit'][attr][topic][base_key] = result
                    
                    elif key.endswith('_pre_brexit'):
                        base_key = key.rsplit('_', 2)[0]
                        if attr not in structured_output['individual_model_metrics']['error_based']['pre_brexit']:
                            structured_output['individual_model_metrics']['error_based']['pre_brexit'][attr] = {}
                        if topic not in structured_output['individual_model_metrics']['error_based']['pre_brexit'][attr]:
                            structured_output['individual_model_metrics']['error_based']['pre_brexit'][attr][topic] = {}
                        structured_output['individual_model_metrics']['error_based']['pre_brexit'][attr][topic][base_key] = result
    
    # Calculate metadata
    topics_analyzed = set()
    total_comparisons = 0
    
    for metric_type in structured_output['individual_model_metrics']:
        for model in structured_output['individual_model_metrics'][metric_type]:
            for attr in structured_output['individual_model_metrics'][metric_type][model]:
                topics_analyzed.update(structured_output['individual_model_metrics'][metric_type][model][attr].keys())
                for topic in structured_output['individual_model_metrics'][metric_type][model][attr]:
                    total_comparisons += len(structured_output['individual_model_metrics'][metric_type][model][attr][topic])
    
    structured_output['metadata']['total_topics_analyzed'] = len(topics_analyzed)
    structured_output['metadata']['total_comparisons'] = total_comparisons
    
    return structured_output

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
    
    # Calculate relative metrics
    logger.info("Calculating relative metrics between models...")
    relative_metrics = calculate_relative_metrics(all_results)
    
    # Structure final output
    logger.info("Structuring final output...")
    structured_output = structure_final_output(all_results, relative_metrics)
    
    # Save structured results
    comprehensive_file = output_dir / "comprehensive_fairness_analysis.json"
    with open(comprehensive_file, 'w') as f:
        json.dump(structured_output, f, indent=2)
    
    # Generate summary report
    summary_lines = [
        "=" * 80,
        "COMPREHENSIVE FAIRNESS METRICS ANALYSIS",
        "=" * 80,
        "",
        f"Analysis completed at: {datetime.now().isoformat()}",
        "",
        "METRICS CALCULATED:",
        "✅ Statistical Parity (Pre/Post Brexit models)",
        "✅ Counterfactual Metrics (Decision flip rates)",
        "✅ Relative Metrics (Model comparison)",
        "",
        "OUTPUT STRUCTURE:",
        "1. Individual Model Metrics:",
        "   - Statistical Parity (Pre-Brexit)",
        "   - Statistical Parity (Post-Brexit)", 
        "   - Counterfactual Metrics",
        "2. Relative Metrics:",
        "   - Statistical Parity Changes",
        "   - Counterfactual Changes",
        "   - Temporal Comparison",
        "",
        "Next Steps:",
        "- Run significance testing in 04_significance_testing",
        "- Review individual model performance",
        "- Analyze relative metric changes",
        "",
        "=" * 80
    ]
    
    summary_text = "\n".join(summary_lines)
    
    # Save summary
    summary_file = reports_dir / "metrics_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(summary_text)
    
    # Print summary
    print(summary_text)
    
    logger.info(f"Saved comprehensive analysis to {comprehensive_file}")
    logger.info(f"Saved summary to {summary_file}")
    logger.info("Comprehensive fairness analysis completed successfully!")
    
    return structured_output

if __name__ == "__main__":
    results = main() 