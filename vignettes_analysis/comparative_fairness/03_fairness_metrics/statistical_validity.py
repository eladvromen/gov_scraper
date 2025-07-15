#!/usr/bin/env python3
"""
Statistical Validity Analysis
=============================

Applies robust statistical methods with multiple testing corrections 
to fairness significance vectors and compares with original results.

Key Features:
- Bonferroni correction for conservative approach
- False Discovery Rate (FDR) for balanced approach  
- Comprehensive comparison statistics
- Robust significance vectors output
- Statistical validity reporting

Author: Enhanced Analysis Pipeline
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Any
from scipy.stats import false_discovery_control
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StatisticalValidityAnalyzer:
    """
    Analyzes statistical validity of fairness significance vectors
    """
    
    def __init__(self, unified_df_path: str):
        """
        Initialize the analyzer
        
        Args:
            unified_df_path: Path to unified fairness dataframe
        """
        self.unified_df_path = unified_df_path
        self.df = None
        self.results = {}
        
    def load_data(self):
        """Load the unified fairness dataframe"""
        logger.info(f"Loading data from {self.unified_df_path}")
        self.df = pd.read_csv(self.unified_df_path)
        logger.info(f"Loaded {len(self.df)} fairness comparisons")
        
        # Validate required columns
        required_cols = [
            'pre_brexit_sp_p_value', 'post_brexit_sp_p_value',
            'pre_brexit_sp_significance', 'post_brexit_sp_significance'
        ]
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
    def apply_multiple_testing_corrections(self) -> Dict[str, Any]:
        """
        Apply multiple testing corrections to p-values
        
        Returns:
            Dictionary with corrected significance results
        """
        logger.info("Applying multiple testing corrections...")
        
        results = {}
        n_comparisons = len(self.df)
        
        for model in ['pre_brexit', 'post_brexit']:
            logger.info(f"Processing {model} model...")
            
            # Extract p-values and original significance
            p_col = f'{model}_sp_p_value'
            sig_col = f'{model}_sp_significance'
            
            p_values = self.df[p_col].fillna(1.0).values
            original_significance = self.df[sig_col].fillna(False).values
            
            # Apply corrections
            corrections = self._apply_corrections(p_values, n_comparisons)
            
            # Store results
            model_results = {
                'original_p_values': p_values,
                'original_significance': original_significance,
                'n_comparisons': n_comparisons,
                **corrections
            }
            
            results[model] = model_results
            
            # Log summary statistics
            self._log_correction_summary(model, model_results)
            
        return results
    
    def _apply_corrections(self, p_values: np.ndarray, n_comparisons: int) -> Dict[str, Any]:
        """
        Apply various multiple testing corrections
        
        Args:
            p_values: Array of p-values
            n_comparisons: Total number of comparisons
            
        Returns:
            Dictionary with correction results
        """
        
        # Remove NaN values for correction calculations
        valid_mask = ~np.isnan(p_values)
        valid_p_values = p_values[valid_mask]
        
        corrections = {}
        
        # 1. Bonferroni Correction (Conservative)
        bonferroni_alpha = 0.05 / n_comparisons
        bonferroni_significance = np.zeros_like(p_values, dtype=bool)
        bonferroni_significance[valid_mask] = valid_p_values < bonferroni_alpha
        
        corrections['bonferroni_alpha'] = bonferroni_alpha
        corrections['bonferroni_significance'] = bonferroni_significance
        corrections['bonferroni_count'] = np.sum(bonferroni_significance)
        
        # 2. False Discovery Rate (FDR) - Benjamini-Hochberg (Balanced)
        if len(valid_p_values) > 0:
            fdr_adjusted_p_values = false_discovery_control(valid_p_values, method='bh')
            fdr_significance_valid = fdr_adjusted_p_values < 0.05  # Compare adjusted p-values to alpha
            fdr_significance = np.zeros_like(p_values, dtype=bool)
            fdr_significance[valid_mask] = fdr_significance_valid
        else:
            fdr_significance = np.zeros_like(p_values, dtype=bool)
            
        corrections['fdr_significance'] = fdr_significance
        corrections['fdr_count'] = np.sum(fdr_significance)
        
        # 3. Holm-Bonferroni (Step-down procedure)
        holm_significance = self._apply_holm_bonferroni(valid_p_values, n_comparisons)
        holm_full = np.zeros_like(p_values, dtype=bool)
        holm_full[valid_mask] = holm_significance
        
        corrections['holm_significance'] = holm_full
        corrections['holm_count'] = np.sum(holm_full)
        
        return corrections
    
    def _apply_holm_bonferroni(self, p_values: np.ndarray, n_comparisons: int) -> np.ndarray:
        """
        Apply Holm-Bonferroni step-down correction
        
        Args:
            p_values: Array of valid p-values
            n_comparisons: Total number of comparisons
            
        Returns:
            Boolean array of significant results
        """
        if len(p_values) == 0:
            return np.array([], dtype=bool)
            
        # Sort p-values and track original indices
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        # Apply Holm correction
        significance = np.zeros(len(p_values), dtype=bool)
        
        for i, p_val in enumerate(sorted_p_values):
            # Holm correction: alpha / (n - i)
            corrected_alpha = 0.05 / (n_comparisons - i)
            
            if p_val <= corrected_alpha:
                significance[sorted_indices[i]] = True
            else:
                # Once we fail to reject, stop (step-down property)
                break
                
        return significance
    
    def _log_correction_summary(self, model: str, results: Dict[str, Any]):
        """Log summary of correction results"""
        original_count = np.sum(results['original_significance'])
        total = len(results['original_significance'])
        
        logger.info(f"\n=== {model.upper()} STATISTICAL CORRECTIONS ===")
        logger.info(f"Total comparisons: {total:,}")
        logger.info(f"Original significant (α=0.05): {original_count:,} ({original_count/total:.2%})")
        logger.info(f"Bonferroni significant (α={results['bonferroni_alpha']:.2e}): {results['bonferroni_count']:,} ({results['bonferroni_count']/total:.2%})")
        logger.info(f"FDR significant: {results['fdr_count']:,} ({results['fdr_count']/total:.2%})")
        logger.info(f"Holm-Bonferroni significant: {results['holm_count']:,} ({results['holm_count']/total:.2%})")
    
    def compare_correction_methods(self, correction_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare different correction methods
        
        Args:
            correction_results: Results from apply_multiple_testing_corrections
            
        Returns:
            Comparison statistics
        """
        logger.info("Comparing correction methods...")
        
        comparison = {}
        
        for model in ['pre_brexit', 'post_brexit']:
            model_data = correction_results[model]
            
            original = model_data['original_significance']
            bonferroni = model_data['bonferroni_significance']
            fdr = model_data['fdr_significance']
            holm = model_data['holm_significance']
            
            # Calculate reduction rates
            model_comparison = {
                'original_count': np.sum(original),
                'bonferroni_count': np.sum(bonferroni),
                'fdr_count': np.sum(fdr),
                'holm_count': np.sum(holm),
                'total_comparisons': len(original),
                
                # Reduction rates
                'bonferroni_reduction': (np.sum(original) - np.sum(bonferroni)) / max(np.sum(original), 1),
                'fdr_reduction': (np.sum(original) - np.sum(fdr)) / max(np.sum(original), 1),
                'holm_reduction': (np.sum(original) - np.sum(holm)) / max(np.sum(original), 1),
                
                # Agreement between methods
                'bonferroni_fdr_agreement': np.mean(bonferroni == fdr),
                'bonferroni_holm_agreement': np.mean(bonferroni == holm),
                'fdr_holm_agreement': np.mean(fdr == holm),
                
                # Conservative vs liberal comparison
                'only_original': np.sum(original & ~fdr),  # Only significant in original
                'survived_fdr': np.sum(original & fdr),   # Survived FDR correction
                'survived_bonferroni': np.sum(original & bonferroni),  # Survived Bonferroni
            }
            
            comparison[model] = model_comparison
            
        # Cross-model comparison
        pre_data = correction_results['pre_brexit']
        post_data = correction_results['post_brexit']
        
        comparison['cross_model'] = {
            'original_both_significant': np.sum(pre_data['original_significance'] & post_data['original_significance']),
            'fdr_both_significant': np.sum(pre_data['fdr_significance'] & post_data['fdr_significance']),
            'bonferroni_both_significant': np.sum(pre_data['bonferroni_significance'] & post_data['bonferroni_significance']),
            
            'original_consistency': np.mean(pre_data['original_significance'] == post_data['original_significance']),
            'fdr_consistency': np.mean(pre_data['fdr_significance'] == post_data['fdr_significance']),
            'bonferroni_consistency': np.mean(pre_data['bonferroni_significance'] == post_data['bonferroni_significance']),
        }
        
        return comparison
    
    def create_robust_vectors(self, correction_results: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Create robust significance vectors using corrected results
        
        Args:
            correction_results: Results from corrections
            
        Returns:
            Dictionary of robust significance vectors
        """
        logger.info("Creating robust significance vectors...")
        
        vectors = {}
        
        # Extract corrected significance vectors
        for model in ['pre_brexit', 'post_brexit']:
            model_data = correction_results[model]
            
            # Add all correction methods
            vectors[f'{model}_sp_significance_original'] = model_data['original_significance']
            vectors[f'{model}_sp_significance_bonferroni'] = model_data['bonferroni_significance']
            vectors[f'{model}_sp_significance_fdr'] = model_data['fdr_significance']
            vectors[f'{model}_sp_significance_holm'] = model_data['holm_significance']
        
        # Create cross-model comparison vectors for FDR (recommended method)
        pre_fdr = correction_results['pre_brexit']['fdr_significance']
        post_fdr = correction_results['post_brexit']['fdr_significance']
        
        vectors['sp_significance_change_fdr'] = pre_fdr != post_fdr
        vectors['sp_both_significant_fdr'] = pre_fdr & post_fdr
        vectors['sp_neither_significant_fdr'] = ~pre_fdr & ~post_fdr
        vectors['sp_gained_significance_fdr'] = ~pre_fdr & post_fdr
        vectors['sp_lost_significance_fdr'] = pre_fdr & ~post_fdr
        
        # Summary vectors
        vectors['sp_robust_either_significant'] = pre_fdr | post_fdr
        vectors['sp_robust_consistently_significant'] = pre_fdr & post_fdr
        
        logger.info(f"Created {len(vectors)} robust significance vectors")
        
        return vectors
    
    def generate_validity_report(self, correction_results: Dict[str, Any], 
                               comparison_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive validity report
        
        Args:
            correction_results: Correction results
            comparison_stats: Comparison statistics
            
        Returns:
            Comprehensive validity report
        """
        logger.info("Generating statistical validity report...")
        
        report = {
            'metadata': {
                'analysis_date': pd.Timestamp.now().isoformat(),
                'total_comparisons': len(self.df),
                'correction_methods': ['bonferroni', 'fdr_bh', 'holm_bonferroni'],
                'alpha_level': 0.05
            },
            'original_analysis': {},
            'corrected_analysis': {},
            'validity_assessment': {},
            'recommendations': {}
        }
        
        # Original analysis summary
        for model in ['pre_brexit', 'post_brexit']:
            original_sig = correction_results[model]['original_significance']
            report['original_analysis'][model] = {
                'significant_count': int(np.sum(original_sig)),
                'significant_rate': float(np.mean(original_sig)),
                'total_comparisons': len(original_sig)
            }
        
        # Corrected analysis summary
        for model in ['pre_brexit', 'post_brexit']:
            model_stats = comparison_stats[model]
            report['corrected_analysis'][model] = {
                'bonferroni': {
                    'significant_count': int(model_stats['bonferroni_count']),
                    'reduction_rate': float(model_stats['bonferroni_reduction']),
                    'survival_rate': float(1 - model_stats['bonferroni_reduction'])
                },
                'fdr': {
                    'significant_count': int(model_stats['fdr_count']),
                    'reduction_rate': float(model_stats['fdr_reduction']),
                    'survival_rate': float(1 - model_stats['fdr_reduction'])
                },
                'holm': {
                    'significant_count': int(model_stats['holm_count']),
                    'reduction_rate': float(model_stats['holm_reduction']),
                    'survival_rate': float(1 - model_stats['holm_reduction'])
                }
            }
        
        # Validity assessment
        total_comparisons = len(self.df)
        expected_false_positives = total_comparisons * 0.05
        
        report['validity_assessment'] = {
            'multiple_testing_severity': 'HIGH' if total_comparisons > 1000 else 'MODERATE' if total_comparisons > 100 else 'LOW',
            'expected_false_positives_original': float(expected_false_positives),
            'false_positive_risk': 'Substantial risk of false discoveries without correction',
            'correction_necessity': 'REQUIRED' if total_comparisons > 100 else 'RECOMMENDED',
            'cross_model_consistency': {
                'original': float(comparison_stats['cross_model']['original_consistency']),
                'fdr_corrected': float(comparison_stats['cross_model']['fdr_consistency']),
                'bonferroni_corrected': float(comparison_stats['cross_model']['bonferroni_consistency'])
            }
        }
        
        # Recommendations
        report['recommendations'] = {
            'primary_method': 'fdr',
            'rationale': 'False Discovery Rate provides optimal balance between Type I and Type II error control',
            'use_cases': {
                'exploratory_analysis': 'Use FDR-corrected vectors for balanced discovery',
                'confirmatory_analysis': 'Use Bonferroni-corrected vectors for conservative claims',
                'reporting': 'Report both original and FDR-corrected results with clear methodology'
            },
            'interpretation_guidelines': [
                'FDR-corrected results represent more reliable significant differences',
                'Original results may contain ~{:.0f} false positives'.format(expected_false_positives),
                'Magnitude vectors remain valid regardless of significance correction',
                'Use corrected significance for hypothesis testing, original for pattern exploration'
            ]
        }
        
        return report
    
    def save_results(self, correction_results: Dict[str, Any], robust_vectors: Dict[str, np.ndarray],
                    comparison_stats: Dict[str, Any], validity_report: Dict[str, Any],
                    output_dir: str):
        """
        Save all results to files
        
        Args:
            correction_results: Statistical correction results
            robust_vectors: Robust significance vectors
            comparison_stats: Method comparison statistics
            validity_report: Comprehensive validity report
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving results to {output_path}")
        
        # 1. Save robust vectors (main output)
        vectors_output = {
            'metadata': {
                'creation_date': pd.Timestamp.now().isoformat(),
                'source_analysis': 'statistical_validity.py',
                'correction_methods': ['bonferroni', 'fdr_bh', 'holm_bonferroni'],
                'total_dimensions': len(next(iter(robust_vectors.values()))),
                'recommendation': 'Use FDR-corrected vectors for balanced analysis'
            },
            'vectors': {k: v.tolist() for k, v in robust_vectors.items()}
        }
        
        with open(output_path / 'robust_significance_vectors.json', 'w') as f:
            json.dump(vectors_output, f, indent=2)
        
        # Helper function to convert numpy types for JSON serialization
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types for JSON serialization"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # 2. Save detailed correction results
        correction_output = {}
        for model, data in correction_results.items():
            correction_output[model] = convert_numpy_types(data)
        
        with open(output_path / 'correction_results_detailed.json', 'w') as f:
            json.dump(correction_output, f, indent=2)
        
        # 3. Save comparison statistics
        with open(output_path / 'method_comparison_stats.json', 'w') as f:
            json.dump(convert_numpy_types(comparison_stats), f, indent=2)
        
        # 4. Save validity report
        with open(output_path / 'statistical_validity_report.json', 'w') as f:
            json.dump(convert_numpy_types(validity_report), f, indent=2)
        
        # 5. Create summary CSV for easy inspection
        summary_data = []
        for model in ['pre_brexit', 'post_brexit']:
            stats = comparison_stats[model]
            summary_data.append({
                'model': model,
                'original_significant': stats['original_count'],
                'bonferroni_significant': stats['bonferroni_count'],
                'fdr_significant': stats['fdr_count'],
                'holm_significant': stats['holm_count'],
                'bonferroni_reduction_rate': f"{stats['bonferroni_reduction']:.1%}",
                'fdr_reduction_rate': f"{stats['fdr_reduction']:.1%}",
                'holm_reduction_rate': f"{stats['holm_reduction']:.1%}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path / 'correction_summary.csv', index=False)
        
        logger.info("✅ All statistical validity results saved successfully!")
        
        # Log key findings
        self._log_key_findings(validity_report, comparison_stats)
    
    def _log_key_findings(self, validity_report: Dict[str, Any], comparison_stats: Dict[str, Any]):
        """Log key findings from the analysis"""
        logger.info("\n" + "="*60)
        logger.info("🎯 KEY STATISTICAL VALIDITY FINDINGS")
        logger.info("="*60)
        
        total = validity_report['metadata']['total_comparisons']
        expected_fp = validity_report['validity_assessment']['expected_false_positives_original']
        
        logger.info(f"📊 SCALE: {total:,} simultaneous comparisons")
        logger.info(f"⚠️  RISK: ~{expected_fp:.0f} expected false positives without correction")
        
        for model in ['pre_brexit', 'post_brexit']:
            orig = comparison_stats[model]['original_count']
            fdr = comparison_stats[model]['fdr_count']
            bonf = comparison_stats[model]['bonferroni_count']
            
            logger.info(f"\n🔍 {model.upper()} RESULTS:")
            logger.info(f"   Original: {orig:,} significant ({orig/total:.1%})")
            logger.info(f"   FDR: {fdr:,} significant ({fdr/total:.1%}) - RECOMMENDED")
            logger.info(f"   Bonferroni: {bonf:,} significant ({bonf/total:.1%}) - CONSERVATIVE")
        
        logger.info(f"\n✅ RECOMMENDATION: Use FDR-corrected vectors for balanced analysis")
        logger.info("="*60)

def main():
    """Main execution function"""
    
    # File paths
    unified_df_path = "/data/shil6369/gov_scraper/vignettes_analysis/comparative_fairness/outputs/unified_analysis/unified_fairness_dataframe_topic_granular.csv"
    output_dir = "/data/shil6369/gov_scraper/vignettes_analysis/comparative_fairness/outputs/statistical_validity"
    
    try:
        # Initialize analyzer
        analyzer = StatisticalValidityAnalyzer(unified_df_path)
        
        # Load data
        analyzer.load_data()
        
        # Apply multiple testing corrections
        correction_results = analyzer.apply_multiple_testing_corrections()
        
        # Compare correction methods
        comparison_stats = analyzer.compare_correction_methods(correction_results)
        
        # Create robust vectors
        robust_vectors = analyzer.create_robust_vectors(correction_results)
        
        # Generate validity report
        validity_report = analyzer.generate_validity_report(correction_results, comparison_stats)
        
        # Save all results
        analyzer.save_results(
            correction_results, robust_vectors, comparison_stats, 
            validity_report, output_dir
        )
        
        logger.info("🎉 Statistical validity analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 