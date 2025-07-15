#!/usr/bin/env python3
"""
Core Vector Divergence Analysis
Generic divergence calculation engine for fairness and normative analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.vector_utils import VectorProcessor, calculate_divergence_metrics

logger = logging.getLogger(__name__)

class DivergenceAnalyzer:
    """Core divergence analysis engine"""
    
    def __init__(self, config: Dict):
        """Initialize with configuration"""
        self.config = config
        self.divergence_config = config.get('divergence_analysis', {})
        self.vector_processor = VectorProcessor(config)
        self.results = {}
        
    def analyze_vector_divergence(self, 
                                 vector1: np.ndarray, 
                                 vector2: np.ndarray,
                                 weights: Optional[np.ndarray] = None,
                                 labels: Optional[List[str]] = None,
                                 analysis_name: str = "divergence") -> Dict[str, Any]:
        """
        Perform comprehensive divergence analysis between two vectors
        
        Args:
            vector1: First vector (e.g., pre-Brexit)
            vector2: Second vector (e.g., post-Brexit)
            weights: Optional weight vector
            labels: Optional labels for each comparison
            analysis_name: Name for this analysis
            
        Returns:
            Dictionary containing divergence results
        """
        logger.info(f"Starting divergence analysis: {analysis_name}")
        logger.info(f"Vector dimensions: {len(vector1)} x {len(vector2)}")
        
        # Validate inputs
        if len(vector1) != len(vector2):
            raise ValueError(f"Vector length mismatch: {len(vector1)} vs {len(vector2)}")
        
        if weights is not None and len(weights) != len(vector1):
            raise ValueError(f"Weight vector length mismatch: {len(weights)} vs {len(vector1)}")
        
        # Calculate basic statistics
        vector_stats = self._calculate_vector_statistics(vector1, vector2, weights)
        
        # Calculate divergence metrics
        methods = self.divergence_config.get('methods', ['cosine_similarity', 'pearson_correlation'])
        divergence_metrics = calculate_divergence_metrics(vector1, vector2, weights, methods)
        
        # Create results structure
        results = {
            'analysis_name': analysis_name,
            'vector_statistics': vector_stats,
            'divergence_metrics': divergence_metrics,
            'configuration': {
                'methods': methods,
                'weighted': weights is not None,
                'vector_length': len(vector1)
            }
        }
        
        # Add labels if provided
        if labels is not None:
            results['comparison_labels'] = labels
        
        # Store results
        self.results[analysis_name] = results
        
        logger.info(f"Completed divergence analysis: {analysis_name}")
        logger.info(f"Cosine similarity: {divergence_metrics.get('cosine_similarity', 'N/A'):.4f}")
        logger.info(f"Cosine distance: {divergence_metrics.get('cosine_distance', 'N/A'):.4f}")
        
        return results
    
    def analyze_significance_divergence(self,
                                      sig_vector1: np.ndarray,
                                      sig_vector2: np.ndarray,
                                      analysis_name: str = "significance_divergence") -> Dict[str, Any]:
        """
        Analyze divergence in significance patterns between two boolean vectors
        
        Args:
            sig_vector1: First significance vector (boolean)
            sig_vector2: Second significance vector (boolean)
            analysis_name: Name for this analysis
            
        Returns:
            Dictionary containing significance divergence results
        """
        logger.info(f"Starting significance divergence analysis: {analysis_name}")
        
        # Validate inputs
        if len(sig_vector1) != len(sig_vector2):
            raise ValueError(f"Significance vector length mismatch: {len(sig_vector1)} vs {len(sig_vector2)}")
        
        # Convert to boolean if needed
        sig_vector1 = sig_vector1.astype(bool)
        sig_vector2 = sig_vector2.astype(bool)
        
        # Calculate significance pattern statistics
        sig_stats = {
            'vector1_significant_count': int(np.sum(sig_vector1)),
            'vector2_significant_count': int(np.sum(sig_vector2)),
            'vector1_significance_rate': float(np.mean(sig_vector1)),
            'vector2_significance_rate': float(np.mean(sig_vector2)),
            'both_significant_count': int(np.sum(sig_vector1 & sig_vector2)),
            'neither_significant_count': int(np.sum(~sig_vector1 & ~sig_vector2)),
            'pattern_agreement_count': int(np.sum(sig_vector1 == sig_vector2)),
            'pattern_agreement_rate': float(np.mean(sig_vector1 == sig_vector2)),
            'pattern_change_count': int(np.sum(sig_vector1 != sig_vector2)),
            'pattern_change_rate': float(np.mean(sig_vector1 != sig_vector2))
        }
        
        # Calculate transitions
        transitions = {
            'gained_significance': int(np.sum(~sig_vector1 & sig_vector2)),  # False -> True
            'lost_significance': int(np.sum(sig_vector1 & ~sig_vector2)),   # True -> False
            'remained_significant': int(np.sum(sig_vector1 & sig_vector2)), # True -> True
            'remained_non_significant': int(np.sum(~sig_vector1 & ~sig_vector2)) # False -> False
        }
        
        # Create results
        results = {
            'analysis_name': analysis_name,
            'significance_statistics': sig_stats,
            'significance_transitions': transitions,
            'vector_length': len(sig_vector1)
        }
        
        # Store results
        self.results[analysis_name] = results
        
        logger.info(f"Completed significance divergence analysis: {analysis_name}")
        logger.info(f"Pattern agreement rate: {sig_stats['pattern_agreement_rate']:.2%}")
        logger.info(f"Pattern change rate: {sig_stats['pattern_change_rate']:.2%}")
        
        return results
    
    def _calculate_vector_statistics(self, 
                                   vector1: np.ndarray, 
                                   vector2: np.ndarray,
                                   weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate comprehensive statistics for two vectors"""
        
        stats = {}
        
        # Vector 1 statistics
        if weights is not None:
            stats['vector1'] = {
                'mean': float(np.average(vector1, weights=weights)),
                'std': float(np.sqrt(np.average((vector1 - np.average(vector1, weights=weights))**2, weights=weights))),
                'min': float(np.min(vector1)),
                'max': float(np.max(vector1)),
                'weighted': True
            }
        else:
            stats['vector1'] = {
                'mean': float(np.mean(vector1)),
                'std': float(np.std(vector1)),
                'min': float(np.min(vector1)),
                'max': float(np.max(vector1)),
                'weighted': False
            }
        
        # Vector 2 statistics
        if weights is not None:
            stats['vector2'] = {
                'mean': float(np.average(vector2, weights=weights)),
                'std': float(np.sqrt(np.average((vector2 - np.average(vector2, weights=weights))**2, weights=weights))),
                'min': float(np.min(vector2)),
                'max': float(np.max(vector2)),
                'weighted': True
            }
        else:
            stats['vector2'] = {
                'mean': float(np.mean(vector2)),
                'std': float(np.std(vector2)),
                'min': float(np.min(vector2)),
                'max': float(np.max(vector2)),
                'weighted': False
            }
        
        # Difference statistics
        diff = vector2 - vector1
        if weights is not None:
            stats['difference'] = {
                'mean': float(np.average(diff, weights=weights)),
                'std': float(np.sqrt(np.average((diff - np.average(diff, weights=weights))**2, weights=weights))),
                'min': float(np.min(diff)),
                'max': float(np.max(diff)),
                'mean_absolute': float(np.average(np.abs(diff), weights=weights)),
                'weighted': True
            }
        else:
            stats['difference'] = {
                'mean': float(np.mean(diff)),
                'std': float(np.std(diff)),
                'min': float(np.min(diff)),
                'max': float(np.max(diff)),
                'mean_absolute': float(np.mean(np.abs(diff))),
                'weighted': False
            }
        
        # Direction analysis
        stats['direction_analysis'] = {
            'positive_changes': int(np.sum(diff > 0)),
            'negative_changes': int(np.sum(diff < 0)),
            'no_changes': int(np.sum(diff == 0)),
            'positive_rate': float(np.mean(diff > 0)),
            'negative_rate': float(np.mean(diff < 0)),
            'no_change_rate': float(np.mean(diff == 0))
        }
        
        return stats
    
    def get_summary_report(self, analysis_name: str = None) -> str:
        """Generate a summary report for analysis results"""
        
        if analysis_name and analysis_name in self.results:
            results_to_report = {analysis_name: self.results[analysis_name]}
        else:
            results_to_report = self.results
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DIVERGENCE ANALYSIS SUMMARY")
        report_lines.append("=" * 80)
        
        for name, result in results_to_report.items():
            report_lines.append(f"\n## {name.upper()}")
            report_lines.append("-" * 40)
            
            # Divergence metrics
            if 'divergence_metrics' in result:
                metrics = result['divergence_metrics']
                report_lines.append("\nDivergence Metrics:")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        report_lines.append(f"  {metric}: {value:.4f}")
                    else:
                        report_lines.append(f"  {metric}: {value}")
            
            # Vector statistics
            if 'vector_statistics' in result:
                stats = result['vector_statistics']
                report_lines.append("\nVector Statistics:")
                for vector_name, vector_stats in stats.items():
                    if isinstance(vector_stats, dict):
                        report_lines.append(f"  {vector_name}:")
                        for stat_name, stat_value in vector_stats.items():
                            if isinstance(stat_value, float):
                                report_lines.append(f"    {stat_name}: {stat_value:.4f}")
                            else:
                                report_lines.append(f"    {stat_name}: {stat_value}")
            
            # Significance statistics
            if 'significance_statistics' in result:
                sig_stats = result['significance_statistics']
                report_lines.append("\nSignificance Pattern Analysis:")
                for stat_name, stat_value in sig_stats.items():
                    if 'rate' in stat_name and isinstance(stat_value, float):
                        report_lines.append(f"  {stat_name}: {stat_value:.2%}")
                    else:
                        report_lines.append(f"  {stat_name}: {stat_value}")
            
            # Transitions
            if 'significance_transitions' in result:
                transitions = result['significance_transitions']
                report_lines.append("\nSignificance Transitions:")
                for transition, count in transitions.items():
                    report_lines.append(f"  {transition}: {count}")
        
        return "\n".join(report_lines)
    
    def save_results(self, output_dir: Path, formats: List[str] = None):
        """Save analysis results to files"""
        
        if formats is None:
            formats = self.config.get('output', {}).get('formats', ['json'])
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        if 'json' in formats:
            import json
            results_file = output_dir / "divergence_analysis_results.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            logger.info(f"Saved results to {results_file}")
        
        if 'csv' in formats and self.results:
            # Create summary CSV
            summary_data = []
            for analysis_name, result in self.results.items():
                row = {'analysis_name': analysis_name}
                
                # Add divergence metrics
                if 'divergence_metrics' in result:
                    for metric, value in result['divergence_metrics'].items():
                        row[f'divergence_{metric}'] = value
                
                # Add key statistics
                if 'vector_statistics' in result:
                    stats = result['vector_statistics']
                    if 'difference' in stats:
                        row['mean_difference'] = stats['difference']['mean']
                        row['std_difference'] = stats['difference']['std']
                        row['mean_absolute_difference'] = stats['difference']['mean_absolute']
                
                # Add significance statistics
                if 'significance_statistics' in result:
                    sig_stats = result['significance_statistics']
                    row['pattern_agreement_rate'] = sig_stats.get('pattern_agreement_rate', None)
                    row['pattern_change_rate'] = sig_stats.get('pattern_change_rate', None)
                
                summary_data.append(row)
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_file = output_dir / "divergence_analysis_summary.csv"
                summary_df.to_csv(summary_file, index=False)
                logger.info(f"Saved summary to {summary_file}")
        
        # Save text report
        report = self.get_summary_report()
        report_file = output_dir / "divergence_analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Saved report to {report_file}")

def run_divergence_analysis(df1: pd.DataFrame, 
                           df2: pd.DataFrame,
                           config: Dict,
                           analysis_type: str = "generic") -> DivergenceAnalyzer:
    """
    Convenience function to run divergence analysis on two dataframes
    
    Args:
        df1: First dataframe
        df2: Second dataframe  
        config: Configuration dictionary
        analysis_type: Type of analysis ("fairness", "normative", "generic")
        
    Returns:
        DivergenceAnalyzer with completed analysis
    """
    analyzer = DivergenceAnalyzer(config)
    
    # Extract vectors based on analysis type
    vector_config = config.get('vector_extraction', {}).get(analysis_type, {})
    
    if not vector_config:
        logger.warning(f"No vector configuration found for analysis type: {analysis_type}")
        return analyzer
    
    magnitude_cols = vector_config.get('magnitude_columns', [])
    significance_cols = vector_config.get('significance_columns', [])
    group_size_cols = vector_config.get('group_size_columns', [])
    
    # Use vector processor to extract and analyze
    processor = VectorProcessor(config)
    
    # This is a simplified example - in practice, you'd extract specific vectors
    # based on the dataframe structure and analysis requirements
    
    logger.info(f"Completed {analysis_type} divergence analysis setup")
    
    return analyzer 