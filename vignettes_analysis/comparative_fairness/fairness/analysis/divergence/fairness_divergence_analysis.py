#!/usr/bin/env python3
"""
Fairness Divergence Analysis
Main orchestrator for comprehensive fairness divergence analysis using SP vectors
"""

import yaml
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from sp_vector_extractor import SPVectorExtractor
from utils.vector_utils import VectorProcessor
sys.path.append(str(Path(__file__).parent.parent / "05_divergence_analysis"))
from vector_divergence import DivergenceAnalyzer

logger = logging.getLogger(__name__)

class FairnessDivergenceAnalyzer:
    """Main fairness divergence analysis orchestrator"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with configuration"""
        
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "divergence_config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize components
        self.sp_extractor = SPVectorExtractor(self.config)
        self.divergence_analyzer = DivergenceAnalyzer(self.config)
        self.vector_processor = VectorProcessor(self.config)
        
        # Results storage
        self.results = {}
        self.vectors_data = None
        
        # Setup logging
        self._setup_logging()
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def _setup_logging(self):
        """Setup logging based on configuration"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        
        # Configure logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if log_config.get('save_detailed_logs', False):
            # Add file handler
            log_dir = Path(self.config_path).parent.parent / "outputs" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_dir / "fairness_divergence_analysis.log")
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
    
    def extract_sp_vectors(self) -> Dict[str, Any]:
        """Extract SP vectors from fairness dataframe"""
        logger.info("=" * 80)
        logger.info("EXTRACTING STATISTICAL PARITY VECTORS")
        logger.info("=" * 80)
        
        # Extract all vectors
        self.vectors_data = self.sp_extractor.extract_all_vectors()
        
        logger.info("SP vector extraction completed successfully")
        logger.info(f"Total vectors extracted: {len(self.vectors_data['vectors'])}")
        logger.info(f"Vector dimensions: {len(list(self.vectors_data['vectors'].values())[0])}")
        
        return self.vectors_data
    
    def analyze_magnitude_divergence(self) -> Dict[str, Any]:
        """Analyze SP magnitude divergence between models"""
        logger.info("=" * 80)
        logger.info("ANALYZING SP MAGNITUDE DIVERGENCE")
        logger.info("=" * 80)
        
        if self.vectors_data is None:
            self.extract_sp_vectors()
        
        vectors = self.vectors_data['vectors']
        labels = self.vectors_data['labels']
        
        # Extract pre and post Brexit magnitude vectors
        pre_brexit_vector = vectors['pre_brexit_sp_magnitude']
        post_brexit_vector = vectors['post_brexit_sp_magnitude']
        
        # Create weights if available
        weights = None
        if 'min_overall_size' in vectors:
            weights = self.vector_processor.create_weights(
                {'group_size_0': vectors['min_overall_size']},
                strategy=self.config.get('divergence_analysis', {}).get('weighting', {}).get('strategy', 'min_sample_size')
            )
        
        # Perform divergence analysis
        magnitude_results = self.divergence_analyzer.analyze_vector_divergence(
            vector1=pre_brexit_vector,
            vector2=post_brexit_vector,
            weights=weights,
            labels=labels,
            analysis_name="sp_magnitude_divergence"
        )
        
        self.results['magnitude_divergence'] = magnitude_results
        
        logger.info("SP magnitude divergence analysis completed")
        return magnitude_results
    
    def analyze_significance_divergence(self) -> Dict[str, Any]:
        """Analyze SP significance pattern divergence between models"""
        logger.info("=" * 80)
        logger.info("ANALYZING SP SIGNIFICANCE DIVERGENCE")
        logger.info("=" * 80)
        
        if self.vectors_data is None:
            self.extract_sp_vectors()
        
        vectors = self.vectors_data['vectors']
        
        # Extract significance vectors
        pre_brexit_sig = vectors['pre_brexit_sp_significance']
        post_brexit_sig = vectors['post_brexit_sp_significance']
        
        # Perform significance divergence analysis
        significance_results = self.divergence_analyzer.analyze_significance_divergence(
            sig_vector1=pre_brexit_sig,
            sig_vector2=post_brexit_sig,
            analysis_name="sp_significance_divergence"
        )
        
        self.results['significance_divergence'] = significance_results
        
        logger.info("SP significance divergence analysis completed")
        return significance_results
    
    def analyze_stratified_divergence(self, stratify_by: str = 'protected_attribute') -> Dict[str, Any]:
        """Analyze divergence stratified by protected attributes or topics"""
        logger.info("=" * 80)
        logger.info(f"ANALYZING STRATIFIED DIVERGENCE BY {stratify_by.upper()}")
        logger.info("=" * 80)
        
        # Get stratified vectors
        stratified_data = self.sp_extractor.get_stratified_vectors(stratify_by=stratify_by)
        
        stratified_results = {}
        
        for stratum_value, stratum_data in stratified_data.items():
            logger.info(f"Analyzing stratum: {stratum_value}")
            
            vectors = stratum_data['vectors']
            labels = stratum_data['labels']
            
            # Skip if insufficient data
            if len(labels) < 5:
                logger.warning(f"Insufficient data for stratum {stratum_value}: {len(labels)} comparisons")
                continue
            
            # Extract vectors
            pre_brexit_vector = vectors['pre_brexit_sp_magnitude']
            post_brexit_vector = vectors['post_brexit_sp_magnitude']
            
            # Create weights if available
            weights = None
            if 'min_overall_size' in vectors:
                weights = self.vector_processor.create_weights(
                    {'group_size_0': vectors['min_overall_size']},
                    strategy=self.config.get('divergence_analysis', {}).get('weighting', {}).get('strategy', 'min_sample_size')
                )
            
            # Analyze magnitude divergence for this stratum
            magnitude_analysis = self.divergence_analyzer.analyze_vector_divergence(
                vector1=pre_brexit_vector,
                vector2=post_brexit_vector,
                weights=weights,
                labels=labels,
                analysis_name=f"{stratify_by}_{stratum_value}_magnitude"
            )
            
            # Analyze significance divergence for this stratum
            if 'pre_brexit_sp_significance' in vectors and 'post_brexit_sp_significance' in vectors:
                significance_analysis = self.divergence_analyzer.analyze_significance_divergence(
                    sig_vector1=vectors['pre_brexit_sp_significance'],
                    sig_vector2=vectors['post_brexit_sp_significance'],
                    analysis_name=f"{stratify_by}_{stratum_value}_significance"
                )
            else:
                significance_analysis = None
            
            # Store results for this stratum
            stratified_results[stratum_value] = {
                'magnitude_divergence': magnitude_analysis,
                'significance_divergence': significance_analysis,
                'stratum_info': stratum_data['subset_info']
            }
        
        self.results[f'stratified_by_{stratify_by}'] = stratified_results
        
        logger.info(f"Stratified divergence analysis by {stratify_by} completed")
        logger.info(f"Analyzed {len(stratified_results)} strata")
        
        return stratified_results
    
    def run_comprehensive_analysis(self, include_stratification: bool = True) -> Dict[str, Any]:
        """Run comprehensive fairness divergence analysis"""
        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE FAIRNESS DIVERGENCE ANALYSIS")
        logger.info("=" * 80)
        
        # Step 1: Extract SP vectors
        self.extract_sp_vectors()
        
        # Step 2: Analyze overall magnitude divergence
        self.analyze_magnitude_divergence()
        
        # Step 3: Analyze overall significance divergence
        self.analyze_significance_divergence()
        
        # Step 4: Stratified analysis if requested
        if include_stratification:
            stratification_config = self.config.get('stratification', {}).get('fairness', {})
            
            # By protected attribute
            if stratification_config.get('by_attribute', {}).get('enabled', True):
                self.analyze_stratified_divergence('protected_attribute')
            
            # By topic
            if stratification_config.get('by_topic', {}).get('enabled', True):
                self.analyze_stratified_divergence('topic')
        
        logger.info("Comprehensive fairness divergence analysis completed")
        
        return self.results
    
    def generate_summary_report(self) -> str:
        """Generate comprehensive summary report"""
        if not self.results:
            return "No analysis results available. Run analysis first."
        
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("COMPREHENSIVE FAIRNESS DIVERGENCE ANALYSIS SUMMARY")
        report_lines.append("=" * 100)
        
        # Overall analysis summary
        if self.vectors_data:
            info = self.vectors_data['dataframe_info']
            report_lines.append(f"\nDataset Summary:")
            report_lines.append(f"  Total comparisons: {info['total_comparisons']}")
            report_lines.append(f"  Protected attributes: {info['protected_attributes']}")
            report_lines.append(f"  Topics: {info['topics']}")
            
            if 'attributes_breakdown' in info:
                report_lines.append(f"\nAttribute breakdown:")
                for attr, count in info['attributes_breakdown'].items():
                    report_lines.append(f"    {attr}: {count} comparisons")
        
        # Overall magnitude divergence
        if 'magnitude_divergence' in self.results:
            mag_results = self.results['magnitude_divergence']
            metrics = mag_results['divergence_metrics']
            
            report_lines.append(f"\n" + "=" * 80)
            report_lines.append("OVERALL SP MAGNITUDE DIVERGENCE")
            report_lines.append("=" * 80)
            
            report_lines.append(f"Cosine Similarity: {metrics.get('cosine_similarity', 'N/A'):.4f}")
            report_lines.append(f"Cosine Distance: {metrics.get('cosine_distance', 'N/A'):.4f}")
            
            if 'pearson_correlation' in metrics:
                report_lines.append(f"Pearson Correlation: {metrics['pearson_correlation']:.4f}")
                if 'pearson_p_value' in metrics:
                    report_lines.append(f"Pearson P-value: {metrics['pearson_p_value']:.6f}")
            
            # Vector statistics
            if 'vector_statistics' in mag_results:
                stats = mag_results['vector_statistics']
                if 'difference' in stats:
                    diff_stats = stats['difference']
                    report_lines.append(f"Mean difference: {diff_stats['mean']:.4f}")
                    report_lines.append(f"Standard deviation: {diff_stats['std']:.4f}")
                    report_lines.append(f"Mean absolute difference: {diff_stats['mean_absolute']:.4f}")
        
        # Overall significance divergence
        if 'significance_divergence' in self.results:
            sig_results = self.results['significance_divergence']
            sig_stats = sig_results['significance_statistics']
            
            report_lines.append(f"\n" + "=" * 80)
            report_lines.append("OVERALL SP SIGNIFICANCE DIVERGENCE")
            report_lines.append("=" * 80)
            
            report_lines.append(f"Pattern Agreement Rate: {sig_stats['pattern_agreement_rate']:.2%}")
            report_lines.append(f"Pattern Change Rate: {sig_stats['pattern_change_rate']:.2%}")
            
            report_lines.append(f"\nSignificance Rates:")
            report_lines.append(f"  Pre-Brexit: {sig_stats['vector1_significance_rate']:.2%}")
            report_lines.append(f"  Post-Brexit: {sig_stats['vector2_significance_rate']:.2%}")
            
            if 'significance_transitions' in sig_results:
                transitions = sig_results['significance_transitions']
                report_lines.append(f"\nSignificance Transitions:")
                report_lines.append(f"  Gained significance: {transitions['gained_significance']}")
                report_lines.append(f"  Lost significance: {transitions['lost_significance']}")
                report_lines.append(f"  Remained significant: {transitions['remained_significant']}")
                report_lines.append(f"  Remained non-significant: {transitions['remained_non_significant']}")
        
        # Stratified analysis summary
        for result_key, result_data in self.results.items():
            if result_key.startswith('stratified_by_'):
                stratify_by = result_key.replace('stratified_by_', '')
                
                report_lines.append(f"\n" + "=" * 80)
                report_lines.append(f"STRATIFIED ANALYSIS BY {stratify_by.upper()}")
                report_lines.append("=" * 80)
                
                # Summary table of divergence by stratum
                report_lines.append(f"\n{'Stratum':<20} | {'Cosine Sim':<10} | {'Cosine Dist':<11} | {'Pattern Agr':<11} | {'Comparisons':<12}")
                report_lines.append("-" * 80)
                
                for stratum, stratum_results in result_data.items():
                    mag_div = stratum_results.get('magnitude_divergence', {})
                    sig_div = stratum_results.get('significance_divergence', {})
                    
                    cos_sim = mag_div.get('divergence_metrics', {}).get('cosine_similarity', 0)
                    cos_dist = mag_div.get('divergence_metrics', {}).get('cosine_distance', 0)
                    
                    pattern_agr = sig_div.get('significance_statistics', {}).get('pattern_agreement_rate', 0) if sig_div else 0
                    
                    comparisons = stratum_results.get('stratum_info', {}).get('comparisons', 0)
                    
                    stratum_name = stratum[:18] + '..' if len(stratum) > 20 else stratum
                    
                    report_lines.append(f"{stratum_name:<20} | {cos_sim:<10.4f} | {cos_dist:<11.4f} | {pattern_agr:<11.2%} | {comparisons:<12}")
        
        # Interpretation guidelines
        report_lines.append(f"\n" + "=" * 80)
        report_lines.append("INTERPRETATION GUIDELINES")
        report_lines.append("=" * 80)
        
        report_lines.append("\nMagnitude Divergence (Cosine Similarity):")
        report_lines.append("  1.0 = Identical bias patterns")
        report_lines.append("  0.8-1.0 = High similarity (low divergence)")
        report_lines.append("  0.6-0.8 = Moderate similarity")
        report_lines.append("  0.0-0.6 = Low similarity (high divergence)")
        report_lines.append("  <0.0 = Opposite bias patterns")
        
        report_lines.append("\nSignificance Divergence (Pattern Agreement Rate):")
        report_lines.append("  90-100% = Very consistent significance patterns")
        report_lines.append("  75-90% = Moderately consistent patterns")
        report_lines.append("  50-75% = Somewhat inconsistent patterns")
        report_lines.append("  <50% = Highly inconsistent patterns")
        
        return "\n".join(report_lines)
    
    def save_results(self, output_dir: Optional[str] = None):
        """Save all analysis results to files"""
        if output_dir is None:
            output_dir = Path(self.config_path).parent.parent / "outputs" / "fairness_divergence"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving results to {output_dir}")
        
        # Save detailed results
        results_file = output_dir / "fairness_divergence_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save vectors data
        if self.vectors_data:
            vectors_file = output_dir / "sp_vectors_data.json"
            # Convert numpy arrays to lists for JSON serialization
            serializable_vectors = {}
            for key, value in self.vectors_data['vectors'].items():
                if isinstance(value, np.ndarray):
                    serializable_vectors[key] = value.tolist()
                else:
                    serializable_vectors[key] = value
            
            vectors_data_to_save = {
                'vectors': serializable_vectors,
                'labels': self.vectors_data['labels'],
                'dataframe_info': self.vectors_data['dataframe_info']
            }
            
            with open(vectors_file, 'w') as f:
                json.dump(vectors_data_to_save, f, indent=2, default=str)
        
        # Save summary report
        report = self.generate_summary_report()
        report_file = output_dir / "fairness_divergence_summary.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Use divergence analyzer to save its results too
        self.divergence_analyzer.save_results(output_dir)
        
        logger.info(f"Results saved successfully to {output_dir}")

def main():
    """Main function for running fairness divergence analysis"""
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Initialize analyzer
        analyzer = FairnessDivergenceAnalyzer()
        
        # Run comprehensive analysis
        results = analyzer.run_comprehensive_analysis(include_stratification=True)
        
        # Generate and print summary
        summary = analyzer.generate_summary_report()
        print(summary)
        
        # Save results
        analyzer.save_results()
        
        logger.info("Fairness divergence analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 