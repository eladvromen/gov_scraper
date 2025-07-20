#!/usr/bin/env python3
"""
Bias Vector Drift Analysis
==========================

Analyzes how fairness disparities changed between pre-Brexit and post-Brexit models
using cosine similarity, Jaccard similarity, and significance change analysis.

Key Features:
- Bidirectional redundancy removal (e.g., keeps China_vs_Ukraine, removes Ukraine_vs_China)
- Cosine similarity between magnitude vectors (full + significance-filtered)
- Jaccard similarity between significance vectors  
- Detailed significance change counts and percentages
- Visualization of magnitude differences
- Table of most shifted bias pairs

Author: Bias Drift Analysis Pipeline
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from scipy.spatial.distance import cosine
from sklearn.metrics import jaccard_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BiasVectorDriftAnalyzer:
    """
    Analyzes bias vector drift between pre-Brexit and post-Brexit models
    """
    
    def __init__(self, sp_vectors_path: str):
        """
        Initialize the analyzer
        
        Args:
            sp_vectors_path: Path to SP vectors data JSON file
        """
        self.sp_vectors_path = sp_vectors_path
        self.data = None
        self.results = {}
        
    def load_vectors(self) -> Dict[str, Any]:
        """
        Step 1: Load and prepare vectors from the SP vectors data file
        
        Returns:
            Dictionary containing all loaded vectors
        """
        logger.info(f"Loading vectors from {self.sp_vectors_path}")
        
        with open(self.sp_vectors_path, 'r') as f:
            raw_data = json.load(f)
        
        # Extract vectors
        vectors = raw_data['vectors']
        
        # Load FDR-corrected significance vectors from robust vectors file
        robust_vectors_path = self.sp_vectors_path.replace('sp_vectors_data.json', '../statistical_validity/robust_significance_vectors.json')
        logger.info(f"Loading FDR-corrected significance vectors from {robust_vectors_path}")
        
        with open(robust_vectors_path, 'r') as f:
            robust_data = json.load(f)
        
        robust_vectors = robust_data['vectors']
        
        # Raw data before filtering
        raw_vectors = {
            'pre_brexit_sp_magnitude': np.array(vectors['pre_brexit_sp_magnitude']),
            'post_brexit_sp_magnitude': np.array(vectors['post_brexit_sp_magnitude']),
            'sp_magnitude_difference': np.array(vectors['sp_magnitude_difference']),
            'pre_brexit_sp_significance': np.array(robust_vectors['pre_brexit_sp_significance_fdr']),  # FDR corrected
            'post_brexit_sp_significance': np.array(robust_vectors['post_brexit_sp_significance_fdr']),  # FDR corrected
            'sp_gained_significance': np.array(robust_vectors['sp_gained_significance_fdr']),  # FDR corrected
            'sp_lost_significance': np.array(robust_vectors['sp_lost_significance_fdr']),  # FDR corrected
            'sp_both_significant': np.array(robust_vectors['sp_both_significant_fdr']),  # FDR corrected
            'sp_neither_significant': np.array(robust_vectors['sp_neither_significant_fdr']),  # FDR corrected
            'labels': raw_data['labels'],  # Labels are at root level, not in vectors
        }
        
        logger.info(f"Loaded {len(raw_vectors['pre_brexit_sp_magnitude'])} raw comparisons")
        
        # Step 1a: Filter out bidirectional redundancies
        self.data = self._filter_bidirectional_redundancies(raw_vectors, raw_data['dataframe_info'])
        
        logger.info(f"After deduplication: {len(self.data['pre_brexit_sp_magnitude'])} unique comparisons")
        logger.info(f"Vector dimensions: {self.data['pre_brexit_sp_magnitude'].shape}")
        logger.info(f"Using FDR-corrected significance vectors")
        
        return self.data
    
    def _filter_bidirectional_redundancies(self, raw_vectors: Dict[str, np.ndarray], dataframe_info: Dict) -> Dict[str, Any]:
        """
        Filter out bidirectional redundancies (e.g., keep only China_vs_Ukraine, remove Ukraine_vs_China)
        
        Args:
            raw_vectors: Dictionary of raw vectors before filtering
            dataframe_info: Metadata about the dataframe
            
        Returns:
            Dictionary of filtered vectors with redundancies removed
        """
        logger.info("Filtering bidirectional redundancies...")
        
        labels = raw_vectors['labels']
        unique_indices = []
        seen_pairs = set()
        
        for i, label in enumerate(labels):
            # Parse the comparison label to extract groups and context
            if '_vs_' in label:
                # Split by _vs_ first to get the main comparison
                parts = label.split('_vs_')
                if len(parts) == 2:
                    group1_full = parts[0]
                    group2_and_context = parts[1]
                    
                    # Special handling for cases where group2 is '_' (empty religion)
                    # This happens when group2_and_context starts with " (" (space + parenthesis)
                    # indicating the group name was '_' which got consumed by the split
                    if group2_and_context.startswith(' ('):
                        # Case: Something_vs_ (context) -> group2 was '_' (empty)
                        group2 = '_'
                        context = group2_and_context  # Keep the entire context including the space
                    else:
                        # Extract group2 by finding the pattern with parentheses
                        if '(' in group2_and_context and ')' in group2_and_context:
                            # Find the end of group2 name (before the space and parentheses)
                            paren_start = group2_and_context.find(' (')
                            if paren_start > 0:
                                group2 = group2_and_context[:paren_start]
                                context = group2_and_context[paren_start:]
                            else:
                                # Special case: group2 might be just "_" or other single char before space
                                space_idx = group2_and_context.find(' ')
                                if space_idx > 0:
                                    group2 = group2_and_context[:space_idx]
                                    context = group2_and_context[space_idx:]
                                else:
                                    group2 = group2_and_context
                                    context = ""
                        else:
                            group2 = group2_and_context
                            context = ""
                    
                    # Normalize empty religion representations: convert '_' to empty string
                    if group1_full == '_':
                        group1_full = ''
                    if group2 == '_':
                        group2 = ''
                    
                    # Create normalized pair (alphabetically sorted) + context
                    normalized_pair = tuple(sorted([group1_full, group2])) + (context,)
                    
                    if normalized_pair not in seen_pairs:
                        seen_pairs.add(normalized_pair)
                        unique_indices.append(i)
                    else:
                        logger.debug(f"Skipping redundant comparison: {label}")
                else:
                    # Keep comparisons that don't follow expected pattern
                    unique_indices.append(i)
            else:
                # Keep comparisons without _vs_ pattern
                unique_indices.append(i)
        
        logger.info(f"Identified {len(unique_indices)} unique comparisons out of {len(labels)} total")
        logger.info(f"Removed {len(labels) - len(unique_indices)} redundant bidirectional pairs")
        
        # Filter all vectors using unique indices
        filtered_vectors = {}
        for key, vector in raw_vectors.items():
            if key == 'labels':
                filtered_vectors[key] = [labels[i] for i in unique_indices]
            else:
                filtered_vectors[key] = vector[unique_indices]
        
        # Add dataframe info
        filtered_vectors['dataframe_info'] = dataframe_info.copy()
        filtered_vectors['dataframe_info']['original_total_comparisons'] = len(labels)
        filtered_vectors['dataframe_info']['filtered_total_comparisons'] = len(unique_indices)
        filtered_vectors['dataframe_info']['redundancy_removed'] = len(labels) - len(unique_indices)
        
        # Log some examples of kept vs removed
        logger.info("Examples of filtering:")
        example_count = 0
        for i, label in enumerate(labels[:50]):  # Check first 50 for examples
            if i in unique_indices and example_count < 3:
                logger.info(f"  ✅ Kept: {label}")
                example_count += 1
            elif i not in unique_indices and example_count < 6:
                logger.info(f"  ❌ Removed: {label}")
                example_count += 1
            if example_count >= 6:
                break
        
        return filtered_vectors
    
    def compute_cosine_similarity(self) -> Dict[str, float]:
        """
        Step 2: Compute cosine similarity between magnitude vectors
        
        Returns:
            Dictionary with full and significance-filtered cosine similarities
        """
        logger.info("Computing cosine similarity between magnitude vectors...")
        
        pre_mag = self.data['pre_brexit_sp_magnitude']
        post_mag = self.data['post_brexit_sp_magnitude']
        both_significant = self.data['sp_both_significant']
        
        # Full vector cosine similarity
        cosine_similarity_full = 1 - cosine(pre_mag, post_mag)
        
        # Significance-filtered cosine similarity (only where both models are significant)
        if np.sum(both_significant) > 0:
            pre_mag_filtered = pre_mag[both_significant]
            post_mag_filtered = post_mag[both_significant]
            cosine_similarity_filtered = 1 - cosine(pre_mag_filtered, post_mag_filtered)
        else:
            cosine_similarity_filtered = np.nan
            logger.warning("No comparisons where both models are significant")
        
        cosine_results = {
            'full_vector': cosine_similarity_full,
            'significance_filtered': cosine_similarity_filtered,
            'filtered_sample_size': np.sum(both_significant)
        }
        
        logger.info(f"Cosine similarity (full): {cosine_similarity_full:.4f}")
        logger.info(f"Cosine similarity (filtered): {cosine_similarity_filtered:.4f}")
        logger.info(f"Filtered sample size: {np.sum(both_significant)}")
        
        self.results['cosine_similarity'] = cosine_results
        return cosine_results
    
    def compute_jaccard_similarity(self) -> Dict[str, float]:
        """
        Step 3: Compute Jaccard similarity between significance vectors
        
        Returns:
            Dictionary with Jaccard similarity metrics
        """
        logger.info("Computing Jaccard similarity between significance vectors...")
        
        pre_sig = self.data['pre_brexit_sp_significance']
        post_sig = self.data['post_brexit_sp_significance']
        
        # Jaccard similarity for significance overlap
        jaccard_similarity = jaccard_score(pre_sig, post_sig)
        
        # Additional overlap metrics
        intersection = np.sum(pre_sig & post_sig)
        union = np.sum(pre_sig | post_sig)
        overlap_coefficient = intersection / min(np.sum(pre_sig), np.sum(post_sig)) if min(np.sum(pre_sig), np.sum(post_sig)) > 0 else 0
        
        jaccard_results = {
            'jaccard_similarity': jaccard_similarity,
            'intersection_count': int(intersection),
            'union_count': int(union),
            'overlap_coefficient': overlap_coefficient,
            'pre_significant_count': int(np.sum(pre_sig)),
            'post_significant_count': int(np.sum(post_sig))
        }
        
        logger.info(f"Jaccard similarity: {jaccard_similarity:.4f}")
        logger.info(f"Intersection: {intersection}, Union: {union}")
        logger.info(f"Pre-Brexit significant: {np.sum(pre_sig)}, Post-Brexit significant: {np.sum(post_sig)}")
        
        self.results['jaccard_similarity'] = jaccard_results
        return jaccard_results
    
    def count_significance_changes(self) -> Dict[str, Any]:
        """
        Step 4: Calculate raw counts and percentages for significance changes
        
        Returns:
            Dictionary with detailed significance change statistics
        """
        logger.info("Counting significance changes...")
        
        total_comparisons = len(self.data['pre_brexit_sp_magnitude'])
        
        gained = np.sum(self.data['sp_gained_significance'])
        lost = np.sum(self.data['sp_lost_significance'])
        both = np.sum(self.data['sp_both_significant'])
        neither = np.sum(self.data['sp_neither_significant'])
        
        # Verify counts add up
        total_check = gained + lost + both + neither
        assert total_check == total_comparisons, f"Count mismatch: {total_check} != {total_comparisons}"
        
        change_results = {
            'total_comparisons': total_comparisons,
            'gained_significance': {
                'count': int(gained),
                'percentage': (gained / total_comparisons) * 100
            },
            'lost_significance': {
                'count': int(lost),
                'percentage': (lost / total_comparisons) * 100
            },
            'both_significant': {
                'count': int(both),
                'percentage': (both / total_comparisons) * 100
            },
            'neither_significant': {
                'count': int(neither),
                'percentage': (neither / total_comparisons) * 100
            },
            'total_changes': {
                'count': int(gained + lost),
                'percentage': ((gained + lost) / total_comparisons) * 100
            },
            'total_stable': {
                'count': int(both + neither),
                'percentage': ((both + neither) / total_comparisons) * 100
            }
        }
        
        logger.info(f"Significance changes:")
        logger.info(f"  Gained: {gained} ({gained/total_comparisons*100:.1f}%)")
        logger.info(f"  Lost: {lost} ({lost/total_comparisons*100:.1f}%)")
        logger.info(f"  Both significant: {both} ({both/total_comparisons*100:.1f}%)")
        logger.info(f"  Neither significant: {neither} ({neither/total_comparisons*100:.1f}%)")
        
        self.results['significance_changes'] = change_results
        return change_results
    
    def visualize_magnitude_changes(self, save_path: str = None) -> plt.Figure:
        """
        Step 5: Plot histogram of magnitude differences
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        logger.info("Creating magnitude difference visualization...")
        
        mag_diff = self.data['sp_magnitude_difference']
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram of magnitude differences
        ax1.hist(mag_diff, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--', alpha=0.8, label='No change')
        ax1.axvline(0.2, color='orange', linestyle='--', alpha=0.8, label='±0.2 threshold')
        ax1.axvline(-0.2, color='orange', linestyle='--', alpha=0.8)
        ax1.set_xlabel('Statistical Parity Magnitude Difference (Post - Pre Brexit)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Bias Magnitude Changes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""Statistics:
Mean: {np.mean(mag_diff):.4f}
Std: {np.std(mag_diff):.4f}
Min: {np.min(mag_diff):.4f}
Max: {np.max(mag_diff):.4f}
|Change| > 0.2: {np.sum(np.abs(mag_diff) > 0.2)}
"""
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Box plot by significance category
        significance_categories = []
        magnitude_by_category = []
        
        categories = ['Gained Sig.', 'Lost Sig.', 'Both Sig.', 'Neither Sig.']
        masks = [
            self.data['sp_gained_significance'],
            self.data['sp_lost_significance'], 
            self.data['sp_both_significant'],
            self.data['sp_neither_significant']
        ]
        
        for i, (cat, mask) in enumerate(zip(categories, masks)):
            if np.sum(mask) > 0:
                significance_categories.extend([cat] * np.sum(mask))
                magnitude_by_category.extend(mag_diff[mask])
        
        if magnitude_by_category:
            df_plot = pd.DataFrame({
                'Category': significance_categories,
                'Magnitude_Difference': magnitude_by_category
            })
            
            sns.boxplot(data=df_plot, x='Category', y='Magnitude_Difference', ax=ax2)
            ax2.axhline(0, color='red', linestyle='--', alpha=0.8)
            ax2.axhline(0.2, color='orange', linestyle='--', alpha=0.8)
            ax2.axhline(-0.2, color='orange', linestyle='--', alpha=0.8)
            ax2.set_ylabel('Magnitude Difference')
            ax2.set_title('Magnitude Changes by Significance Category')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        self.results['magnitude_visualization'] = {
            'figure_created': True,
            'save_path': save_path,
            'large_changes_count': int(np.sum(np.abs(mag_diff) > 0.2)),
            'large_changes_percentage': (np.sum(np.abs(mag_diff) > 0.2) / len(mag_diff)) * 100
        }
        
        return fig
    
    def identify_most_shifted_pairs(self, top_n: int = 20) -> pd.DataFrame:
        """
        Create table of most dramatically shifted bias pairs
        
        Args:
            top_n: Number of top shifted pairs to return
            
        Returns:
            DataFrame with most shifted comparisons
        """
        logger.info(f"Identifying top {top_n} most shifted bias pairs...")
        
        mag_diff = self.data['sp_magnitude_difference']
        abs_mag_diff = np.abs(mag_diff)
        
        # Get top shifted indices
        top_indices = np.argsort(abs_mag_diff)[-top_n:][::-1]
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'Rank': range(1, top_n + 1),
            'Comparison': [self.data['labels'][i] for i in top_indices],
            'Pre_Brexit_Magnitude': self.data['pre_brexit_sp_magnitude'][top_indices],
            'Post_Brexit_Magnitude': self.data['post_brexit_sp_magnitude'][top_indices],
            'Magnitude_Difference': mag_diff[top_indices],
            'Absolute_Difference': abs_mag_diff[top_indices],
            'Pre_Brexit_Significant': self.data['pre_brexit_sp_significance'][top_indices],
            'Post_Brexit_Significant': self.data['post_brexit_sp_significance'][top_indices],
            'Significance_Change': self.data['sp_gained_significance'][top_indices] | self.data['sp_lost_significance'][top_indices]
        })
        
        logger.info(f"Top shifted pair: {results_df.iloc[0]['Comparison']}")
        logger.info(f"  Magnitude difference: {results_df.iloc[0]['Magnitude_Difference']:.4f}")
        
        self.results['most_shifted_pairs'] = results_df
        return results_df
    def save_deduplicated_dataframe(self, output_dir: str):
        """
        Save the deduplicated data as a CSV for use by other analysis scripts
        
        Args:
            output_dir: Directory to save the deduplicated dataframe
        """
        logger.info("Saving deduplicated dataframe...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create dataframe from deduplicated vectors
        deduplicated_df = pd.DataFrame({
            'comparison_label': self.data['labels'],
            'pre_brexit_sp_magnitude': self.data['pre_brexit_sp_magnitude'],
            'post_brexit_sp_magnitude': self.data['post_brexit_sp_magnitude'],
            'sp_magnitude_difference': self.data['sp_magnitude_difference'],
            'pre_brexit_sp_significance': self.data['pre_brexit_sp_significance'],
            'post_brexit_sp_significance': self.data['post_brexit_sp_significance'],
            'sp_gained_significance': self.data['sp_gained_significance'],
            'sp_lost_significance': self.data['sp_lost_significance'],
            'sp_both_significant': self.data['sp_both_significant'],
            'sp_neither_significant': self.data['sp_neither_significant']
        })
        
        # Save deduplicated dataframe
        csv_path = output_path / "deduplicated_fairness_comparisons.csv"
        deduplicated_df.to_csv(csv_path, index=False)
        
        logger.info(f"Deduplicated dataframe saved to {csv_path}")
        logger.info(f"Shape: {deduplicated_df.shape}")
        
        return deduplicated_df
    
    def run_complete_analysis(self, output_dir: str = None) -> Dict[str, Any]:
        """
        Run the complete bias vector drift analysis
        
        Args:
            output_dir: Directory to save outputs
            
        Returns:
            Complete analysis results
        """
        logger.info("Starting complete bias vector drift analysis...")
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Load vectors
        self.load_vectors()
        
        # Step 1b: Save deduplicated dataframe for other analyses
        if output_dir:
            self.save_deduplicated_dataframe(output_dir)
        
        # Step 2: Cosine similarity
        cosine_results = self.compute_cosine_similarity()
        
        # Step 3: Jaccard similarity  
        jaccard_results = self.compute_jaccard_similarity()
        
        # Step 4: Significance changes
        change_results = self.count_significance_changes()
        
        # Step 5: Visualization
        if output_dir:
            plot_path = output_path / "bias_magnitude_changes_histogram.png"
        else:
            plot_path = None
        
        fig = self.visualize_magnitude_changes(plot_path)
        
        # Optional: Most shifted pairs
        shifted_pairs = self.identify_most_shifted_pairs()
        
        # Save shifted pairs table
        if output_dir:
            shifted_pairs.to_csv(output_path / "most_shifted_bias_pairs.csv", index=False)
            logger.info(f"Most shifted pairs saved to {output_path / 'most_shifted_bias_pairs.csv'}")
        
        # Compile complete results
        complete_results = {
            'metadata': {
                'analysis_date': pd.Timestamp.now().isoformat(),
                'total_comparisons': len(self.data['pre_brexit_sp_magnitude']),
                'data_source': self.sp_vectors_path
            },
            'cosine_similarity': cosine_results,
            'jaccard_similarity': jaccard_results,
            'significance_changes': change_results,
            'magnitude_visualization': self.results['magnitude_visualization'],
            'most_shifted_pairs_summary': {
                'top_10_mean_absolute_difference': shifted_pairs.head(10)['Absolute_Difference'].mean(),
                'largest_single_shift': shifted_pairs.iloc[0]['Magnitude_Difference'],
                'largest_shift_comparison': shifted_pairs.iloc[0]['Comparison']
            }
        }
        
        # Save complete results
        if output_dir:
            # Create a clean copy without circular references
            clean_results = {}
            for key, value in complete_results.items():
                if key != 'magnitude_visualization':  # Skip the matplotlib figure reference
                    clean_results[key] = value
                else:
                    clean_results[key] = {
                        'figure_created': value['figure_created'],
                        'save_path': str(value['save_path']) if value['save_path'] else None,
                        'large_changes_count': value['large_changes_count'],
                        'large_changes_percentage': value['large_changes_percentage']
                    }
            
            with open(output_path / "bias_vector_drift_analysis_results.json", 'w') as f:
                # Convert numpy types for JSON serialization
                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj
                
                json.dump(clean_results, f, indent=2, default=convert_numpy)
            
            logger.info(f"Complete results saved to {output_path / 'bias_vector_drift_analysis_results.json'}")
        
        self.results.update(complete_results)
        
        # Print summary
        self._print_summary()
        
        return complete_results
    
    def _print_summary(self):
        """Print a summary of key findings"""
        print("\n" + "="*80)
        print("BIAS VECTOR DRIFT ANALYSIS SUMMARY")
        print("="*80)
        
        # Print deduplication statistics
        if 'dataframe_info' in self.data:
            info = self.data['dataframe_info']
            if 'original_total_comparisons' in info:
                print(f"\n🔧 DATA DEDUPLICATION")
                print(f"   Original Comparisons: {info['original_total_comparisons']}")
                print(f"   After Removing Bidirectional Redundancy: {info['filtered_total_comparisons']}")
                print(f"   Redundant Pairs Removed: {info['redundancy_removed']}")
                print(f"   Efficiency Gain: {(info['redundancy_removed']/info['original_total_comparisons']*100):.1f}% reduction")
        
        cosine = self.results['cosine_similarity']
        jaccard = self.results['jaccard_similarity']
        changes = self.results['significance_changes']
        
        print(f"\n📊 OVERALL BIAS STATE SHIFT")
        print(f"   Cosine Similarity (full): {cosine['full_vector']:.4f}")
        print(f"   Cosine Similarity (significant only): {cosine['significance_filtered']:.4f}")
        print(f"   Interpretation: {'High similarity' if cosine['full_vector'] > 0.7 else 'Moderate similarity' if cosine['full_vector'] > 0.4 else 'Low similarity'}")
        
        print(f"\n🎯 SIGNIFICANCE OVERLAP") 
        print(f"   Jaccard Similarity: {jaccard['jaccard_similarity']:.4f}")
        print(f"   Shared Significant Biases: {jaccard['intersection_count']}")
        print(f"   Pre-Brexit Significant: {jaccard['pre_significant_count']}")
        print(f"   Post-Brexit Significant: {jaccard['post_significant_count']}")
        
        print(f"\n🔄 FAIRNESS DISPARITY EVOLUTION")
        print(f"   Newly Emerged Disparities: {changes['gained_significance']['count']} ({changes['gained_significance']['percentage']:.1f}%)")
        print(f"   Disappeared Disparities: {changes['lost_significance']['count']} ({changes['lost_significance']['percentage']:.1f}%)")  
        print(f"   Persistent Disparities: {changes['both_significant']['count']} ({changes['both_significant']['percentage']:.1f}%)")
        print(f"   Consistently Non-significant: {changes['neither_significant']['count']} ({changes['neither_significant']['percentage']:.1f}%)")
        
        print(f"\n📈 MAGNITUDE CHANGES")
        mag_viz = self.results['magnitude_visualization']
        print(f"   Large Changes (|diff| > 0.2): {mag_viz['large_changes_count']} ({mag_viz['large_changes_percentage']:.1f}%)")
        
        if 'most_shifted_pairs_summary' in self.results:
            shifted = self.results['most_shifted_pairs_summary']
            print(f"   Largest Single Shift: {shifted['largest_single_shift']:.4f}")
            print(f"   Most Shifted Comparison: {shifted['largest_shift_comparison']}")
        
        print("\n" + "="*80)


def main():
    """Main execution function"""
    
    # Set paths
    sp_vectors_path = "../outputs/fairness_divergence/sp_vectors_data.json"
    output_dir = "../outputs/bias_vector_drift"
    
    # Initialize analyzer
    analyzer = BiasVectorDriftAnalyzer(sp_vectors_path)
    
    # Run complete analysis
    results = analyzer.run_complete_analysis(output_dir)
    
    print("\n✅ Bias Vector Drift Analysis completed successfully!")
    print(f"📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main() 