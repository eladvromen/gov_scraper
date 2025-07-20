#!/usr/bin/env python3
"""
Fairness Disparity Qualitative Analysis
=======================================

Provides detailed qualitative analysis of fairness disparity evolution between 
pre-Brexit and post-Brexit models, focusing on which protected groups and topics 
are involved in different types of significance changes.

Key Features:
- Breakdown of significance changes by protected attribute and topic
- Visualization of disparity patterns
- Identification of most volatile groups and topics
- Cross-tabulation analysis of attribute-topic combinations

Author: Fairness Disparity Analysis Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FairnessDisparityQualitativeAnalyzer:
    """
    Analyzes fairness disparity evolution patterns by protected groups and topics
    """
    
    def __init__(self, unified_df_path: str, deduplicated_df_path: str, bias_drift_results_path: str):
        """
        Initialize the analyzer
        
        Args:
            unified_df_path: Path to unified fairness dataframe (for metadata)
            deduplicated_df_path: Path to deduplicated fairness comparisons
            bias_drift_results_path: Path to bias drift analysis results
        """
        self.unified_df_path = unified_df_path
        self.deduplicated_df_path = deduplicated_df_path
        self.bias_drift_results_path = bias_drift_results_path
        self.df = None
        self.bias_results = None
        self.analysis_results = {}
        
    def load_data(self):
        """Load the deduplicated dataframe and enrich with metadata"""
        logger.info("Loading data for qualitative analysis...")
        
        # Load deduplicated fairness dataframe
        self.df = pd.read_csv(self.deduplicated_df_path)
        logger.info(f"Loaded {len(self.df)} deduplicated fairness comparisons")
        
        # Load original unified dataframe for metadata enrichment
        unified_df = pd.read_csv(self.unified_df_path)
        logger.info(f"Loaded {len(unified_df)} total comparisons from unified dataframe")
        
        # Enrich deduplicated dataframe with metadata by joining on comparison labels
        # Extract group comparison from comparison_label (remove protected attribute and topic info)
        if 'comparison_label' in self.df.columns and 'group_comparison' in unified_df.columns:
            # Extract just the group comparison part (before the first space-parenthesis)
            self.df['group_comparison'] = self.df['comparison_label'].apply(
                lambda x: x.split(' (')[0] if ' (' in x else x
            )
            
            # Join to get protected_attribute and topic
            metadata_cols = ['group_comparison', 'protected_attribute', 'topic']
            if all(col in unified_df.columns for col in metadata_cols):
                unified_subset = unified_df[metadata_cols].drop_duplicates()
                
                # Join deduplicated data with metadata
                self.df = self.df.merge(unified_subset, on='group_comparison', how='left')
                logger.info("Successfully enriched deduplicated data with protected attributes and topics")
                
                # Check for missing metadata
                missing_metadata = self.df[['protected_attribute', 'topic']].isnull().any(axis=1).sum()
                if missing_metadata > 0:
                    logger.warning(f"{missing_metadata} comparisons missing protected attribute or topic metadata")
                    
                # Log success stats
                successful_joins = (~self.df[['protected_attribute', 'topic']].isnull().any(axis=1)).sum()
                logger.info(f"Successfully matched {successful_joins}/{len(self.df)} comparisons with metadata")
            else:
                logger.warning("Could not find required metadata columns in unified dataframe")
        else:
            logger.warning("Could not find required columns for joining dataframes")
        
        # Load bias drift results if available
        if Path(self.bias_drift_results_path).exists():
            with open(self.bias_drift_results_path, 'r') as f:
                self.bias_results = json.load(f)
            logger.info("Loaded bias drift analysis results")
        
        # Create significance change categories
        self._create_significance_categories()
        
    def _create_significance_categories(self):
        """Create significance change category columns"""
        logger.info("Creating significance change categories...")
        
        # Use FDR-corrected significance if available, otherwise use original
        pre_sig_col = 'pre_brexit_sp_significance'
        post_sig_col = 'post_brexit_sp_significance'
        
        # Fill NaN values with False for significance columns
        self.df[pre_sig_col] = self.df[pre_sig_col].fillna(False)
        self.df[post_sig_col] = self.df[post_sig_col].fillna(False)
        
        # Create significance change categories
        self.df['gained_significance'] = ~self.df[pre_sig_col] & self.df[post_sig_col]
        self.df['lost_significance'] = self.df[pre_sig_col] & ~self.df[post_sig_col]
        self.df['both_significant'] = self.df[pre_sig_col] & self.df[post_sig_col]
        self.df['neither_significant'] = ~self.df[pre_sig_col] & ~self.df[post_sig_col]
        
        # Create a categorical column for easier analysis
        conditions = [
            self.df['gained_significance'],
            self.df['lost_significance'], 
            self.df['both_significant'],
            self.df['neither_significant']
        ]
        choices = [
            'Newly Emerged Disparities',
            'Disappeared Disparities',
            'Persistent Disparities', 
            'Consistently Non-significant'
        ]
        self.df['significance_category'] = np.select(conditions, choices, default='Unknown')
        
        # Log category counts
        category_counts = self.df['significance_category'].value_counts()
        logger.info("Significance change categories:")
        for category, count in category_counts.items():
            percentage = (count / len(self.df)) * 100
            logger.info(f"  {category}: {count} ({percentage:.1f}%)")
    
    def analyze_by_protected_attribute(self) -> Dict[str, pd.DataFrame]:
        """
        Analyze significance changes by protected attribute
        
        Returns:
            Dictionary mapping significance categories to breakdown by protected attribute
        """
        logger.info("Analyzing significance changes by protected attribute...")
        
        results = {}
        
        # Overall breakdown by protected attribute
        attr_breakdown = self.df.groupby(['protected_attribute', 'significance_category']).size().unstack(fill_value=0)
        
        # Add percentages
        attr_percentages = attr_breakdown.div(attr_breakdown.sum(axis=1), axis=0) * 100
        
        # Combine counts and percentages
        attr_analysis = pd.DataFrame()
        for col in attr_breakdown.columns:
            attr_analysis[f'{col}_count'] = attr_breakdown[col]
            attr_analysis[f'{col}_pct'] = attr_percentages[col]
        
        results['by_attribute'] = attr_analysis
        
        # Breakdown for each significance category
        categories = ['Newly Emerged Disparities', 'Disappeared Disparities', 
                     'Persistent Disparities', 'Consistently Non-significant']
        
        for category in categories:
            category_data = self.df[self.df['significance_category'] == category]
            if len(category_data) > 0:
                attr_counts = category_data['protected_attribute'].value_counts()
                attr_percentages = (attr_counts / len(category_data)) * 100
                
                category_df = pd.DataFrame({
                    'count': attr_counts,
                    'percentage': attr_percentages
                })
                results[category] = category_df
            else:
                results[category] = pd.DataFrame()
        
        self.analysis_results['by_protected_attribute'] = results
        return results
    
    def analyze_by_topic(self) -> Dict[str, pd.DataFrame]:
        """
        Analyze significance changes by vignette topic
        
        Returns:
            Dictionary mapping significance categories to breakdown by topic
        """
        logger.info("Analyzing significance changes by vignette topic...")
        
        results = {}
        
        # Overall breakdown by topic
        topic_breakdown = self.df.groupby(['topic', 'significance_category']).size().unstack(fill_value=0)
        
        # Add percentages
        topic_percentages = topic_breakdown.div(topic_breakdown.sum(axis=1), axis=0) * 100
        
        # Combine counts and percentages
        topic_analysis = pd.DataFrame()
        for col in topic_breakdown.columns:
            topic_analysis[f'{col}_count'] = topic_breakdown[col]
            topic_analysis[f'{col}_pct'] = topic_percentages[col]
        
        results['by_topic'] = topic_analysis
        
        # Breakdown for each significance category
        categories = ['Newly Emerged Disparities', 'Disappeared Disparities', 
                     'Persistent Disparities', 'Consistently Non-significant']
        
        for category in categories:
            category_data = self.df[self.df['significance_category'] == category]
            if len(category_data) > 0:
                topic_counts = category_data['topic'].value_counts()
                topic_percentages = (topic_counts / len(category_data)) * 100
                
                category_df = pd.DataFrame({
                    'count': topic_counts,
                    'percentage': topic_percentages
                })
                results[category] = category_df
            else:
                results[category] = pd.DataFrame()
        
        self.analysis_results['by_topic'] = results
        return results
    
    def analyze_cross_tabulation(self) -> Dict[str, pd.DataFrame]:
        """
        Analyze cross-tabulation of protected attributes and topics for each significance category
        
        Returns:
            Dictionary with cross-tabulation results for each category
        """
        logger.info("Creating cross-tabulation analysis...")
        
        results = {}
        categories = ['Newly Emerged Disparities', 'Disappeared Disparities', 
                     'Persistent Disparities', 'Consistently Non-significant']
        
        for category in categories:
            category_data = self.df[self.df['significance_category'] == category]
            if len(category_data) > 0:
                # Create cross-tabulation
                crosstab = pd.crosstab(category_data['protected_attribute'], 
                                     category_data['topic'], 
                                     margins=True)
                results[category] = crosstab
            else:
                results[category] = pd.DataFrame()
        
        self.analysis_results['cross_tabulation'] = results
        return results
    
    def create_visualizations(self, output_dir: str = None):
        """
        Create visualizations for the qualitative analysis
        
        Args:
            output_dir: Directory to save plots
        """
        logger.info("Creating visualizations...")
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style for better-looking plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Protected Attribute Breakdown
        self._plot_protected_attribute_breakdown(output_path)
        
        # 2. Topic Breakdown  
        self._plot_topic_breakdown(output_path)
        
        # 3. Heatmap of significance changes
        self._plot_significance_heatmap(output_path)
        
        # 4. Cross-tabulation heatmaps
        self._plot_crosstab_heatmaps(output_path)
    
    def _plot_protected_attribute_breakdown(self, output_path: Path):
        """Plot breakdown by protected attribute"""
        if 'by_protected_attribute' not in self.analysis_results:
            return
            
        results = self.analysis_results['by_protected_attribute']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        categories = ['Newly Emerged Disparities', 'Disappeared Disparities', 
                     'Persistent Disparities', 'Consistently Non-significant']
        colors = ['red', 'orange', 'blue', 'gray']
        
        for i, (category, color) in enumerate(zip(categories, colors)):
            ax = [ax1, ax2, ax3, ax4][i]
            
            if category in results and not results[category].empty:
                data = results[category].sort_values('count', ascending=True)
                
                bars = ax.barh(data.index, data['count'], color=color, alpha=0.7)
                ax.set_title(f'{category}\n({data["count"].sum()} total comparisons)')
                ax.set_xlabel('Number of Comparisons')
                
                # Add percentage labels
                for j, (idx, row) in enumerate(data.iterrows()):
                    ax.text(row['count'] + 0.5, j, f'{row["percentage"]:.1f}%', 
                           va='center', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{category}\n(0 total comparisons)')
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path / "protected_attribute_breakdown.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_topic_breakdown(self, output_path: Path):
        """Plot breakdown by topic"""
        if 'by_topic' not in self.analysis_results:
            return
            
        results = self.analysis_results['by_topic']
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.ravel()
        
        categories = ['Newly Emerged Disparities', 'Disappeared Disparities', 
                     'Persistent Disparities', 'Consistently Non-significant']
        colors = ['red', 'orange', 'blue', 'gray']
        
        for i, (category, color) in enumerate(zip(categories, colors)):
            ax = axes[i]
            
            if category in results and not results[category].empty:
                data = results[category].sort_values('count', ascending=True)
                
                # Show top 10 topics for readability
                if len(data) > 10:
                    data = data.tail(10)
                
                bars = ax.barh(range(len(data)), data['count'], color=color, alpha=0.7)
                ax.set_yticks(range(len(data)))
                ax.set_yticklabels([label[:30] + '...' if len(label) > 30 else label 
                                   for label in data.index], fontsize=8)
                ax.set_title(f'{category}\n(Top topics, {data["count"].sum()} total comparisons)')
                ax.set_xlabel('Number of Comparisons')
                
                # Add percentage labels
                for j, count in enumerate(data['count']):
                    ax.text(count + 0.2, j, f'{data.iloc[j]["percentage"]:.1f}%', 
                           va='center', fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{category}\n(0 total comparisons)')
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path / "topic_breakdown.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_significance_heatmap(self, output_path: Path):
        """Plot heatmap of significance changes by attribute and topic"""
        # Create a summary heatmap showing change patterns
        summary_data = self.df.groupby(['protected_attribute', 'significance_category']).size().unstack(fill_value=0)
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(summary_data, annot=True, fmt='d', cmap='YlOrRd', 
                   cbar_kws={'label': 'Number of Comparisons'})
        plt.title('Significance Changes by Protected Attribute')
        plt.xlabel('Significance Change Category')
        plt.ylabel('Protected Attribute')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path / "significance_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_crosstab_heatmaps(self, output_path: Path):
        """Plot cross-tabulation heatmaps for each significance category"""
        if 'cross_tabulation' not in self.analysis_results:
            return
            
        results = self.analysis_results['cross_tabulation']
        
        categories = ['Newly Emerged Disparities', 'Disappeared Disparities', 
                     'Persistent Disparities']
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        for i, category in enumerate(categories):
            ax = axes[i]
            
            if category in results and not results[category].empty:
                # Remove margins for cleaner visualization
                data = results[category].drop('All', axis=0, errors='ignore').drop('All', axis=1, errors='ignore')
                
                if not data.empty:
                    sns.heatmap(data, annot=True, fmt='d', cmap='Blues', ax=ax,
                               cbar_kws={'label': 'Count'})
                    ax.set_title(f'{category}')
                    ax.set_xlabel('Topic')
                    ax.set_ylabel('Protected Attribute')
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{category}')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{category}')
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path / "crosstab_heatmaps.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self) -> str:
        """
        Generate a comprehensive summary report of findings
        
        Returns:
            Formatted summary report string
        """
        report = []
        report.append("=" * 80)
        report.append("FAIRNESS DISPARITY QUALITATIVE ANALYSIS SUMMARY")
        report.append("=" * 80)
        
        # Overall statistics
        total_comparisons = len(self.df)
        category_counts = self.df['significance_category'].value_counts()
        
        report.append(f"\n📊 OVERALL STATISTICS")
        report.append(f"   Total Unique Comparisons: {total_comparisons}")
        for category, count in category_counts.items():
            percentage = (count / total_comparisons) * 100
            report.append(f"   {category}: {count} ({percentage:.1f}%)")
        
        # Protected attribute insights
        if 'by_protected_attribute' in self.analysis_results:
            report.append(f"\n🏷️  PROTECTED ATTRIBUTE INSIGHTS")
            
            attr_results = self.analysis_results['by_protected_attribute']
            
            # Most volatile attributes (highest proportion of changes)
            for category in ['Newly Emerged Disparities', 'Disappeared Disparities']:
                if category in attr_results and not attr_results[category].empty:
                    top_attr = attr_results[category].sort_values('percentage', ascending=False).index[0]
                    top_pct = attr_results[category].loc[top_attr, 'percentage']
                    top_count = attr_results[category].loc[top_attr, 'count']
                    report.append(f"   {category}:")
                    report.append(f"     Most affected: {top_attr} ({top_count} comparisons, {top_pct:.1f}%)")
        
        # Topic insights
        if 'by_topic' in self.analysis_results:
            report.append(f"\n📋 TOPIC INSIGHTS")
            
            topic_results = self.analysis_results['by_topic']
            
            # Most volatile topics
            for category in ['Newly Emerged Disparities', 'Disappeared Disparities']:
                if category in topic_results and not topic_results[category].empty:
                    top_topic = topic_results[category].sort_values('count', ascending=False).index[0]
                    top_count = topic_results[category].loc[top_topic, 'count']
                    top_pct = topic_results[category].loc[top_topic, 'percentage']
                    report.append(f"   {category}:")
                    report.append(f"     Most affected: {top_topic}")
                    report.append(f"     Count: {top_count} ({top_pct:.1f}%)")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def run_complete_analysis(self, output_dir: str = None) -> Dict[str, Any]:
        """
        Run the complete qualitative analysis
        
        Args:
            output_dir: Directory to save outputs
            
        Returns:
            Complete analysis results
        """
        logger.info("Starting complete fairness disparity qualitative analysis...")
        
        # Load data
        self.load_data()
        
        # Run analyses
        attr_results = self.analyze_by_protected_attribute()
        topic_results = self.analyze_by_topic()
        crosstab_results = self.analyze_cross_tabulation()
        
        # Create visualizations
        if output_dir:
            self.create_visualizations(output_dir)
        
        # Generate summary report
        summary_report = self.generate_summary_report()
        
        # Save results
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save detailed results
            with open(output_path / "qualitative_analysis_results.json", 'w') as f:
                # Convert numpy types for JSON serialization
                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, pd.DataFrame):
                        return obj.to_dict()
                    return obj
                
                json.dump(self.analysis_results, f, indent=2, default=convert_numpy)
            
            # Save summary report
            with open(output_path / "qualitative_analysis_summary.txt", 'w') as f:
                f.write(summary_report)
            
            logger.info(f"Results saved to {output_dir}")
        
        # Print summary
        print(summary_report)
        
        return self.analysis_results


def main():
    """Main execution function"""
    
    # Set paths
    unified_df_path = "../outputs/unified_analysis/unified_fairness_dataframe_topic_granular.csv"
    deduplicated_df_path = "../outputs/bias_vector_drift/deduplicated_fairness_comparisons.csv"
    bias_drift_results_path = "../outputs/bias_vector_drift/bias_vector_drift_analysis_results.json"
    output_dir = "../outputs/fairness_disparity_qualitative"
    
    # Initialize analyzer
    analyzer = FairnessDisparityQualitativeAnalyzer(unified_df_path, deduplicated_df_path, bias_drift_results_path)
    
    # Run complete analysis
    results = analyzer.run_complete_analysis(output_dir)
    
    print("\n✅ Fairness Disparity Qualitative Analysis completed successfully!")
    print(f"📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main() 