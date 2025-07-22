#!/usr/bin/env python3
"""
Significant Bias Subset Analysis
===============================

Simple analysis of the subset of significant bias comparisons to show:
- What protected groups are involved in significant bias changes
- What topics are involved in significant bias changes
- Counts for each category

Author: Fairness Analysis Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

def parse_comparison_label(label):
    """
    Parse a comparison label to extract protected attribute and topic
    
    Example: "Syria_vs_Myanmar (country) [Asylum seeker circumstances]"
    Returns: {'protected_attribute': 'country', 'topic': 'Asylum seeker circumstances'}
    """
    # Extract protected attribute (in parentheses)
    attr_match = re.search(r'\(([^)]+)\)', label)
    protected_attribute = attr_match.group(1) if attr_match else 'unknown'
    
    # Extract topic (in square brackets)
    topic_match = re.search(r'\[([^\]]+)\]', label)
    topic = topic_match.group(1) if topic_match else 'unknown'
    
    return {
        'protected_attribute': protected_attribute,
        'topic': topic
    }

def analyze_significant_biases():
    """Analyze the subset of significant bias comparisons"""
    
    print("🔍 ANALYZING SIGNIFICANT BIAS SUBSET")
    print("=" * 60)
    
    # Load deduplicated data
    df = pd.read_csv("../outputs/bias_vector_drift/deduplicated_fairness_comparisons.csv")
    print(f"📊 Total deduplicated comparisons: {len(df)}")
    
    # Filter to only significant comparisons
    significant_mask = (
        df['sp_gained_significance'] | 
        df['sp_lost_significance'] | 
        df['sp_both_significant']
    )
    
    significant_df = df[significant_mask].copy()
    print(f"📊 Significant comparisons: {len(significant_df)}")
    
    # Break down by significance type
    gained = df['sp_gained_significance'].sum()
    lost = df['sp_lost_significance'].sum() 
    persistent = df['sp_both_significant'].sum()
    
    print(f"   • Newly Emerged Disparities: {gained}")
    print(f"   • Disappeared Disparities: {lost}")
    print(f"   • Persistent Disparities: {persistent}")
    print()
    
    # Parse labels to extract metadata
    parsed_data = []
    for _, row in significant_df.iterrows():
        parsed = parse_comparison_label(row['comparison_label'])
        parsed_data.append({
            'comparison_label': row['comparison_label'],
            'protected_attribute': parsed['protected_attribute'],
            'topic': parsed['topic'],
            'newly_emerged': row['sp_gained_significance'],
            'disappeared': row['sp_lost_significance'],
            'persistent': row['sp_both_significant']
        })
    
    analysis_df = pd.DataFrame(parsed_data)
    
    # PROTECTED ATTRIBUTE ANALYSIS
    print("🏷️  PROTECTED ATTRIBUTE BREAKDOWN")
    print("-" * 40)
    
    attr_counts = analysis_df['protected_attribute'].value_counts()
    total_significant = len(analysis_df)
    
    for attr, count in attr_counts.items():
        percentage = (count / total_significant) * 100
        print(f"   {attr}: {count} comparisons ({percentage:.1f}%)")
    
    print()
    
    # Break down by significance type for each attribute
    print("📈 BY SIGNIFICANCE TYPE:")
    for attr in attr_counts.index:
        attr_subset = analysis_df[analysis_df['protected_attribute'] == attr]
        newly_emerged = attr_subset['newly_emerged'].sum()
        disappeared = attr_subset['disappeared'].sum()
        persistent = attr_subset['persistent'].sum()
        
        print(f"   {attr}:")
        print(f"     Newly Emerged: {newly_emerged}")
        print(f"     Disappeared: {disappeared}")
        print(f"     Persistent: {persistent}")
    
    print()
    
    # TOPIC ANALYSIS
    print("📋 TOPIC BREAKDOWN")
    print("-" * 40)
    
    topic_counts = analysis_df['topic'].value_counts()
    
    for topic, count in topic_counts.items():
        percentage = (count / total_significant) * 100
        print(f"   {topic}: {count} comparisons ({percentage:.1f}%)")
    
    print()
    
    # Break down by significance type for each topic (top 10)
    print("📈 TOP 10 TOPICS BY SIGNIFICANCE TYPE:")
    for topic in topic_counts.head(10).index:
        topic_subset = analysis_df[analysis_df['topic'] == topic]
        newly_emerged = topic_subset['newly_emerged'].sum()
        disappeared = topic_subset['disappeared'].sum()
        persistent = topic_subset['persistent'].sum()
        
        print(f"   {topic}:")
        print(f"     Newly Emerged: {newly_emerged}")
        print(f"     Disappeared: {disappeared}")
        print(f"     Persistent: {persistent}")
    
    print()
    
    # CROSS-TABULATION
    print("📊 CROSS-TABULATION: PROTECTED ATTRIBUTE vs TOPIC")
    print("-" * 60)
    
    crosstab = pd.crosstab(analysis_df['protected_attribute'], analysis_df['topic'])
    print(crosstab)
    
    print()
    print("✅ Analysis complete!")
    
    return analysis_df, attr_counts, topic_counts, crosstab

def create_heatmap_visualizations(analysis_df, attr_counts, topic_counts):
    """Create beautiful heatmap visualizations of the significant bias subset"""
    
    # Set style for better looking plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Main Cross-tabulation Heatmap (Protected Attribute vs Topic)
    ax1 = plt.subplot(2, 3, (1, 2))  # Span 2 columns
    
    # Create cross-tabulation for all comparisons
    crosstab_all = pd.crosstab(analysis_df['protected_attribute'], analysis_df['topic'])
    
    # Only show topics with at least 3 comparisons for readability
    topic_filter = topic_counts >= 3
    filtered_topics = topic_counts[topic_filter].index
    crosstab_filtered = crosstab_all[filtered_topics]
    
    # Create heatmap
    sns.heatmap(crosstab_filtered, 
                annot=True, 
                fmt='d', 
                cmap='YlOrRd',
                cbar_kws={'label': 'Number of Significant Comparisons'},
                ax=ax1)
    ax1.set_title('Significant Bias Patterns: Protected Attributes vs Topics\n(Topics with ≥3 comparisons)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Topics', fontweight='bold')
    ax1.set_ylabel('Protected Attributes', fontweight='bold')
    
    # Rotate x-axis labels for better readability
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 2. Significance Type by Attribute Heatmap
    ax2 = plt.subplot(2, 3, 3)
    
    # Create significance type matrix
    significance_matrix = []
    sig_types = ['Newly Emerged', 'Disappeared', 'Persistent']
    
    for attr in attr_counts.index:
        attr_subset = analysis_df[analysis_df['protected_attribute'] == attr]
        row = [
            attr_subset['newly_emerged'].sum(),
            attr_subset['disappeared'].sum(), 
            attr_subset['persistent'].sum()
        ]
        significance_matrix.append(row)
    
    sig_df = pd.DataFrame(significance_matrix, 
                         index=attr_counts.index, 
                         columns=sig_types)
    
    sns.heatmap(sig_df, 
                annot=True, 
                fmt='d', 
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Count'},
                ax=ax2)
    ax2.set_title('Significance Types by\nProtected Attribute', fontweight='bold')
    ax2.set_xlabel('Significance Type', fontweight='bold')
    ax2.set_ylabel('')
    
    # 3. Protected Attribute Bar Chart
    ax3 = plt.subplot(2, 3, 4)
    
    bars = ax3.bar(attr_counts.index, attr_counts.values, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
    ax3.set_title('Total Significant Biases\nby Protected Attribute', fontweight='bold')
    ax3.set_xlabel('Protected Attribute', fontweight='bold')
    ax3.set_ylabel('Count', fontweight='bold')
    
    # Add count labels on bars
    for bar, count in zip(bars, attr_counts.values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Top Topics Horizontal Bar Chart
    ax4 = plt.subplot(2, 3, 5)
    
    top_topics = topic_counts.head(8)  # Show top 8 for better fit
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_topics)))
    
    bars = ax4.barh(range(len(top_topics)), top_topics.values, color=colors, alpha=0.8)
    ax4.set_yticks(range(len(top_topics)))
    ax4.set_yticklabels([label[:25] + '...' if len(label) > 25 else label 
                        for label in top_topics.index])
    ax4.set_title('Top Topics with\nSignificant Biases', fontweight='bold')
    ax4.set_xlabel('Count', fontweight='bold')
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, top_topics.values)):
        ax4.text(count + 0.2, i, f'{count}', va='center', fontweight='bold')
    
    # 5. Pie Chart of Significance Types
    ax5 = plt.subplot(2, 3, 6)
    
    total_newly_emerged = analysis_df['newly_emerged'].sum()
    total_disappeared = analysis_df['disappeared'].sum()
    total_persistent = analysis_df['persistent'].sum()
    
    sizes = [total_newly_emerged, total_disappeared, total_persistent]
    labels = ['Newly Emerged\n(Post-Brexit)', 'Disappeared\n(Pre-Brexit)', 'Persistent\n(Both Models)']
    colors = ['#FF6B6B', '#FFA726', '#42A5F5']
    explode = (0.05, 0.05, 0.05)  # Slightly separate all slices
    
    wedges, texts, autotexts = ax5.pie(sizes, labels=labels, colors=colors, 
                                      autopct='%1.1f%%', explode=explode,
                                      startangle=90, textprops={'fontweight': 'bold'})
    ax5.set_title('Distribution of\nSignificance Types', fontweight='bold')
    
    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
    
    plt.tight_layout()
    plt.savefig("../outputs/bias_vector_drift/significant_bias_heatmap_analysis.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a separate detailed heatmap for significance types across topics
    create_detailed_significance_heatmap(analysis_df, topic_counts)

def create_detailed_significance_heatmap(analysis_df, topic_counts):
    """Create detailed heatmap showing significance types across topics"""
    
    # Focus on top topics for readability
    top_topics = topic_counts.head(12).index
    
    # Create matrices for each significance type
    significance_data = []
    
    for topic in top_topics:
        topic_subset = analysis_df[analysis_df['topic'] == topic]
        
        # Count by protected attribute and significance type
        topic_data = {}
        for attr in ['country', 'age', 'gender', 'religion']:
            attr_topic_subset = topic_subset[topic_subset['protected_attribute'] == attr]
            topic_data[attr] = {
                'newly_emerged': attr_topic_subset['newly_emerged'].sum(),
                'disappeared': attr_topic_subset['disappeared'].sum(),
                'persistent': attr_topic_subset['persistent'].sum()
            }
        significance_data.append((topic, topic_data))
    
    # Create separate heatmaps for each significance type
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
    attributes = ['country', 'age', 'gender', 'religion']
    significance_types = ['newly_emerged', 'disappeared', 'persistent']
    titles = ['Newly Emerged Disparities', 'Disappeared Disparities', 'Persistent Disparities']
    cmaps = ['Reds', 'Oranges', 'Blues']
    
    for i, (sig_type, title, cmap, ax) in enumerate(zip(significance_types, titles, cmaps, [ax1, ax2, ax3])):
        # Build matrix for this significance type
        matrix = []
        for attr in attributes:
            row = []
            for topic, topic_data in significance_data:
                row.append(topic_data[attr][sig_type])
            matrix.append(row)
        
        sig_matrix = pd.DataFrame(matrix, 
                                 index=attributes, 
                                 columns=[topic[:20] + '...' if len(topic) > 20 else topic 
                                         for topic, _ in significance_data])
        
        sns.heatmap(sig_matrix, 
                    annot=True, 
                    fmt='d', 
                    cmap=cmap,
                    cbar_kws={'label': 'Count'},
                    ax=ax)
        ax.set_title(f'{title}\nAcross Top Topics', fontweight='bold')
        ax.set_xlabel('Topics', fontweight='bold')
        if i == 0:
            ax.set_ylabel('Protected Attributes', fontweight='bold')
        else:
            ax.set_ylabel('')
        
        # Rotate labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig("../outputs/bias_vector_drift/significance_types_detailed_heatmap.png", 
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution"""
    analysis_df, attr_counts, topic_counts, crosstab = analyze_significant_biases()
    create_heatmap_visualizations(analysis_df, attr_counts, topic_counts)

if __name__ == "__main__":
    main() 