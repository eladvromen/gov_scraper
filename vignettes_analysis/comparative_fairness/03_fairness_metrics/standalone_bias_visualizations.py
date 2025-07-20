#!/usr/bin/env python3
"""
Standalone Bias Visualizations
==============================

Creates specific standalone plots for bias analysis:
1. Significant bias types by protected attributes
2. Topic-level fairness reconfiguration (stacked bar plot)
3. Topic-level fairness salience by model (dual bar plots)

Author: Fairness Analysis Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import textwrap

def wrap_topic_labels(topic_list):
    """
    Split long topic names into 2 lines
    """
    wrapped_labels = []
    for topic in topic_list:
        if len(topic) > 25:  # Split long topics
            words = topic.split()
            mid = len(words) // 2
            line1 = ' '.join(words[:mid])
            line2 = ' '.join(words[mid:])
            wrapped_labels.append(line1 + '\n' + line2)
        else:
            wrapped_labels.append(topic)
    return wrapped_labels

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

def load_and_prepare_data():
    """Load and prepare the significant bias data"""
    
    print("📂 Loading bias analysis data...")
    
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
            'persistent': row['sp_both_significant'],
            'pre_brexit_significant': row['pre_brexit_sp_significance'],
            'post_brexit_significant': row['post_brexit_sp_significance']
        })
    
    analysis_df = pd.DataFrame(parsed_data)
    
    # Also get all comparisons for model-level analysis
    all_parsed_data = []
    for _, row in df.iterrows():
        parsed = parse_comparison_label(row['comparison_label'])
        all_parsed_data.append({
            'comparison_label': row['comparison_label'],
            'protected_attribute': parsed['protected_attribute'],
            'topic': parsed['topic'],
            'pre_brexit_significant': row['pre_brexit_sp_significance'],
            'post_brexit_significant': row['post_brexit_sp_significance']
        })
    
    all_df = pd.DataFrame(all_parsed_data)
    
    return analysis_df, all_df

def plot_1_protected_attribute_significance_types(analysis_df):
    """
    Plot 1: Significant bias types by protected attributes (Centered/Diverging)
    """
    
    print("🎨 Creating Plot 1: Protected Attribute Significance Types (Centered)...")
    
    # Prepare data for centered stacked bar plot
    attr_breakdown = {}
    
    for attr in analysis_df['protected_attribute'].unique():
        attr_subset = analysis_df[analysis_df['protected_attribute'] == attr]
        attr_breakdown[attr] = {
            'Disappeared': attr_subset['disappeared'].sum(),
            'Persistent': attr_subset['persistent'].sum(),
            'Newly Emerged': attr_subset['newly_emerged'].sum()
        }
    
    # Convert to DataFrame
    breakdown_df = pd.DataFrame(attr_breakdown).T
    
    # Simple centered approach
    persistent_half = breakdown_df['Persistent'] / 2
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color scheme (flipped design with original colors)
    colors = {
        'Disappeared': '#FFA94D',   # Original orange (now on right)
        'Persistent': '#4D96FF',    # Original blue (stays center)
        'Newly Emerged': '#FF6B6B'  # Original coral red (now on left)
    }
    
    # Create horizontal bars
    y_pos = range(len(breakdown_df.index))
    
    # Far left: Newly Emerged (yellow) - extending leftward from left edge of persistent
    ax.barh(y_pos, -breakdown_df['Newly Emerged'], left=-persistent_half, 
            color=colors['Newly Emerged'], alpha=0.8, label='Newly Emerged', height=0.6)
    
    # Left half of Persistent (blue) - extending leftward from zero
    ax.barh(y_pos, -persistent_half, 
            color=colors['Persistent'], alpha=0.8, height=0.6)
    
    # Right half of Persistent (blue) - extending rightward from zero
    ax.barh(y_pos, persistent_half, 
            color=colors['Persistent'], alpha=0.8, label='Persistent', height=0.6)
    
    # Far right: Disappeared (red) - extending rightward from right edge of persistent
    ax.barh(y_pos, breakdown_df['Disappeared'], left=persistent_half,
            color=colors['Disappeared'], alpha=0.8, label='Disappeared', height=0.6)
    
    # Customize plot
    ax.set_title('Bias Significance Changes by Protected Attribute\n(Centered View)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('← Newly Emerged    |    Persistent + Disappeared →', fontsize=12, fontweight='bold')
    ax.set_ylabel('Protected Attributes', fontsize=12, fontweight='bold')
    
    # Set y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(breakdown_df.index)
    
    # Add simple value labels
    for i, attr in enumerate(breakdown_df.index):
        # Newly Emerged label (far left coral red bar)
        if breakdown_df.loc[attr, 'Newly Emerged'] > 0:
            center_pos = -persistent_half.iloc[i] - breakdown_df.loc[attr, 'Newly Emerged']/2
            ax.text(center_pos, i, str(int(breakdown_df.loc[attr, 'Newly Emerged'])), 
                   ha='center', va='center', fontweight='bold', fontsize=10, color='white')
        
        # Persistent label (center at x=0)
        if breakdown_df.loc[attr, 'Persistent'] > 0:
            ax.text(0, i, str(int(breakdown_df.loc[attr, 'Persistent'])), 
                   ha='center', va='center', fontweight='bold', fontsize=10, color='white')
        
        # Disappeared label (far right red bar)
        if breakdown_df.loc[attr, 'Disappeared'] > 0:
            center_pos = persistent_half.iloc[i] + breakdown_df.loc[attr, 'Disappeared']/2
            ax.text(center_pos, i, str(int(breakdown_df.loc[attr, 'Disappeared'])), 
                   ha='center', va='center', fontweight='bold', fontsize=10, color='white')
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
    
    # Customize legend
    ax.legend(title='Significance Change Type', title_fontsize=12, 
              fontsize=11, loc='upper right')
    
    # Style improvements
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig("../outputs/bias_vector_drift/plot1_protected_attribute_significance_types.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return breakdown_df

def plot_2_topic_fairness_reconfiguration(analysis_df):
    """
    Plot 2: Topic-Level Fairness Reconfiguration (Centered/Diverging)
    """
    
    print("🎨 Creating Plot 2: Topic-Level Fairness Reconfiguration (Centered)...")
    
    # Prepare data by topic
    topic_breakdown = {}
    
    for topic in analysis_df['topic'].unique():
        topic_subset = analysis_df[analysis_df['topic'] == topic]
        topic_breakdown[topic] = {
            'Disappeared': topic_subset['disappeared'].sum(),
            'Persistent': topic_subset['persistent'].sum(),
            'Newly Emerged': topic_subset['newly_emerged'].sum()
        }
    
    # Convert to DataFrame and sort by total significance
    breakdown_df = pd.DataFrame(topic_breakdown).T
    breakdown_df['Total'] = breakdown_df.sum(axis=1)
    breakdown_df = breakdown_df.sort_values('Total', ascending=True)
    breakdown_df = breakdown_df.drop('Total', axis=1)
    
    # Simple centered approach
    persistent_half = breakdown_df['Persistent'] / 2
    
    # Create the plot with extra space for 2-line labels
    fig, ax = plt.subplots(figsize=(18, 14))
    
    # Color scheme (flipped design with original colors)
    colors = {
        'Disappeared': '#FFA94D',   # Original orange (now on right)
        'Persistent': '#4D96FF',    # Original blue (stays center)
        'Newly Emerged': '#FF6B6B'  # Original coral red (now on left)
    }
    
    # Create horizontal bars
    y_pos = range(len(breakdown_df.index))
    
    # Far left: Newly Emerged (yellow) - extending leftward from left edge of persistent
    ax.barh(y_pos, -breakdown_df['Newly Emerged'], left=-persistent_half, 
            color=colors['Newly Emerged'], alpha=0.8, label='Newly Emerged', height=0.7)
    
    # Left half of Persistent (blue) - extending leftward from zero
    ax.barh(y_pos, -persistent_half, 
            color=colors['Persistent'], alpha=0.8, height=0.7)
    
    # Right half of Persistent (blue) - extending rightward from zero
    ax.barh(y_pos, persistent_half, 
            color=colors['Persistent'], alpha=0.8, label='Persistent', height=0.7)
    
    # Far right: Disappeared (red) - extending rightward from right edge of persistent
    ax.barh(y_pos, breakdown_df['Disappeared'], left=persistent_half,
            color=colors['Disappeared'], alpha=0.8, label='Disappeared', height=0.7)
    
    # Customize plot
    ax.set_title('Bias Significance Drift Across Topics\n(Centered View)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('← Newly Emerged    |    Persistent + Disappeared →', fontsize=12, fontweight='bold')
    ax.set_ylabel('Asylum Topics', fontsize=12, fontweight='bold')
    
    # Set y-axis labels with simple 2-line wrapping, left-aligned
    ax.set_yticks(y_pos)
    wrapped_labels = wrap_topic_labels(breakdown_df.index.tolist())
    ax.set_yticklabels(wrapped_labels, fontsize=10, ha='left', va='center')
    
    # Add simple value labels
    for i, topic in enumerate(breakdown_df.index):
        # Newly Emerged label (far left coral red bar)
        if breakdown_df.loc[topic, 'Newly Emerged'] > 0:
            center_pos = -persistent_half.iloc[i] - breakdown_df.loc[topic, 'Newly Emerged']/2
            ax.text(center_pos, i, str(int(breakdown_df.loc[topic, 'Newly Emerged'])), 
                   ha='center', va='center', fontweight='bold', fontsize=9, color='white')
        
        # Persistent label (center at x=0)
        if breakdown_df.loc[topic, 'Persistent'] > 0:
            ax.text(0, i, str(int(breakdown_df.loc[topic, 'Persistent'])), 
                   ha='center', va='center', fontweight='bold', fontsize=9, color='white')
        
        # Disappeared label (far right red bar)
        if breakdown_df.loc[topic, 'Disappeared'] > 0:
            center_pos = persistent_half.iloc[i] + breakdown_df.loc[topic, 'Disappeared']/2
            ax.text(center_pos, i, str(int(breakdown_df.loc[topic, 'Disappeared'])), 
                   ha='center', va='center', fontweight='bold', fontsize=9, color='white')
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
    
    # Customize legend
    ax.legend(title='Significance Type', title_fontsize=12, 
              fontsize=11, loc='upper right')
    
    # Style improvements
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.35)  # Much more space for full y-labels  
    plt.savefig("../outputs/bias_vector_drift/plot2_topic_fairness_reconfiguration.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return breakdown_df

def plot_3_topic_model_salience(all_df):
    """
    Plot 3: Topic-Level Fairness Salience by Model (Dual Bar Plots)
    """
    
    print("🎨 Creating Plot 3: Topic-Level Fairness Salience by Model...")
    
    # Calculate significance counts by topic and model
    pre_brexit_counts = all_df[all_df['pre_brexit_significant']].groupby('topic').size()
    post_brexit_counts = all_df[all_df['post_brexit_significant']].groupby('topic').size()
    
    # Get all topics and fill missing values with 0
    all_topics = set(pre_brexit_counts.index) | set(post_brexit_counts.index)
    pre_brexit_counts = pre_brexit_counts.reindex(all_topics, fill_value=0)
    post_brexit_counts = post_brexit_counts.reindex(all_topics, fill_value=0)
    
    # Sort by total significance (descending)
    total_counts = pre_brexit_counts + post_brexit_counts
    sorted_topics = total_counts.sort_values(ascending=False).index
    
    # Create dual bar plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Colors
    pre_color = '#FFD43B'  # Gold
    post_color = '#38A3A5'  # Teal
    
    # Left plot: Pre-Brexit
    pre_sorted = pre_brexit_counts[sorted_topics]
    bars1 = ax1.barh(range(len(pre_sorted)), pre_sorted.values, color=pre_color, alpha=0.8)
    ax1.set_yticks(range(len(pre_sorted)))
    ax1.set_yticklabels(pre_sorted.index, fontsize=9)
    ax1.set_title('Top Topics with Significant Bias\n(Pre-Brexit Model)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Number of Significant Disparities', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_axisbelow(True)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars1, pre_sorted.values)):
        if value > 0:
            ax1.text(value + max(pre_sorted.values) * 0.01, i, str(int(value)), 
                    va='center', fontsize=9, fontweight='bold')
    
    # Right plot: Post-Brexit
    post_sorted = post_brexit_counts[sorted_topics]
    bars2 = ax2.barh(range(len(post_sorted)), post_sorted.values, color=post_color, alpha=0.8)
    ax2.set_yticks(range(len(post_sorted)))
    ax2.set_yticklabels(post_sorted.index, fontsize=9)
    ax2.set_title('Top Topics with Significant Bias\n(Post-Brexit Model)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Number of Significant Disparities', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_axisbelow(True)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars2, post_sorted.values)):
        if value > 0:
            ax2.text(value + max(post_sorted.values) * 0.01, i, str(int(value)), 
                    va='center', fontsize=9, fontweight='bold')
    
    # Ensure both plots have the same x-axis scale
    max_value = max(max(pre_sorted.values), max(post_sorted.values))
    ax1.set_xlim(0, max_value * 1.15)
    ax2.set_xlim(0, max_value * 1.15)
    
    plt.tight_layout()
    plt.savefig("../outputs/bias_vector_drift/plot3_topic_model_salience.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return pre_sorted, post_sorted

def create_summary_stats(analysis_df, all_df):
    """Create summary statistics for the plots"""
    
    print("\n📊 SUMMARY STATISTICS")
    print("=" * 50)
    
    # Plot 1 stats
    total_significant = len(analysis_df)
    attr_counts = analysis_df['protected_attribute'].value_counts()
    
    print("🏷️  Protected Attribute Breakdown:")
    for attr, count in attr_counts.items():
        percentage = (count / total_significant) * 100
        print(f"   {attr}: {count} comparisons ({percentage:.1f}%)")
    
    # Plot 2 stats
    topic_counts = analysis_df['topic'].value_counts()
    print(f"\n📋 Topic Analysis ({len(topic_counts)} unique topics):")
    print(f"   Most affected topic: {topic_counts.index[0]} ({topic_counts.iloc[0]} disparities)")
    print(f"   Average disparities per topic: {topic_counts.mean():.1f}")
    
    # Plot 3 stats
    pre_total = all_df['pre_brexit_significant'].sum()
    post_total = all_df['post_brexit_significant'].sum()
    print(f"\n🔄 Model Comparison:")
    print(f"   Pre-Brexit significant comparisons: {pre_total}")
    print(f"   Post-Brexit significant comparisons: {post_total}")
    print(f"   Change: {post_total - pre_total:+d} ({((post_total - pre_total) / pre_total * 100):+.1f}%)")

def main():
    """Main execution function"""
    
    print("🎨 STANDALONE BIAS VISUALIZATIONS")
    print("=" * 50)
    
    # Load data
    analysis_df, all_df = load_and_prepare_data()
    
    # Create plots
    print("\n📈 Generating standalone visualizations...")
    
    # Plot 1: Protected Attribute Significance Types
    breakdown_1 = plot_1_protected_attribute_significance_types(analysis_df)
    
    # Plot 2: Topic-Level Fairness Reconfiguration
    breakdown_2 = plot_2_topic_fairness_reconfiguration(analysis_df)
    
    # Plot 3: Topic-Level Fairness Salience by Model
    pre_counts, post_counts = plot_3_topic_model_salience(all_df)
    
    # Summary statistics
    create_summary_stats(analysis_df, all_df)
    
    print("\n✅ All visualizations created successfully!")
    print("📁 Saved to: ../outputs/bias_vector_drift/")
    print("   • plot1_protected_attribute_significance_types.png")
    print("   • plot2_topic_fairness_reconfiguration.png") 
    print("   • plot3_topic_model_salience.png")

if __name__ == "__main__":
    main() 