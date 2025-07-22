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
import matplotlib.patches as mpatches

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

def create_attribute_color_palette():
    """
    Create color palette with 4 shades for each category
    Each protected attribute gets a different shade within the category color
    """
    # Base colors
    base_colors = {
        'Newly Emerged': '#FF6B6B',  # Red
        'Persistent': '#4D96FF',     # Blue  
        'Disappeared': '#FFA94D'     # Orange
    }
    
    # Attribute order for consistent shading
    attributes = ['gender', 'religion', 'age', 'country']
    
    # Create shaded versions (darkest to lightest)
    shade_factors = [0.4, 0.6, 0.8, 1.0]  # gender=darkest, country=lightest
    
    color_palette = {}
    
    for category, base_color in base_colors.items():
        # Convert hex to RGB
        hex_color = base_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        for i, attribute in enumerate(attributes):
            # Apply shade factor
            shaded_rgb = tuple(int(c * shade_factors[i]) for c in rgb)
            # Convert back to hex
            shaded_hex = '#%02x%02x%02x' % shaded_rgb
            color_palette[f'{category}_{attribute}'] = shaded_hex
    
    return color_palette, attributes

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

def plot_enhanced_topic_breakdown_by_attribute(analysis_df):
    """
    Enhanced Plot: Topic-Level Bias Breakdown by Protected Attribute
    Each topic bar is subdivided by protected attribute with different color shades
    """
    
    print("🎨 Creating Enhanced Plot: Topic Breakdown by Protected Attribute...")
    
    # Get color palette
    color_palette, attributes = create_attribute_color_palette()
    
    # Prepare data by topic AND protected attribute
    topic_attr_breakdown = {}
    
    for topic in analysis_df['topic'].unique():
        topic_subset = analysis_df[analysis_df['topic'] == topic]
        
        # Initialize topic breakdown
        topic_attr_breakdown[topic] = {}
        
        for category in ['Newly Emerged', 'Persistent', 'Disappeared']:
            topic_attr_breakdown[topic][category] = {}
            
            for attr in attributes:
                # Filter by category and attribute
                if category == 'Newly Emerged':
                    mask = topic_subset['newly_emerged'] & (topic_subset['protected_attribute'] == attr)
                elif category == 'Persistent': 
                    mask = topic_subset['persistent'] & (topic_subset['protected_attribute'] == attr)
                else:  # Disappeared
                    mask = topic_subset['disappeared'] & (topic_subset['protected_attribute'] == attr)
                
                count = mask.sum()
                topic_attr_breakdown[topic][category][attr] = count
    
    # Convert to organized DataFrame for plotting
    plot_data = []
    for topic, categories in topic_attr_breakdown.items():
        row = {'topic': topic}
        total = 0
        
        for category in ['Newly Emerged', 'Persistent', 'Disappeared']:
            for attr in attributes:
                key = f'{category}_{attr}'
                value = categories[category][attr]
                row[key] = value
                total += value
        
        row['total'] = total
        plot_data.append(row)
    
    # Convert to DataFrame and sort by total
    plot_df = pd.DataFrame(plot_data)
    plot_df = plot_df.sort_values('total', ascending=True)
    
    # Calculate totals for each category per topic
    topic_totals = {}
    for _, row in plot_df.iterrows():
        topic = row['topic']
        newly_emerged_total = sum(row[f'Newly Emerged_{attr}'] for attr in attributes)
        persistent_total = sum(row[f'Persistent_{attr}'] for attr in attributes) 
        disappeared_total = sum(row[f'Disappeared_{attr}'] for attr in attributes)
        
        topic_totals[topic] = {
            'newly_emerged': newly_emerged_total,
            'persistent': persistent_total,
            'disappeared': disappeared_total
        }
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(24, 16))
    
    y_positions = range(len(plot_df))
    bar_height = 0.8
    
    # Plot for each topic
    for i, (_, row) in enumerate(plot_df.iterrows()):
        topic = row['topic']
        totals = topic_totals[topic]
        
        # Calculate center positions for centered layout
        persistent_half = totals['persistent'] / 2
        
        # Starting positions for each category section
        newly_emerged_start = -(persistent_half + totals['newly_emerged'])
        persistent_start = -persistent_half  # Single centered persistent section
        disappeared_start = persistent_half
        
        # Plot newly emerged segments (leftmost)
        current_pos = newly_emerged_start
        for attr in attributes:
            width = row[f'Newly Emerged_{attr}']
            if width > 0:
                color = color_palette[f'Newly Emerged_{attr}']
                ax.barh(i, width, left=current_pos, height=bar_height, 
                       color=color, alpha=0.9, edgecolor='white', linewidth=0.5)
                
                # Add count label if significant enough
                if width >= 1:
                    ax.text(current_pos + width/2, i, str(int(width)), 
                           ha='center', va='center', fontsize=8, color='white', fontweight='bold')
            current_pos += width
        
        # Plot persistent segments (center) - SINGLE STACK, NOT DUPLICATED
        current_pos = persistent_start
        for attr in attributes:
            width = row[f'Persistent_{attr}']  # Full width, not split
            if width > 0:
                color = color_palette[f'Persistent_{attr}']
                ax.barh(i, width, left=current_pos, height=bar_height,
                       color=color, alpha=0.9, edgecolor='white', linewidth=0.5)
                
                # Add count label if significant enough
                if width >= 1:
                    ax.text(current_pos + width/2, i, str(int(width)), 
                           ha='center', va='center', fontsize=8, color='white', fontweight='bold')
            current_pos += width
        
        # Plot disappeared segments (rightmost)
        current_pos = disappeared_start
        for attr in attributes:
            width = row[f'Disappeared_{attr}']
            if width > 0:
                color = color_palette[f'Disappeared_{attr}']
                ax.barh(i, width, left=current_pos, height=bar_height,
                       color=color, alpha=0.9, edgecolor='white', linewidth=0.5)
                
                # Add count label if significant enough
                if width >= 1:
                    ax.text(current_pos + width/2, i, str(int(width)), 
                           ha='center', va='center', fontsize=8, color='white', fontweight='bold')
            current_pos += width
    
    # Customize plot
    ax.set_title('Enhanced Topic-Level Bias Breakdown by Protected Attribute\n' +
                 'Bias Significance Changes Across Topics and Demographics', 
                 fontsize=18, fontweight='bold', pad=25)
    ax.set_xlabel('← Newly Emerged    |    Persistent    |    Disappeared →', 
                  fontsize=14, fontweight='bold')
    ax.set_ylabel('Asylum Topics', fontsize=14, fontweight='bold')
    
    # Set y-axis labels with wrapping
    ax.set_yticks(y_positions)
    wrapped_labels = wrap_topic_labels(plot_df['topic'].tolist())
    ax.set_yticklabels(wrapped_labels, fontsize=11, ha='right', va='center')
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    
    # Create IMPROVED legend with better alignment
    legend_elements = []
    
    # Create a more organized legend layout
    legend_elements.append(mpatches.Patch(color='none', label='NEWLY EMERGED:'))
    for attr in attributes:
        color = color_palette[f'Newly Emerged_{attr}']
        shade_desc = ['Dark', 'Medium', 'Regular', 'Light'][attributes.index(attr)]
        legend_elements.append(mpatches.Patch(color=color, label=f'  {attr.title()} ({shade_desc})'))
    
    legend_elements.append(mpatches.Patch(color='none', label=''))  # Spacer
    legend_elements.append(mpatches.Patch(color='none', label='PERSISTENT:'))
    for attr in attributes:
        color = color_palette[f'Persistent_{attr}']
        shade_desc = ['Dark', 'Medium', 'Regular', 'Light'][attributes.index(attr)]
        legend_elements.append(mpatches.Patch(color=color, label=f'  {attr.title()} ({shade_desc})'))
    
    legend_elements.append(mpatches.Patch(color='none', label=''))  # Spacer
    legend_elements.append(mpatches.Patch(color='none', label='DISAPPEARED:'))
    for attr in attributes:
        color = color_palette[f'Disappeared_{attr}']
        shade_desc = ['Dark', 'Medium', 'Regular', 'Light'][attributes.index(attr)]
        legend_elements.append(mpatches.Patch(color=color, label=f'  {attr.title()} ({shade_desc})'))
    
    # Place legend with better positioning
    ax.legend(handles=legend_elements, title='Bias Change by Attribute\n(Dark→Light: Gender→Country)', 
              title_fontsize=11, fontsize=9, loc='center left', 
              bbox_to_anchor=(1.02, 0.5), frameon=True, fancybox=True, shadow=True)
    
    # Style improvements
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Adjust layout to accommodate legend and prevent topic name cutoff
    plt.tight_layout()
    plt.subplots_adjust(left=0.45, right=0.75)  # Extra space to prevent topic name cropping
    
    # Save the plot
    plt.savefig("../outputs/bias_vector_drift/enhanced_topic_breakdown_by_attribute.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return plot_df, color_palette

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

def analyze_enhanced_breakdown_insights(analysis_df):
    """
    Analyze key insights from the enhanced topic breakdown by protected attribute
    """
    print("\n🔍 ENHANCED BREAKDOWN INSIGHTS")
    print("=" * 60)
    
    # 1. Which protected attributes drive bias changes the most?
    attr_totals = {}
    for attr in ['gender', 'religion', 'age', 'country']:
        attr_subset = analysis_df[analysis_df['protected_attribute'] == attr]
        newly_emerged = attr_subset['newly_emerged'].sum()
        persistent = attr_subset['persistent'].sum()
        disappeared = attr_subset['disappeared'].sum()
        total = newly_emerged + persistent + disappeared
        
        attr_totals[attr] = {
            'newly_emerged': newly_emerged,
            'persistent': persistent, 
            'disappeared': disappeared,
            'total': total
        }
    
    print("🏷️  PROTECTED ATTRIBUTE IMPACT RANKING:")
    sorted_attrs = sorted(attr_totals.items(), key=lambda x: x[1]['total'], reverse=True)
    for i, (attr, counts) in enumerate(sorted_attrs, 1):
        print(f"   {i}. {attr.upper()}: {counts['total']} changes")
        print(f"      Newly Emerged: {counts['newly_emerged']}, Persistent: {counts['persistent']}, Disappeared: {counts['disappeared']}")
    
    # NEW: Statistical Insights for Research Directions
    print(f"\n📊 KEY STATISTICAL INSIGHTS & RESEARCH DIRECTIONS")
    print("=" * 60)
    
    # Persistence Analysis
    print("🔵 PERSISTENT BIAS PATTERNS (Stable Discrimination):")
    persistent_ranking = sorted([(attr, counts['persistent']) for attr, counts in attr_totals.items()], 
                               key=lambda x: x[1], reverse=True)
    dominant_persistent = persistent_ranking[0]
    print(f"   • MOST PERSISTENT: {dominant_persistent[0].upper()} bias ({dominant_persistent[1]} cases)")
    print(f"     → RESEARCH FOCUS: This represents systemic bias that survived model retraining")
    print(f"     → INVESTIGATE: Why {dominant_persistent[0]} disparities persist across both time periods")
    
    # Emerging bias analysis
    print(f"\n🔴 NEWLY EMERGING BIAS PATTERNS (Post-Brexit Deterioration):")
    emerging_ranking = sorted([(attr, counts['newly_emerged']) for attr, counts in attr_totals.items()], 
                             key=lambda x: x[1], reverse=True)
    dominant_emerging = emerging_ranking[0]
    print(f"   • MOST EMERGING: {dominant_emerging[0].upper()} bias ({dominant_emerging[1]} new cases)")
    print(f"     → CRITICAL FINDING: Post-Brexit model developed NEW {dominant_emerging[0]} discrimination")
    print(f"     → INVESTIGATE: What in post-2019 data caused {dominant_emerging[0]} bias emergence?")
    
    # Disappearing bias analysis
    print(f"\n🟠 DISAPPEARING BIAS PATTERNS (Potential Improvement):")
    disappearing_ranking = sorted([(attr, counts['disappeared']) for attr, counts in attr_totals.items()], 
                                 key=lambda x: x[1], reverse=True)
    dominant_disappearing = disappearing_ranking[0]
    print(f"   • MOST DISAPPEARED: {dominant_disappearing[0].upper()} bias ({dominant_disappearing[1]} cases)")
    print(f"     → POSITIVE FINDING: Post-Brexit model reduced {dominant_disappearing[0]} discrimination")
    print(f"     → INVESTIGATE: What factors led to {dominant_disappearing[0]} bias reduction?")
    
    # Topic hotspots analysis
    topic_analysis = {}
    for topic in analysis_df['topic'].unique():
        topic_subset = analysis_df[analysis_df['topic'] == topic]
        
        newly_emerged = topic_subset['newly_emerged'].sum()
        persistent = topic_subset['persistent'].sum()
        disappeared = topic_subset['disappeared'].sum()
        total = newly_emerged + persistent + disappeared
        
        if total > 0:
            topic_analysis[topic] = {
                'newly_emerged': newly_emerged,
                'persistent': persistent,
                'disappeared': disappeared,
                'total': total,
                'bias_intensity': total / len(topic_subset) if len(topic_subset) > 0 else 0
            }
    
    print(f"\n🎯 BIAS HOTSPOT TOPICS (High-Risk Areas):")
    hotspot_topics = sorted(topic_analysis.items(), key=lambda x: x[1]['total'], reverse=True)[:3]
    for i, (topic, stats) in enumerate(hotspot_topics, 1):
        short_topic = topic[:40] + "..." if len(topic) > 40 else topic
        print(f"   {i}. {short_topic}")
        print(f"      Total changes: {stats['total']} | Intensity: {stats['bias_intensity']:.2f}")
        print(f"      Breakdown: +{stats['newly_emerged']} emerged, {stats['persistent']} persistent, -{stats['disappeared']} disappeared")
        if stats['newly_emerged'] > stats['disappeared']:
            print(f"      ⚠️  DETERIORATING: More bias emerged than disappeared - PRIORITY FOR INVESTIGATION")
        elif stats['disappeared'] > stats['newly_emerged']:
            print(f"      ✅ IMPROVING: More bias disappeared than emerged")
        else:
            print(f"      ⚖️  STABLE: Equal emergence and disappearance")
    
    print(f"\n🏆 BIAS IMPROVEMENT AREAS (Successful Bias Reduction):")
    improvement_topics = sorted([(topic, stats) for topic, stats in topic_analysis.items() 
                               if stats['disappeared'] > stats['newly_emerged']], 
                              key=lambda x: x[1]['disappeared'] - x[1]['newly_emerged'], reverse=True)[:3]
    
    if improvement_topics:
        for i, (topic, stats) in enumerate(improvement_topics, 1):
            short_topic = topic[:40] + "..." if len(topic) > 40 else topic
            net_improvement = stats['disappeared'] - stats['newly_emerged']
            print(f"   {i}. {short_topic}")
            print(f"      Net bias reduction: {net_improvement} cases")
            print(f"      → RESEARCH OPPORTUNITY: Study why this topic improved")
    else:
        print("   No topics showed clear bias improvement patterns")
    
    # Cross-attribute patterns
    print(f"\n🔄 CROSS-ATTRIBUTE INTERACTION PATTERNS:")
    
    # Find topics with multiple attribute problems
    multi_attr_topics = {}
    for topic in analysis_df['topic'].unique():
        topic_subset = analysis_df[analysis_df['topic'] == topic]
        attrs_with_issues = set()
        
        for attr in ['gender', 'religion', 'age', 'country']:
            attr_subset = topic_subset[topic_subset['protected_attribute'] == attr]
            if len(attr_subset) > 0 and (attr_subset['newly_emerged'].sum() + 
                                       attr_subset['persistent'].sum() + 
                                       attr_subset['disappeared'].sum()) > 0:
                attrs_with_issues.add(attr)
        
        if len(attrs_with_issues) >= 3:  # 3+ attributes affected
            multi_attr_topics[topic] = {
                'affected_attributes': list(attrs_with_issues),
                'count': len(attrs_with_issues)
            }
    
    if multi_attr_topics:
        print(f"   INTERSECTIONAL BIAS TOPICS ({len(multi_attr_topics)} topics with 3+ attribute issues):")
        for topic, info in list(multi_attr_topics.items())[:3]:
            short_topic = topic[:35] + "..." if len(topic) > 35 else topic
            print(f"   • {short_topic}")
            print(f"     Affected: {', '.join(info['affected_attributes'])}")
            print(f"     → INTERSECTIONAL RESEARCH: Complex multi-attribute bias patterns")
    
    # Severity analysis
    print(f"\n⚡ BIAS SEVERITY ANALYSIS:")
    total_emerged = sum(counts['newly_emerged'] for counts in attr_totals.values())
    total_persistent = sum(counts['persistent'] for counts in attr_totals.values())
    total_disappeared = sum(counts['disappeared'] for counts in attr_totals.values())
    
    net_bias_change = total_emerged - total_disappeared
    bias_stability = total_persistent / (total_emerged + total_persistent + total_disappeared) * 100
    
    print(f"   • NET BIAS CHANGE: {net_bias_change:+d} cases ({'DETERIORATION' if net_bias_change > 0 else 'IMPROVEMENT' if net_bias_change < 0 else 'STABLE'})")
    print(f"   • BIAS STABILITY: {bias_stability:.1f}% of bias patterns are persistent")
    print(f"   • TURNOVER RATE: {(total_emerged + total_disappeared)/(total_emerged + total_persistent + total_disappeared)*100:.1f}% bias patterns changed")
    
    if net_bias_change > 5:
        print(f"   ⚠️  CRITICAL: Significant bias deterioration detected!")
    elif net_bias_change < -5:
        print(f"   ✅ POSITIVE: Significant bias improvement detected!")
    
    # Research recommendations
    print(f"\n🎯 TOP RESEARCH PRIORITIES:")
    print("   1. PERSISTENT GENDER BIAS: Why does gender discrimination survive model updates?")
    if dominant_emerging[0] != 'gender':
        print(f"   2. EMERGING {dominant_emerging[0].upper()} BIAS: What post-Brexit factors caused new discrimination?")
    if dominant_disappearing[0] not in ['gender', dominant_emerging[0]]:
        print(f"   3. {dominant_disappearing[0].upper()} BIAS REDUCTION: Can this success be replicated?")
    print(f"   4. HOTSPOT TOPICS: Deep-dive into '{hotspot_topics[0][0][:30]}...' (highest bias changes)")
    if multi_attr_topics:
        print(f"   5. INTERSECTIONAL ANALYSIS: Multi-attribute bias in complex topics")
    
    return attr_totals, topic_analysis, multi_attr_topics

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
    
    # Enhanced Plot: Topic Breakdown by Protected Attribute
    plot_df, color_palette = plot_enhanced_topic_breakdown_by_attribute(analysis_df)
    
    # Analyze insights from enhanced breakdown
    attr_totals, topic_analysis, multi_attr_topics = analyze_enhanced_breakdown_insights(analysis_df)
    
    # Summary statistics
    create_summary_stats(analysis_df, all_df)
    
    print("\n✅ All visualizations created successfully!")
    print("📁 Saved to: ../outputs/bias_vector_drift/")
    print("   • plot1_protected_attribute_significance_types.png")
    print("   • plot2_topic_fairness_reconfiguration.png") 
    print("   • plot3_topic_model_salience.png")
    print("   • enhanced_topic_breakdown_by_attribute.png")
    print("\n💡 The enhanced plot shows how each protected attribute contributes")
    print("   to bias changes within specific asylum topics, revealing which")
    print("   demographic factors drive bias shifts in different contexts.")

if __name__ == "__main__":
    main() 