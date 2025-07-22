"""
Systematic Gender Bias Analysis - Characterizing Persistent Gender Disparities
================================================================================

This script analyzes the 18 persistent gender bias cases that survived model retraining,
examining their topic clustering, magnitude, and directionality.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_and_filter_gender_data():
    """Load data and filter for gender-related comparisons."""
    
    # Load the deduplicated fairness comparisons
    data_path = Path("../../outputs/vector_drift/deduplicated_fairness_comparisons.csv")
    df = pd.read_csv(data_path)
    
    # Filter for gender comparisons only
    gender_df = df[df['comparison_label'].str.contains(r'\(gender\)', regex=True)]
    
    print(f"Total gender comparisons: {len(gender_df)}")
    print(f"Data columns: {list(gender_df.columns)}")
    
    return gender_df

def identify_persistent_gender_bias(gender_df):
    """Identify the persistent gender bias cases (both significant pre and post)."""
    
    # Filter for persistent bias cases
    persistent_bias = gender_df[gender_df['sp_both_significant'] == True].copy()
    
    print(f"\n🔍 PERSISTENT GENDER BIAS CASES")
    print(f"=" * 50)
    print(f"Total persistent cases: {len(persistent_bias)}")
    
    # Extract topic and comparison info
    persistent_bias['comparison_type'] = persistent_bias['comparison_label'].str.extract(r'([^_]+_vs_[^_]+)')
    persistent_bias['topic'] = persistent_bias['comparison_label'].str.extract(r'\[([^\]]+)\]')
    
    return persistent_bias

def categorize_topics(persistent_bias):
    """Categorize topics into credibility, contradiction, trauma-related, etc."""
    
    # Define topic categories
    topic_categories = {
        'Credibility Assessment': [
            'Asylum seeker circumstances', 'Nature of persecution', 
            'Activist Persecution Ground', 'PSG (particular social group)'
        ],
        'Contradiction Detection': [
            'Contradiction: Dates of persecution', 'Contradiction: Family involvement in the persecution',
            'Contradiction: Location of harm', 'Contradiction: Sequence of events',
            'Contradiction: Persecutor identity confusion'
        ],
        'Trauma/Disclosure Related': [
            'Disclosure: Political persecution & sexual violence',
            'Disclosure: Religious persecution & mental health',
            'Disclosure: Domestic violence & criminal threats',
            'Disclosure: Ethnic violence & family separation',
            'Disclosure: Persecution for sexual orientation & mental health crisis'
        ],
        'Integration Assessment': [
            'Firm settlement', 'Assimilation Potential', 'Financial stability',
            'Intentions regarding work in the UK', 'Intentions regarding education in the UK'
        ],
        'Protection Framework': [
            '3rd safe country - Country safety definition',
            'Non-Refoulement - Interception at Sea'
        ]
    }
    
    # Add category column
    persistent_bias['topic_category'] = 'Other'
    for category, topics in topic_categories.items():
        mask = persistent_bias['topic'].isin(topics)
        persistent_bias.loc[mask, 'topic_category'] = category
    
    return persistent_bias, topic_categories

def analyze_bias_directionality(persistent_bias):
    """Analyze the directionality and magnitude of persistent biases."""
    
    # Extract gender comparison types
    persistent_bias['favors_gender'] = 'Unknown'
    persistent_bias['disfavors_gender'] = 'Unknown'
    
    # Determine bias direction based on magnitude sign
    for idx, row in persistent_bias.iterrows():
        comparison = row['comparison_type']
        pre_mag = row['pre_brexit_sp_magnitude']
        post_mag = row['post_brexit_sp_magnitude']
        
        # Parse comparison
        if 'Female_vs_Male' in comparison:
            if pre_mag > 0 and post_mag > 0:  # Positive = Female favored
                persistent_bias.loc[idx, 'favors_gender'] = 'Female'
                persistent_bias.loc[idx, 'disfavors_gender'] = 'Male'
            elif pre_mag < 0 and post_mag < 0:  # Negative = Male favored
                persistent_bias.loc[idx, 'favors_gender'] = 'Male'
                persistent_bias.loc[idx, 'disfavors_gender'] = 'Female'
        elif 'Male_vs_Non-binary' in comparison:
            if pre_mag > 0 and post_mag > 0:  # Positive = Non-binary favored
                persistent_bias.loc[idx, 'favors_gender'] = 'Non-binary'
                persistent_bias.loc[idx, 'disfavors_gender'] = 'Male'
            elif pre_mag < 0 and post_mag < 0:  # Negative = Male favored
                persistent_bias.loc[idx, 'favors_gender'] = 'Male'
                persistent_bias.loc[idx, 'disfavors_gender'] = 'Non-binary'
        elif 'Female_vs_Non-binary' in comparison:
            if pre_mag > 0 and post_mag > 0:  # Positive = Non-binary favored
                persistent_bias.loc[idx, 'favors_gender'] = 'Non-binary'
                persistent_bias.loc[idx, 'disfavors_gender'] = 'Female'
            elif pre_mag < 0 and post_mag < 0:  # Negative = Female favored
                persistent_bias.loc[idx, 'favors_gender'] = 'Female'
                persistent_bias.loc[idx, 'disfavors_gender'] = 'Non-binary'
    
    return persistent_bias

def create_comprehensive_summary_table(persistent_bias):
    """Create a comprehensive summary table of persistent gender bias cases."""
    
    # Create summary table
    summary_columns = [
        'topic', 'topic_category', 'comparison_type',
        'favors_gender', 'disfavors_gender',
        'pre_brexit_sp_magnitude', 'post_brexit_sp_magnitude', 
        'sp_magnitude_difference'
    ]
    
    summary_table = persistent_bias[summary_columns].copy()
    
    # Add magnitude interpretation
    summary_table['bias_strength'] = summary_table.apply(
        lambda row: 'Very Strong' if abs(row['post_brexit_sp_magnitude']) > 0.3
        else 'Strong' if abs(row['post_brexit_sp_magnitude']) > 0.15
        else 'Moderate' if abs(row['post_brexit_sp_magnitude']) > 0.05
        else 'Weak', axis=1
    )
    
    # Add trend analysis
    summary_table['bias_trend'] = summary_table.apply(
        lambda row: 'Strengthening' if abs(row['sp_magnitude_difference']) > 0.05 and 
        abs(row['post_brexit_sp_magnitude']) > abs(row['pre_brexit_sp_magnitude'])
        else 'Weakening' if abs(row['sp_magnitude_difference']) > 0.05 and 
        abs(row['post_brexit_sp_magnitude']) < abs(row['pre_brexit_sp_magnitude'])
        else 'Stable', axis=1
    )
    
    return summary_table

def analyze_topic_clustering(persistent_bias):
    """Analyze which topic categories show the most persistent bias."""
    
    category_analysis = persistent_bias.groupby('topic_category').agg({
        'comparison_label': 'count',
        'post_brexit_sp_magnitude': ['mean', 'std', lambda x: abs(x).mean()],
        'sp_magnitude_difference': ['mean', 'std']
    }).round(4)
    
    category_analysis.columns = [
        'bias_count', 'avg_magnitude', 'magnitude_std', 'avg_abs_magnitude',
        'avg_change', 'change_std'
    ]
    
    return category_analysis

def create_visualizations(persistent_bias, summary_table):
    """Create visualizations of persistent gender bias patterns."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('🎯 Systematic Gender Bias Analysis: 18 Persistent Cases', fontsize=20, fontweight='bold')
    
    # 1. Topic Category Distribution
    ax1 = axes[0, 0]
    category_counts = persistent_bias['topic_category'].value_counts()
    colors = sns.color_palette("viridis", len(category_counts))
    bars = ax1.bar(range(len(category_counts)), category_counts.values, color=colors)
    ax1.set_title('📊 Persistent Bias by Topic Category', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Topic Category')
    ax1.set_ylabel('Number of Persistent Bias Cases')
    ax1.set_xticks(range(len(category_counts)))
    ax1.set_xticklabels(category_counts.index, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, category_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(value), ha='center', va='bottom', fontweight='bold')
    
    # 2. Bias Magnitude Distribution
    ax2 = axes[0, 1]
    ax2.scatter(persistent_bias['pre_brexit_sp_magnitude'], 
                persistent_bias['post_brexit_sp_magnitude'],
                c=range(len(persistent_bias)), cmap='plasma', s=100, alpha=0.7)
    ax2.plot([-0.5, 0.5], [-0.5, 0.5], 'k--', alpha=0.5, label='No Change Line')
    ax2.set_title('📈 Bias Magnitude: Pre vs Post Brexit', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Pre-Brexit SP Magnitude')
    ax2.set_ylabel('Post-Brexit SP Magnitude')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Gender Group Analysis
    ax3 = axes[1, 0]
    gender_favored = persistent_bias['favors_gender'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(gender_favored)]
    wedges, texts, autotexts = ax3.pie(gender_favored.values, labels=gender_favored.index, 
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax3.set_title('⚖️ Which Gender Groups Are Favored?', fontsize=14, fontweight='bold')
    
    # 4. Bias Strength Distribution
    ax4 = axes[1, 1]
    strength_counts = summary_table['bias_strength'].value_counts()
    colors = ['#FF4444', '#FF8800', '#FFBB00', '#88DD88'][:len(strength_counts)]
    bars = ax4.bar(range(len(strength_counts)), strength_counts.values, color=colors)
    ax4.set_title('💪 Bias Strength Distribution', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Bias Strength Category')
    ax4.set_ylabel('Number of Cases')
    ax4.set_xticks(range(len(strength_counts)))
    ax4.set_xticklabels(strength_counts.index)
    
    # Add value labels
    for bar, value in zip(bars, strength_counts.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(value), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path("../../outputs/gender_analysis/systematic_gender_bias_analysis.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved: {output_path}")
    
    return fig

def generate_research_insights(persistent_bias, summary_table, category_analysis):
    """Generate actionable research insights."""
    
    insights = []
    
    # Overall patterns
    total_cases = len(persistent_bias)
    avg_magnitude = abs(persistent_bias['post_brexit_sp_magnitude']).mean()
    
    insights.append(f"🔍 SYSTEMATIC GENDER BIAS FINDINGS")
    insights.append(f"=" * 50)
    insights.append(f"• Total persistent cases: {total_cases}")
    insights.append(f"• Average bias magnitude: {avg_magnitude:.3f}")
    
    # Topic clustering insights
    top_category = category_analysis.index[0]
    top_count = category_analysis.loc[top_category, 'bias_count']
    
    insights.append(f"\n🎯 TOPIC CLUSTERING PATTERNS")
    insights.append(f"• Highest concentration: {top_category} ({top_count} cases)")
    insights.append(f"• Most problematic area: {category_analysis.loc[category_analysis['avg_abs_magnitude'].idxmax()].name}")
    
    # Gender group analysis
    favored_counts = persistent_bias['favors_gender'].value_counts()
    most_favored = favored_counts.index[0]
    
    insights.append(f"\n⚖️ GENDER GROUP DISPARITIES")
    insights.append(f"• Most favored group: {most_favored} ({favored_counts[most_favored]} cases)")
    insights.append(f"• Gender distribution: {dict(favored_counts)}")
    
    # Trend analysis
    strengthening = summary_table[summary_table['bias_trend'] == 'Strengthening']
    weakening = summary_table[summary_table['bias_trend'] == 'Weakening']
    
    insights.append(f"\n📈 BIAS EVOLUTION TRENDS")
    insights.append(f"• Strengthening biases: {len(strengthening)} cases")
    insights.append(f"• Weakening biases: {len(weakening)} cases")
    
    # Critical cases
    very_strong = summary_table[summary_table['bias_strength'] == 'Very Strong']
    if len(very_strong) > 0:
        insights.append(f"\n🚨 CRITICAL CASES REQUIRING IMMEDIATE ATTENTION")
        for _, case in very_strong.iterrows():
            insights.append(f"• {case['topic']}: {case['favors_gender']} vs {case['disfavors_gender']} "
                          f"(magnitude: {case['post_brexit_sp_magnitude']:.3f})")
    
    return insights

def create_gender_topic_heatmap(gender_df):
    """Create comprehensive heatmap of SP scores across gender-topic pairs."""
    
    # Extract comparison info for all gender comparisons (not just persistent)
    gender_df = gender_df.copy()
    
    # Extract gender pairs and topics separately  
    gender_df['gender_pair_raw'] = gender_df['comparison_label'].str.extract(r'([^_]+_vs_[^_]+) \(gender\)')
    gender_df['topic'] = gender_df['comparison_label'].str.extract(r'\[([^\]]+)\]')
    
    # Map to clean gender pair names (exactly 3 columns as requested)
    def map_to_clean_gender_pair(raw_pair):
        if pd.isna(raw_pair):
            return None
        elif raw_pair == 'Female_vs_Male':
            return 'Female vs Male'
        elif raw_pair == 'Male_vs_Non-binary':
            return 'Male vs Non-binary'
        elif raw_pair == 'Female_vs_Non-binary':
            return 'Female vs Non-binary'
        else:
            return None
    
    gender_df['gender_pair'] = gender_df['gender_pair_raw'].apply(map_to_clean_gender_pair)
    
    # Filter to only include the 3 main gender pairs
    gender_df = gender_df[gender_df['gender_pair'].notna()]
    
    # Truncate long topic names for better Y-axis readability
    def truncate_topic_name(topic_name, max_length=18):  # Reduced max length
        if pd.isna(topic_name) or len(topic_name) <= max_length:
            return topic_name
        
        # Try to split at logical points first
        if ':' in topic_name:
            parts = topic_name.split(':', 1)
            if len(parts[0]) <= max_length:
                second_part = parts[1].strip()
                if len(second_part) > max_length:
                    second_part = second_part[:max_length-3] + "..."
                return f"{parts[0]}:\n{second_part}"
        
        # Split at word boundaries with shorter lines
        words = topic_name.split()
        line1, line2 = [], []
        line1_len, line2_len = 0, 0
        
        for word in words:
            if line1_len + len(word) + 1 <= max_length and len(line1) < 3:  # Limit first line to 3 words max
                line1.append(word)
                line1_len += len(word) + 1
            else:
                if line2_len + len(word) + 1 <= max_length:
                    line2.append(word)
                    line2_len += len(word) + 1
                else:
                    break
        
        line1_str = ' '.join(line1)
        line2_str = ' '.join(line2)
        
        # Add ellipsis if there are remaining words
        remaining_words = words[len(line1) + len(line2):]
        if remaining_words:
            if len(line2_str) <= max_length - 3:
                line2_str += "..."
            else:
                line2_str = line2_str[:max_length-3] + "..."
            
        return f"{line1_str}\n{line2_str}" if line2_str else line1_str
    
    gender_df['topic_truncated'] = gender_df['topic'].apply(truncate_topic_name)
    
    # Create pivot tables for pre and post Brexit using truncated topic names
    pre_pivot = gender_df.pivot_table(
        values='pre_brexit_sp_magnitude',
        index='topic_truncated',
        columns='gender_pair',
        aggfunc='first'
    ).fillna(0)
    
    post_pivot = gender_df.pivot_table(
        values='post_brexit_sp_magnitude', 
        index='topic_truncated',
        columns='gender_pair',
        aggfunc='first'
    ).fillna(0)
    
    # Create significance masks
    pre_sig_pivot = gender_df.pivot_table(
        values='pre_brexit_sp_significance',
        index='topic_truncated',
        columns='gender_pair', 
        aggfunc='first'
    ).fillna(False)
    
    post_sig_pivot = gender_df.pivot_table(
        values='post_brexit_sp_significance',
        index='topic_truncated',
        columns='gender_pair',
        aggfunc='first'
    ).fillna(False)
    
    # Set up the figure with subplots - adjust layout for legend and Y-axis labels
    fig, axes = plt.subplots(1, 2, figsize=(22, 16))
    fig.suptitle('🔥 Gender Bias Heatmaps: SP Scores Across Topics & Gender Pairs', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # Define color scheme - diverging for bias direction
    cmap = sns.diverging_palette(250, 10, as_cmap=True, center='light')
    
    # Determine common scale for both heatmaps
    vmin = min(pre_pivot.values.min(), post_pivot.values.min()) 
    vmax = max(pre_pivot.values.max(), post_pivot.values.max())
    abs_max = max(abs(vmin), abs(vmax))
    
    # Pre-Brexit Heatmap
    ax1 = axes[0]
    mask_pre = ~pre_sig_pivot  # Mask non-significant values
    
    sns.heatmap(pre_pivot, 
                annot=True, 
                fmt='.3f',
                cmap=cmap,
                center=0,
                vmin=-abs_max,
                vmax=abs_max,
                mask=mask_pre,
                cbar_kws={'label': 'SP Magnitude', 'shrink': 0.8},
                ax=ax1,
                linewidths=0.5,
                linecolor='white')
    
    ax1.set_title('Pre-Brexit Model (2013-2016)\nSignificant Biases Only', 
                  fontsize=13, fontweight='bold', pad=20)
    ax1.set_xlabel('Gender Comparison Pairs', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Asylum Topics', fontsize=11, fontweight='bold')
    
    # Rotate labels for better readability
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=9)
    
    # Post-Brexit Heatmap
    ax2 = axes[1]
    mask_post = ~post_sig_pivot  # Mask non-significant values
    
    sns.heatmap(post_pivot,
                annot=True,
                fmt='.3f', 
                cmap=cmap,
                center=0,
                vmin=-abs_max,
                vmax=abs_max,
                mask=mask_post,
                cbar_kws={'label': 'SP Magnitude', 'shrink': 0.8},
                ax=ax2,
                linewidths=0.5,
                linecolor='white')
    
    ax2.set_title('Post-Brexit Model (2019-2025)\nSignificant Biases Only', 
                  fontsize=13, fontweight='bold', pad=20)
    ax2.set_xlabel('Gender Comparison Pairs', fontsize=11, fontweight='bold')
    ax2.set_ylabel('')  # Remove duplicate y-label
    
    # Rotate labels for better readability
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=9)
    
    # Add interpretation legend at the bottom
    legend_text = """📊 Reading the Heatmap: Red = Bias AGAINST first group (negative SP) | Blue = Bias FOR first group (positive SP) | White/Missing = Non-significant bias | Darker colors = Stronger bias
🔍 Gender Pair Examples: Female vs Male: + = Female favored, - = Male favored | Male vs Non-binary: + = Non-binary favored, - = Male favored | Female vs Non-binary: + = Non-binary favored, - = Female favored"""
    
    fig.text(0.5, 0.02, legend_text, fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
             ha='center', va='bottom', wrap=True)
    
    plt.tight_layout(rect=[0.12, 0.08, 1, 0.95])  # Leave space for legend at bottom and Y-axis labels on left
    
    # Save the heatmap
    output_path = Path("../../outputs/gender_analysis/gender_topic_sp_heatmap.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"🔥 Gender-Topic SP Heatmap saved: {output_path}")
    
    return fig, pre_pivot, post_pivot

def create_difference_heatmap(gender_df):
    """Create heatmap showing change in SP scores from pre to post Brexit."""
    
    # Extract comparison info
    gender_df = gender_df.copy()
    
    # Extract gender pairs and topics separately  
    gender_df['gender_pair_raw'] = gender_df['comparison_label'].str.extract(r'([^_]+_vs_[^_]+) \(gender\)')
    gender_df['topic'] = gender_df['comparison_label'].str.extract(r'\[([^\]]+)\]')
    
    # Map to clean gender pair names (exactly 3 columns as requested)
    def map_to_clean_gender_pair(raw_pair):
        if pd.isna(raw_pair):
            return None
        elif raw_pair == 'Female_vs_Male':
            return 'Female vs Male'
        elif raw_pair == 'Male_vs_Non-binary':
            return 'Male vs Non-binary'
        elif raw_pair == 'Female_vs_Non-binary':
            return 'Female vs Non-binary'
        else:
            return None
    
    gender_df['gender_pair'] = gender_df['gender_pair_raw'].apply(map_to_clean_gender_pair)
    
    # Filter to only include the 3 main gender pairs
    gender_df = gender_df[gender_df['gender_pair'].notna()]
    
    # Truncate long topic names for better Y-axis readability
    def truncate_topic_name(topic_name, max_length=18):  # Reduced max length
        if pd.isna(topic_name) or len(topic_name) <= max_length:
            return topic_name
        
        # Try to split at logical points first
        if ':' in topic_name:
            parts = topic_name.split(':', 1)
            if len(parts[0]) <= max_length:
                second_part = parts[1].strip()
                if len(second_part) > max_length:
                    second_part = second_part[:max_length-3] + "..."
                return f"{parts[0]}:\n{second_part}"
        
        # Split at word boundaries with shorter lines
        words = topic_name.split()
        line1, line2 = [], []
        line1_len, line2_len = 0, 0
        
        for word in words:
            if line1_len + len(word) + 1 <= max_length and len(line1) < 3:  # Limit first line to 3 words max
                line1.append(word)
                line1_len += len(word) + 1
            else:
                if line2_len + len(word) + 1 <= max_length:
                    line2.append(word)
                    line2_len += len(word) + 1
                else:
                    break
        
        line1_str = ' '.join(line1)
        line2_str = ' '.join(line2)
        
        # Add ellipsis if there are remaining words
        remaining_words = words[len(line1) + len(line2):]
        if remaining_words:
            if len(line2_str) <= max_length - 3:
                line2_str += "..."
            else:
                line2_str = line2_str[:max_length-3] + "..."
            
        return f"{line1_str}\n{line2_str}" if line2_str else line1_str
    
    gender_df['topic_truncated'] = gender_df['topic'].apply(truncate_topic_name)
    
    # Create pivot table for differences using truncated topic names
    diff_pivot = gender_df.pivot_table(
        values='sp_magnitude_difference',
        index='topic_truncated',
        columns='gender_pair',
        aggfunc='first'
    ).fillna(0)
    
    # Create significance mask - show only where at least one period was significant
    both_sig_pivot = gender_df.pivot_table(
        values='sp_both_significant',
        index='topic_truncated', 
        columns='gender_pair',
        aggfunc='first'
    ).fillna(False)
    
    gained_sig_pivot = gender_df.pivot_table(
        values='sp_gained_significance',
        index='topic_truncated',
        columns='gender_pair', 
        aggfunc='first'
    ).fillna(False)
    
    lost_sig_pivot = gender_df.pivot_table(
        values='sp_lost_significance',
        index='topic_truncated',
        columns='gender_pair',
        aggfunc='first'
    ).fillna(False)
    
    # Mask for any significance
    any_significance = both_sig_pivot | gained_sig_pivot | lost_sig_pivot
    
    # Set up figure - adjust layout for legend
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
    fig.suptitle('📈 Gender Bias Evolution: Change in SP Scores (Post - Pre Brexit)', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # Create custom colormap for changes
    cmap = sns.diverging_palette(10, 250, as_cmap=True, center='light')
    
    # Create annotations with significance indicators
    annot_data = diff_pivot.copy()
    for i, topic in enumerate(diff_pivot.index):
        for j, comparison in enumerate(diff_pivot.columns):
            value = diff_pivot.iloc[i, j]
            
            # Add significance indicators
            if both_sig_pivot.iloc[i, j]:
                indicator = ' ●●'  # Persistent bias
            elif gained_sig_pivot.iloc[i, j]:
                indicator = ' ▲'   # Emerged bias
            elif lost_sig_pivot.iloc[i, j]:
                indicator = ' ▼'   # Disappeared bias
            else:
                indicator = ''
                
            annot_data.iloc[i, j] = f'{value:.3f}{indicator}'
    
    # Create heatmap
    sns.heatmap(diff_pivot,
                annot=annot_data,
                fmt='',
                cmap=cmap,
                center=0,
                mask=~any_significance,
                cbar_kws={'label': 'SP Magnitude Change', 'shrink': 0.8},
                ax=ax,
                linewidths=0.5,
                linecolor='white')
    
    ax.set_title('Bias Evolution Patterns (Significant Cases Only)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Gender Comparison Pairs', fontsize=12, fontweight='bold')
    ax.set_ylabel('Asylum Topics', fontsize=12, fontweight='bold')
    
    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # Add legend at the bottom
    legend_text = """📊 Change Interpretation: Red = Bias increased (worse) | Blue = Bias decreased (better) | White/Missing = No significant bias
🔍 Significance Indicators: ●● = Persistent bias (both periods) | ▲ = Emerged bias (gained significance) | ▼ = Disappeared bias (lost significance)"""
    
    fig.text(0.5, 0.02, legend_text, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
             ha='center', va='bottom', wrap=True)
    
    plt.tight_layout(rect=[0.12, 0.08, 1, 0.95])  # Leave space for legend at bottom and Y-axis labels on left
    
    # Save the difference heatmap
    output_path = Path("../../outputs/gender_analysis/gender_bias_evolution_heatmap.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📈 Gender Bias Evolution Heatmap saved: {output_path}")
    
    return fig, diff_pivot

def analyze_heatmap_patterns(pre_pivot, post_pivot, diff_pivot):
    """Analyze patterns in the heatmaps to extract insights."""
    
    insights = []
    
    insights.append("🔥 HEATMAP PATTERN ANALYSIS")
    insights.append("=" * 50)
    
    # 1. Most biased topics
    pre_topic_bias = pre_pivot.abs().sum(axis=1).sort_values(ascending=False)
    post_topic_bias = post_pivot.abs().sum(axis=1).sort_values(ascending=False)
    
    insights.append(f"\n📊 MOST BIASED TOPICS")
    insights.append(f"Pre-Brexit: {pre_topic_bias.head(3).index.tolist()}")
    insights.append(f"Post-Brexit: {post_topic_bias.head(3).index.tolist()}")
    
    # 2. Most problematic gender pairs
    pre_pair_bias = pre_pivot.abs().sum(axis=0).sort_values(ascending=False)
    post_pair_bias = post_pivot.abs().sum(axis=0).sort_values(ascending=False)
    
    insights.append(f"\n⚖️ MOST PROBLEMATIC GENDER PAIRS")
    insights.append(f"Pre-Brexit: {pre_pair_bias.head(3).index.tolist()}")
    insights.append(f"Post-Brexit: {post_pair_bias.head(3).index.tolist()}")
    
    # 3. Biggest changes
    biggest_increases = diff_pivot.max(axis=1).sort_values(ascending=False).head(3)
    biggest_decreases = diff_pivot.min(axis=1).sort_values(ascending=True).head(3)
    
    insights.append(f"\n📈 BIGGEST BIAS CHANGES")
    insights.append(f"Worst deterioration: {biggest_increases.index.tolist()}")
    insights.append(f"Best improvement: {biggest_decreases.index.tolist()}")
    
    # 4. Non-binary specific analysis
    non_binary_cols = [col for col in post_pivot.columns if 'Non-binary' in col]
    if non_binary_cols:
        non_binary_bias = post_pivot[non_binary_cols].abs().mean(axis=1)
        insights.append(f"\n🏳️‍⚧️ NON-BINARY BIAS HOTSPOTS")
        insights.append(f"Highest bias topics: {non_binary_bias.sort_values(ascending=False).head(3).index.tolist()}")
    
    return insights

def main():
    """Main analysis function."""
    
    print("🎯 SYSTEMATIC GENDER BIAS ANALYSIS")
    print("=" * 60)
    
    # Load and process data
    gender_df = load_and_filter_gender_data()
    persistent_bias = identify_persistent_gender_bias(gender_df)
    persistent_bias, topic_categories = categorize_topics(persistent_bias)
    persistent_bias = analyze_bias_directionality(persistent_bias)
    
    # Create analysis outputs
    summary_table = create_comprehensive_summary_table(persistent_bias)
    category_analysis = analyze_topic_clustering(persistent_bias)
    
    # Display key findings
    print(f"\n📋 SUMMARY TABLE OF {len(summary_table)} PERSISTENT CASES")
    print("=" * 80)
    display_columns = ['topic', 'topic_category', 'favors_gender', 'bias_strength', 'bias_trend']
    print(summary_table[display_columns].to_string(index=False))
    
    print(f"\n📊 CATEGORY ANALYSIS")
    print("=" * 40)
    print(category_analysis)
    
    # Generate visualizations
    fig = create_visualizations(persistent_bias, summary_table)
    
    # Create comprehensive heatmaps
    print(f"\n🔥 CREATING GENDER-TOPIC HEATMAPS")
    print("=" * 50)
    heatmap_fig, pre_pivot, post_pivot = create_gender_topic_heatmap(gender_df)
    
    print(f"\n📈 CREATING BIAS EVOLUTION HEATMAP")
    print("=" * 50)
    evolution_fig, diff_pivot = create_difference_heatmap(gender_df)
    
    # Analyze heatmap patterns
    heatmap_insights = analyze_heatmap_patterns(pre_pivot, post_pivot, diff_pivot)
    
    # Generate insights
    insights = generate_research_insights(persistent_bias, summary_table, category_analysis)
    
    print(f"\n" + "\n".join(insights))
    print(f"\n" + "\n".join(heatmap_insights))
    
    # Save detailed results
    output_dir = Path("../../outputs/gender_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    
    summary_table.to_csv(output_dir / "persistent_gender_bias_detailed.csv", index=False)
    category_analysis.to_csv(output_dir / "gender_bias_category_analysis.csv")
    
    # Save heatmap data
    pre_pivot.to_csv(output_dir / "pre_brexit_gender_sp_matrix.csv")
    post_pivot.to_csv(output_dir / "post_brexit_gender_sp_matrix.csv")
    diff_pivot.to_csv(output_dir / "gender_bias_change_matrix.csv")
    
    print(f"\n💾 DETAILED RESULTS SAVED TO: {output_dir}")
    print(f"• Summary table: persistent_gender_bias_detailed.csv")
    print(f"• Category analysis: gender_bias_category_analysis.csv")
    print(f"• Main visualization: systematic_gender_bias_analysis.png")
    print(f"• SP Heatmaps: gender_topic_sp_heatmap.png")
    print(f"• Evolution Heatmap: gender_bias_evolution_heatmap.png")
    print(f"• SP Matrices: pre/post_brexit_gender_sp_matrix.csv & gender_bias_change_matrix.csv")
    
    return persistent_bias, summary_table, category_analysis, pre_pivot, post_pivot, diff_pivot

if __name__ == "__main__":
    persistent_bias, summary_table, category_analysis, pre_pivot, post_pivot, diff_pivot = main() 