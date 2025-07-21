#!/usr/bin/env python3
"""
Age Bias Analysis: Unpacking Post-Brexit Age Disparities
========================================================

This script analyzes the emerging age-based disparities in the post-Brexit model,
focusing on:
- Volume: 16 newly emerged cases + 12 disappeared ones → net increase of 4
- Topics: Where age-based changes happened most
- Values: Which age ranges are affected (12, 18, 25, 32, 40, 70)
- Comparison: How age bias flips or appears in post-Brexit model

Author: Bias Analysis Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import re
import textwrap
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

def parse_age_comparison(label):
    """
    Parse age comparison label to extract ages and topic
    
    Example: "25_vs_12 (age) [Asylum seeker circumstances]"
    Returns: {'age1': 25, 'age2': 12, 'topic': 'Asylum seeker circumstances'}
    """
    # Extract ages
    age_match = re.search(r'(\d+)_vs_(\d+) \(age\)', label)
    if not age_match:
        return None
    
    age1, age2 = int(age_match.group(1)), int(age_match.group(2))
    
    # Extract topic
    topic_match = re.search(r'\[([^\]]+)\]', label)
    topic = topic_match.group(1) if topic_match else 'unknown'
    
    return {
        'age1': age1,
        'age2': age2,
        'topic': topic,
        'age_pair': f"{age1}_vs_{age2}"
    }

def load_age_bias_data():
    """Load and filter for age-related bias comparisons"""
    
    print("📂 Loading age bias data...")
    
    # Load deduplicated data
    df = pd.read_csv("../outputs/bias_vector_drift/deduplicated_fairness_comparisons.csv")
    
    # Filter for age comparisons
    age_mask = df['comparison_label'].str.contains(r'\(age\)', regex=True)
    age_df = df[age_mask].copy()
    
    print(f"📊 Total age comparisons: {len(age_df)}")
    
    # Parse age comparison labels
    parsed_data = []
    for _, row in age_df.iterrows():
        parsed = parse_age_comparison(row['comparison_label'])
        if parsed:
            parsed.update(row.to_dict())
            parsed_data.append(parsed)
    
    age_analysis_df = pd.DataFrame(parsed_data)
    
    print(f"✅ Successfully parsed {len(age_analysis_df)} age comparisons")
    
    return age_analysis_df

def categorize_bias_changes(df):
    """Categorize bias changes into emerged, disappeared, persistent categories"""
    
    # Define bias change categories
    df['bias_category'] = 'stable'
    
    # Newly emerged bias (gained significance)
    df.loc[df['sp_gained_significance'] == True, 'bias_category'] = 'newly_emerged'
    
    # Disappeared bias (lost significance)
    df.loc[df['sp_lost_significance'] == True, 'bias_category'] = 'disappeared'
    
    # Persistent bias (significant in both models)
    df.loc[df['sp_both_significant'] == True, 'bias_category'] = 'persistent'
    
    return df

def analyze_age_range_patterns(df):
    """Analyze which age ranges are most affected by bias changes"""
    
    # Create age range categories
    age_categories = {
        12: 'Child (12)',
        18: 'Young Adult (18)', 
        25: 'Young Adult (25)',
        32: 'Adult (32)',
        40: 'Middle-aged (40)',
        70: 'Elderly (70)'
    }
    
    # Analyze patterns by age group
    age_patterns = defaultdict(lambda: defaultdict(int))
    
    for _, row in df.iterrows():
        if row['bias_category'] != 'stable':
            # Count involvement of each age in bias changes
            age1_cat = age_categories[row['age1']]
            age2_cat = age_categories[row['age2']]
            
            age_patterns[age1_cat][row['bias_category']] += 1
            age_patterns[age2_cat][row['bias_category']] += 1
    
    # Convert to DataFrame for analysis
    age_pattern_data = []
    for age_group, patterns in age_patterns.items():
        for bias_type, count in patterns.items():
            age_pattern_data.append({
                'age_group': age_group,
                'bias_type': bias_type,
                'count': count
            })
    
    age_pattern_df = pd.DataFrame(age_pattern_data)
    
    return age_pattern_df, age_categories

def analyze_topic_age_matrix(df):
    """Create topic × age heatmap data"""
    
    # Get unique topics and age pairs
    topics = sorted(df['topic'].unique())
    age_pairs = sorted(df['age_pair'].unique())
    
    # Create matrix for each bias type
    bias_types = ['newly_emerged', 'disappeared', 'persistent']
    
    matrices = {}
    for bias_type in bias_types:
        matrix_data = []
        filtered_df = df[df['bias_category'] == bias_type]
        
        for topic in topics:
            row = []
            for age_pair in age_pairs:
                count = len(filtered_df[
                    (filtered_df['topic'] == topic) & 
                    (filtered_df['age_pair'] == age_pair)
                ])
                row.append(count)
            matrix_data.append(row)
        
        matrices[bias_type] = {
            'data': np.array(matrix_data),
            'topics': topics,
            'age_pairs': age_pairs
        }
    
    return matrices

def create_age_vulnerability_analysis(df):
    """Analyze which ages are more vulnerable to bias"""
    
    # Analyze by individual ages
    individual_age_bias = defaultdict(lambda: defaultdict(int))
    
    for _, row in df.iterrows():
        if row['bias_category'] in ['newly_emerged', 'persistent']:
            # Determine which age is disadvantaged based on magnitude
            if row['post_brexit_sp_magnitude'] > 0:
                # Positive magnitude means age1 is disadvantaged vs age2
                disadvantaged_age = row['age1']
                advantaged_age = row['age2']
            else:
                # Negative magnitude means age2 is disadvantaged vs age1
                disadvantaged_age = row['age2'] 
                advantaged_age = row['age1']
            
            individual_age_bias[disadvantaged_age]['disadvantaged'] += 1
            individual_age_bias[advantaged_age]['advantaged'] += 1
    
    # Convert to analysis format
    vulnerability_data = []
    all_ages = [12, 18, 25, 32, 40, 70]
    
    for age in all_ages:
        disadvantaged = individual_age_bias[age]['disadvantaged']
        advantaged = individual_age_bias[age]['advantaged']
        net_disadvantage = disadvantaged - advantaged
        
        vulnerability_data.append({
            'age': age,
            'disadvantaged_count': disadvantaged,
            'advantaged_count': advantaged,
            'net_disadvantage': net_disadvantage,
            'total_involvement': disadvantaged + advantaged
        })
    
    return pd.DataFrame(vulnerability_data)

def plot_age_bias_overview(df):
    """Create overview plot of age bias changes"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Bias category distribution
    bias_counts = df['bias_category'].value_counts()
    colors_overview = {'newly_emerged': '#FF6B6B', 'disappeared': '#FFA94D', 
                      'persistent': '#4D96FF', 'stable': '#A8E6CF'}
    
    bias_counts.plot(kind='bar', ax=ax1, color=[colors_overview.get(x, '#808080') for x in bias_counts.index])
    ax1.set_title('Age Bias Changes: Volume Analysis', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Bias Change Category', fontsize=12)
    ax1.set_ylabel('Number of Comparisons', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add count labels
    for i, v in enumerate(bias_counts.values):
        ax1.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 2. Topic involvement in age bias changes
    topic_bias = df[df['bias_category'] != 'stable']['topic'].value_counts().head(10)
    topic_bias.plot(kind='barh', ax=ax2, color='#FF6B6B')
    ax2.set_title('Topics Most Affected by Age Bias Changes', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Number of Age Bias Changes', fontsize=12)
    
    # 3. Age pair analysis
    age_pair_bias = df[df['bias_category'] != 'stable']['age_pair'].value_counts().head(12)
    age_pair_bias.plot(kind='bar', ax=ax3, color='#4D96FF')
    ax3.set_title('Most Affected Age Comparisons', fontsize=16, fontweight='bold')
    ax3.set_xlabel('Age Comparison', fontsize=12)
    ax3.set_ylabel('Number of Bias Changes', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Pre vs Post Brexit magnitude comparison for significant changes
    significant_changes = df[df['bias_category'].isin(['newly_emerged', 'disappeared'])]
    
    ax4.scatter(significant_changes['pre_brexit_sp_magnitude'], 
               significant_changes['post_brexit_sp_magnitude'],
               c=significant_changes['bias_category'].map({'newly_emerged': '#FF6B6B', 'disappeared': '#FFA94D'}),
               alpha=0.7, s=60)
    
    # Add diagonal line
    ax4.plot([-0.5, 0.5], [-0.5, 0.5], 'k--', alpha=0.5)
    ax4.set_xlabel('Pre-Brexit Magnitude', fontsize=12)
    ax4.set_ylabel('Post-Brexit Magnitude', fontsize=12)
    ax4.set_title('Age Bias Magnitude: Pre vs Post Brexit', fontsize=16, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add legend
    handles = [plt.scatter([], [], c='#FF6B6B', label='Newly Emerged'),
              plt.scatter([], [], c='#FFA94D', label='Disappeared')]
    ax4.legend(handles=handles)
    
    plt.tight_layout()
    plt.savefig('../outputs/bias_vector_drift/age_bias_analysis/age_bias_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_topic_age_heatmap(matrices):
    """Create comprehensive topic × age heatmap showing bias emergence/disappearance"""
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 12))
    
    bias_types = ['newly_emerged', 'disappeared', 'persistent']
    titles = ['🔴 Newly Emerged Age Bias', '🟡 Disappeared Age Bias', '🔵 Persistent Age Bias']
    cmaps = ['Reds', 'Oranges', 'Blues']
    
    for i, (bias_type, title, cmap) in enumerate(zip(bias_types, titles, cmaps)):
        matrix = matrices[bias_type]
        
        # Create heatmap
        sns.heatmap(matrix['data'], 
                    xticklabels=matrix['age_pairs'],
                    yticklabels=matrix['topics'],
                    annot=True, 
                    fmt='d', 
                    cmap=cmap,
                    ax=axes[i],
                    cbar_kws={'label': 'Number of Cases'})
        
        axes[i].set_title(title, fontsize=16, fontweight='bold')
        axes[i].set_xlabel('Age Comparisons', fontsize=12)
        if i == 0:
            axes[i].set_ylabel('Asylum Topics', fontsize=12)
        
        # Rotate x labels for readability
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig('../outputs/bias_vector_drift/age_bias_analysis/topic_age_bias_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_age_vulnerability_analysis(vulnerability_df):
    """Plot age vulnerability patterns"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # 1. Net disadvantage by age
    ages = vulnerability_df['age']
    net_disadvantage = vulnerability_df['net_disadvantage']
    
    colors = ['#FF6B6B' if x > 0 else '#4D96FF' if x < 0 else '#808080' for x in net_disadvantage]
    
    bars = ax1.bar(ages, net_disadvantage, color=colors, alpha=0.8)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_title('Age Vulnerability: Net Disadvantage Score', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Age', fontsize=12)
    ax1.set_ylabel('Net Disadvantage Score\n(Positive = More Disadvantaged)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, net_disadvantage):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.1),
                f'{value}', ha='center', va='bottom' if height >= 0 else 'top', 
                fontweight='bold')
    
    # 2. Total involvement in bias patterns
    total_involvement = vulnerability_df['total_involvement']
    disadvantaged = vulnerability_df['disadvantaged_count']
    advantaged = vulnerability_df['advantaged_count']
    
    width = 0.8
    ax2.bar(ages, disadvantaged, width, label='Times Disadvantaged', color='#FF6B6B', alpha=0.8)
    ax2.bar(ages, advantaged, width, bottom=disadvantaged, label='Times Advantaged', color='#4D96FF', alpha=0.8)
    
    ax2.set_title('Age Group Involvement in Bias Patterns', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Age', fontsize=12)
    ax2.set_ylabel('Count of Bias Involvements', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add total labels
    for i, (age, total) in enumerate(zip(ages, total_involvement)):
        if total > 0:
            ax2.text(age, total + 0.2, f'Total: {total}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('../outputs/bias_vector_drift/age_bias_analysis/age_vulnerability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def analyze_topic_specific_age_patterns(df):
    """Analyze age patterns for key topics"""
    
    # Focus on topics with most age bias changes
    topic_counts = df[df['bias_category'] != 'stable']['topic'].value_counts()
    top_topics = topic_counts.head(6).index.tolist()
    
    analysis_results = {}
    
    for topic in top_topics:
        topic_df = df[(df['topic'] == topic) & (df['bias_category'] != 'stable')]
        
        # Age involvement analysis
        age_involvement = defaultdict(int)
        bias_direction = defaultdict(list)
        
        for _, row in topic_df.iterrows():
            age1, age2 = row['age1'], row['age2']
            magnitude = row['post_brexit_sp_magnitude']
            bias_type = row['bias_category']
            
            age_involvement[age1] += 1
            age_involvement[age2] += 1
            
            # Determine bias direction
            if magnitude > 0:
                bias_direction[age1].append(f"disadvantaged_vs_{age2}_{bias_type}")
                bias_direction[age2].append(f"advantaged_vs_{age1}_{bias_type}")
            else:
                bias_direction[age2].append(f"disadvantaged_vs_{age1}_{bias_type}")
                bias_direction[age1].append(f"advantaged_vs_{age2}_{bias_type}")
        
        analysis_results[topic] = {
            'age_involvement': dict(age_involvement),
            'bias_patterns': dict(bias_direction),
            'total_changes': len(topic_df)
        }
    
    return analysis_results, top_topics

def generate_insights_report(df, vulnerability_df, topic_analysis, top_topics):
    """Generate comprehensive insights report"""
    
    print("\n" + "="*80)
    print("🔍 AGE BIAS ANALYSIS: COMPREHENSIVE INSIGHTS REPORT")
    print("="*80)
    
    # Overall statistics
    total_age_comparisons = len(df)
    newly_emerged = len(df[df['bias_category'] == 'newly_emerged'])
    disappeared = len(df[df['bias_category'] == 'disappeared'])
    persistent = len(df[df['bias_category'] == 'persistent'])
    
    print(f"\n📊 OVERALL AGE BIAS STATISTICS:")
    print(f"   • Total age comparisons: {total_age_comparisons}")
    print(f"   • Newly emerged bias: {newly_emerged} cases")
    print(f"   • Disappeared bias: {disappeared} cases")
    print(f"   • Persistent bias: {persistent} cases")
    print(f"   • Net change: {newly_emerged - disappeared:+d} bias cases")
    print(f"   • Bias volatility: {((newly_emerged + disappeared) / total_age_comparisons * 100):.1f}%")
    
    # Age vulnerability insights
    print(f"\n🎯 AGE VULNERABILITY INSIGHTS:")
    most_disadvantaged = vulnerability_df.loc[vulnerability_df['net_disadvantage'].idxmax()]
    most_advantaged = vulnerability_df.loc[vulnerability_df['net_disadvantage'].idxmin()]
    most_involved = vulnerability_df.loc[vulnerability_df['total_involvement'].idxmax()]
    
    print(f"   • Most disadvantaged age: {int(most_disadvantaged['age'])} (net disadvantage: {most_disadvantaged['net_disadvantage']:+d})")
    print(f"   • Most advantaged age: {int(most_advantaged['age'])} (net disadvantage: {most_advantaged['net_disadvantage']:+d})")
    print(f"   • Most involved in bias patterns: {int(most_involved['age'])} ({most_involved['total_involvement']} cases)")
    
    # Topic-specific insights
    print(f"\n📝 TOPIC-SPECIFIC AGE BIAS PATTERNS:")
    for i, topic in enumerate(top_topics[:4], 1):
        topic_data = topic_analysis[topic]
        print(f"\n   {i}. {topic}:")
        print(f"      • Total age bias changes: {topic_data['total_changes']}")
        
        # Most involved ages
        age_counts = topic_data['age_involvement']
        if age_counts:
            most_involved_age = max(age_counts.items(), key=lambda x: x[1])
            print(f"      • Most involved age: {most_involved_age[0]} ({most_involved_age[1]} cases)")
        
        # Bias patterns
        bias_patterns = topic_data['bias_patterns']
        if bias_patterns:
            for age, patterns in bias_patterns.items():
                if len(patterns) >= 2:  # Show ages with multiple bias involvements
                    print(f"      • Age {age}: {len(patterns)} bias involvements")
    
    # Critical findings
    print(f"\n🚨 CRITICAL FINDINGS:")
    
    # Check for systematic patterns
    children_bias = df[(df['age1'] == 12) | (df['age2'] == 12)]
    children_emerged = len(children_bias[children_bias['bias_category'] == 'newly_emerged'])
    children_disappeared = len(children_bias[children_bias['bias_category'] == 'disappeared'])
    
    elderly_bias = df[(df['age1'] == 70) | (df['age2'] == 70)]
    elderly_emerged = len(elderly_bias[elderly_bias['bias_category'] == 'newly_emerged'])
    elderly_disappeared = len(elderly_bias[elderly_bias['bias_category'] == 'disappeared'])
    
    print(f"   • Children (age 12): {children_emerged} newly emerged, {children_disappeared} disappeared bias cases")
    print(f"   • Elderly (age 70): {elderly_emerged} newly emerged, {elderly_disappeared} disappeared bias cases")
    
    if newly_emerged > disappeared:
        print(f"   • ⚠️  NET INCREASE in age bias: {newly_emerged - disappeared} more discriminatory patterns post-Brexit")
    elif disappeared > newly_emerged:
        print(f"   • ✅ Net decrease in age bias: {disappeared - newly_emerged} fewer discriminatory patterns post-Brexit")
    else:
        print(f"   • ↔️  Neutral change: Equal emergence and disappearance of age bias")
    
    print(f"\n💡 RESEARCH RECOMMENDATIONS:")
    print(f"   • Investigate systematic disadvantage of age {int(most_disadvantaged['age'])}")
    if len(top_topics) > 0:
        print(f"   • Examine why {top_topics[0]} shows most age bias changes")
    print(f"   • Analyze intersectional effects with other protected attributes")
    print(f"   • Study age-based credibility assessments in asylum decisions")
    
    return {
        'total_comparisons': total_age_comparisons,
        'newly_emerged': newly_emerged,
        'disappeared': disappeared,
        'persistent': persistent,
        'net_change': newly_emerged - disappeared,
        'most_disadvantaged_age': int(most_disadvantaged['age']),
        'most_involved_age': int(most_involved['age']),
        'top_affected_topic': top_topics[0] if len(top_topics) > 0 else None
    }

def main():
    """Main analysis execution"""
    
    print("🔍 Starting Comprehensive Age Bias Analysis...")
    print("=" * 60)
    
    # Load and prepare data
    df = load_age_bias_data()
    df = categorize_bias_changes(df)
    
    # Core analyses
    age_pattern_df, age_categories = analyze_age_range_patterns(df)
    matrices = analyze_topic_age_matrix(df)
    vulnerability_df = create_age_vulnerability_analysis(df)
    topic_analysis, top_topics = analyze_topic_specific_age_patterns(df)
    
    # Generate visualizations
    print("\n📊 Creating visualizations...")
    
    fig1 = plot_age_bias_overview(df)
    fig2 = plot_topic_age_heatmap(matrices)
    fig3 = plot_age_vulnerability_analysis(vulnerability_df)
    
    # Generate insights report
    insights = generate_insights_report(df, vulnerability_df, topic_analysis, top_topics)
    
    # Save detailed analysis data
    print("\n💾 Saving analysis data...")
    
    output_dir = '../outputs/bias_vector_drift/age_bias_analysis'
    df.to_csv(f'{output_dir}/age_bias_detailed_analysis.csv', index=False)
    vulnerability_df.to_csv(f'{output_dir}/age_vulnerability_analysis.csv', index=False)
    
    # Save matrices data
    for bias_type, matrix_data in matrices.items():
        matrix_df = pd.DataFrame(
            matrix_data['data'], 
            index=matrix_data['topics'],
            columns=matrix_data['age_pairs']
        )
        matrix_df.to_csv(f'{output_dir}/topic_age_matrix_{bias_type}.csv')
    
    print("✅ Age bias analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    
    return df, vulnerability_df, matrices, insights

if __name__ == "__main__":
    main() 