#!/usr/bin/env python3
"""
Nigeria Disparity Analysis: Pre-Brexit vs Post-Brexit
=====================================================

Empirical Question: "How did disparity outcomes for Nigeria change between 
the Pre- and Post-Brexit models?"

Sub-questions:
1. Was Nigeria's favorability present in both models or only one?
   (i.e., Is the mean SP +0.125 driven by both models or skewed by one?)

2. Did any of Nigeria's favorable disparities persist, or were they entirely 
   replaced or reversed post-Brexit?
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_nigeria_comparisons():
    """Load all comparisons involving Nigeria from deduplicated data"""
    print("Loading Nigeria-related comparisons...")
    
    data_path = "../../outputs/vector_drift/deduplicated_fairness_comparisons.csv"
    df = pd.read_csv(data_path)
    
    # Extract Nigeria comparisons from country attribute
    nigeria_data = []
    
    for _, row in df.iterrows():
        label = row['comparison_label']
        
        if '(country)' in label and 'Nigeria' in label:
            # Parse label components
            comparison_part = label.split(' (')[0]
            remaining = label.split(' (')[1] if ' (' in label else ""
            
            topic = ""
            if '[' in remaining and ']' in remaining:
                topic = remaining.split('[')[1].split(']')[0]
            
            parts = comparison_part.split('_vs_')
            if len(parts) == 2:
                country1, country2 = parts
                
                # Determine Nigeria's role and opponent
                if country1 == 'Nigeria':
                    nigeria_role = 'country1'
                    opponent = country2
                    # Positive SP means Nigeria is favored
                    pre_nigeria_sp = row['pre_brexit_sp_magnitude']
                    post_nigeria_sp = row['post_brexit_sp_magnitude']
                elif country2 == 'Nigeria':
                    nigeria_role = 'country2'
                    opponent = country1
                    # Negative SP means Nigeria is favored (since it's country2)
                    pre_nigeria_sp = -row['pre_brexit_sp_magnitude']
                    post_nigeria_sp = -row['post_brexit_sp_magnitude']
                else:
                    continue
                
                nigeria_data.append({
                    'comparison_label': label,
                    'nigeria_role': nigeria_role,
                    'opponent_country': opponent,
                    'topic': topic,
                    'pre_nigeria_sp': pre_nigeria_sp,
                    'post_nigeria_sp': post_nigeria_sp,
                    'sp_change': post_nigeria_sp - pre_nigeria_sp,
                    'pre_significant': row['pre_brexit_sp_significance'],
                    'post_significant': row['post_brexit_sp_significance'],
                    'fdr_significant': row['pre_brexit_sp_significance'] or row['post_brexit_sp_significance']
                })
    
    return pd.DataFrame(nigeria_data)

def analyze_nigeria_favorability_by_model(df):
    """
    Sub-question 1: Was Nigeria's favorability present in both models or only one?
    """
    print("\n" + "="*80)
    print("📊 SUB-QUESTION 1: Nigeria's Favorability by Model")
    print("="*80)
    
    # Overall statistics
    pre_brexit_mean = df['pre_nigeria_sp'].mean()
    post_brexit_mean = df['post_nigeria_sp'].mean()
    
    print(f"\n🔍 OVERALL FAVORABILITY STATISTICS:")
    print(f"   Pre-Brexit mean SP (Nigeria perspective): {pre_brexit_mean:+.3f}")
    print(f"   Post-Brexit mean SP (Nigeria perspective): {post_brexit_mean:+.3f}")
    print(f"   Change in favorability: {post_brexit_mean - pre_brexit_mean:+.3f}")
    
    # Test if means are significantly different from zero
    pre_t_stat, pre_p = stats.ttest_1samp(df['pre_nigeria_sp'], 0)
    post_t_stat, post_p = stats.ttest_1samp(df['post_nigeria_sp'], 0)
    
    print(f"\n📈 STATISTICAL SIGNIFICANCE TESTS (vs. zero bias):")
    print(f"   Pre-Brexit: t = {pre_t_stat:.3f}, p = {pre_p:.3f} {'✅ SIGNIFICANT' if pre_p < 0.05 else '❌ NOT SIGNIFICANT'}")
    print(f"   Post-Brexit: t = {post_t_stat:.3f}, p = {post_p:.3f} {'✅ SIGNIFICANT' if post_p < 0.05 else '❌ NOT SIGNIFICANT'}")
    
    # Test if the change between models is significant
    change_t_stat, change_p = stats.ttest_rel(df['post_nigeria_sp'], df['pre_nigeria_sp'])
    print(f"   Change Pre→Post: t = {change_t_stat:.3f}, p = {change_p:.3f} {'✅ SIGNIFICANT CHANGE' if change_p < 0.05 else '❌ NO SIGNIFICANT CHANGE'}")
    
    # Distribution analysis
    pre_positive = (df['pre_nigeria_sp'] > 0).sum()
    pre_negative = (df['pre_nigeria_sp'] < 0).sum()
    post_positive = (df['post_nigeria_sp'] > 0).sum()
    post_negative = (df['post_nigeria_sp'] < 0).sum()
    
    print(f"\n📊 FAVORABILITY DISTRIBUTION:")
    print(f"   Pre-Brexit: {pre_positive} favorable, {pre_negative} unfavorable ({pre_positive/len(df)*100:.1f}% favorable)")
    print(f"   Post-Brexit: {post_positive} favorable, {post_negative} unfavorable ({post_positive/len(df)*100:.1f}% favorable)")
    
    # Answer to sub-question 1
    print(f"\n🎯 ANSWER TO SUB-QUESTION 1:")
    if abs(pre_brexit_mean) > abs(post_brexit_mean):
        print(f"   Nigeria's overall favorability (+0.125) is PRIMARILY driven by the PRE-Brexit model")
        print(f"   The systematic favorability appears to be WEAKENING post-Brexit")
    elif abs(post_brexit_mean) > abs(pre_brexit_mean):
        print(f"   Nigeria's overall favorability (+0.125) is PRIMARILY driven by the POST-Brexit model")
        print(f"   The systematic favorability appears to be STRENGTHENING post-Brexit")
    else:
        print(f"   Nigeria's favorability is EQUALLY present in both models")
    
    return {
        'pre_mean': pre_brexit_mean,
        'post_mean': post_brexit_mean,
        'pre_p_value': pre_p,
        'post_p_value': post_p,
        'change_p_value': change_p
    }

def analyze_pattern_persistence(df):
    """
    Sub-question 2: Did Nigeria's favorable disparities persist or were they replaced/reversed?
    """
    print("\n" + "="*80)
    print("📊 SUB-QUESTION 2: Persistence vs Replacement of Favorable Disparities")
    print("="*80)
    
    # Categorize each comparison by significance pattern
    def get_pattern_type(row):
        pre_sig = row['pre_significant']
        post_sig = row['post_significant']
        pre_favorable = row['pre_nigeria_sp'] > 0
        post_favorable = row['post_nigeria_sp'] > 0
        
        if pre_sig and post_sig:
            if pre_favorable and post_favorable:
                return "PERSISTENT_FAVORABLE"
            elif not pre_favorable and not post_favorable:
                return "PERSISTENT_UNFAVORABLE"
            else:
                return "REVERSED_DIRECTION"
        elif pre_sig and not post_sig:
            return "DISAPPEARED_PRE_FAVORABLE" if pre_favorable else "DISAPPEARED_PRE_UNFAVORABLE"
        elif not pre_sig and post_sig:
            return "EMERGED_POST_FAVORABLE" if post_favorable else "EMERGED_POST_UNFAVORABLE"
        else:
            return "NON_SIGNIFICANT"
    
    df['pattern_type'] = df.apply(get_pattern_type, axis=1)
    
    # Count pattern types
    pattern_counts = df['pattern_type'].value_counts()
    
    print(f"\n🔍 PATTERN CLASSIFICATION:")
    for pattern, count in pattern_counts.items():
        percentage = count / len(df) * 100
        print(f"   {pattern:25} | {count:2d} comparisons ({percentage:.1f}%)")
    
    # Focus on favorable patterns
    favorable_patterns = df[df['pattern_type'].str.contains('FAVORABLE')]
    
    print(f"\n📈 FAVORABLE DISPARITY ANALYSIS:")
    
    # Persistent favorable
    persistent_favorable = df[df['pattern_type'] == 'PERSISTENT_FAVORABLE']
    print(f"   Persistent favorable disparities: {len(persistent_favorable)}")
    if len(persistent_favorable) > 0:
        print(f"   Topics: {', '.join(persistent_favorable['topic'].tolist())}")
        print(f"   Opponents: {', '.join(persistent_favorable['opponent_country'].tolist())}")
    
    # Disappeared favorable
    disappeared_favorable = df[df['pattern_type'] == 'DISAPPEARED_PRE_FAVORABLE']
    print(f"   Disappeared favorable disparities: {len(disappeared_favorable)}")
    if len(disappeared_favorable) > 0:
        print(f"   Topics: {', '.join(disappeared_favorable['topic'].tolist())}")
        print(f"   Opponents: {', '.join(disappeared_favorable['opponent_country'].tolist())}")
    
    # Emerged favorable
    emerged_favorable = df[df['pattern_type'] == 'EMERGED_POST_FAVORABLE']
    print(f"   Newly emerged favorable disparities: {len(emerged_favorable)}")
    if len(emerged_favorable) > 0:
        print(f"   Topics: {', '.join(emerged_favorable['topic'].tolist())}")
        print(f"   Opponents: {', '.join(emerged_favorable['opponent_country'].tolist())}")
    
    # Detailed breakdown by topic
    print(f"\n📋 TOPIC-SPECIFIC ANALYSIS:")
    topic_patterns = df.groupby(['topic', 'pattern_type']).size().unstack(fill_value=0)
    for topic in topic_patterns.index:
        print(f"\n   {topic}:")
        for pattern in topic_patterns.columns:
            count = topic_patterns.loc[topic, pattern]
            if count > 0:
                print(f"     {pattern}: {count}")
    
    # Answer to sub-question 2
    print(f"\n🎯 ANSWER TO SUB-QUESTION 2:")
    
    total_pre_favorable = len(df[(df['pre_significant']) & (df['pre_nigeria_sp'] > 0)])
    persistent_count = len(persistent_favorable)
    disappeared_count = len(disappeared_favorable)
    
    if persistent_count > disappeared_count:
        print(f"   Nigeria's favorable disparities MOSTLY PERSISTED ({persistent_count} persistent vs {disappeared_count} disappeared)")
    elif disappeared_count > persistent_count:
        print(f"   Nigeria's favorable disparities were MOSTLY REPLACED/LOST ({disappeared_count} disappeared vs {persistent_count} persistent)")
    else:
        print(f"   Nigeria's favorable disparities show MIXED patterns (equal persistence and disappearance)")
    
    return pattern_counts

def create_visualization(df, stats_results):
    """Create visualization of Nigeria's bias patterns"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Nigeria Disparity Analysis: Pre-Brexit vs Post-Brexit', fontsize=16, fontweight='bold')
    
    # 1. SP values comparison
    ax1.scatter(df['pre_nigeria_sp'], df['post_nigeria_sp'], alpha=0.7, s=60)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax1.plot([-0.5, 0.5], [-0.5, 0.5], 'k--', alpha=0.3, label='No Change Line')
    ax1.set_xlabel('Pre-Brexit SP (Nigeria perspective)')
    ax1.set_ylabel('Post-Brexit SP (Nigeria perspective)')
    ax1.set_title('SP Values: Pre vs Post-Brexit')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Mean comparison
    means = [stats_results['pre_mean'], stats_results['post_mean']]
    models = ['Pre-Brexit', 'Post-Brexit']
    colors = ['lightblue', 'lightcoral']
    bars = ax2.bar(models, means, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Mean SP (Nigeria perspective)')
    ax2.set_title('Mean Favorability by Model')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005 if height >= 0 else height - 0.015,
                f'{mean:+.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    # 3. Pattern distribution
    pattern_counts = df['pattern_type'].value_counts()
    ax3.pie(pattern_counts.values, labels=pattern_counts.index, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Distribution of Bias Patterns')
    
    # 4. Topic-specific changes
    topic_changes = df.groupby('topic')['sp_change'].mean().sort_values()
    colors_topic = ['green' if x > 0 else 'red' for x in topic_changes.values]
    ax4.barh(range(len(topic_changes)), topic_changes.values, color=colors_topic, alpha=0.7)
    ax4.set_yticks(range(len(topic_changes)))
    ax4.set_yticklabels([t[:30] + '...' if len(t) > 33 else t for t in topic_changes.index])
    ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax4.set_xlabel('Mean SP Change (Post - Pre)')
    ax4.set_title('Average Change by Topic')
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("../../outputs/country_analysis")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "nigeria_disparity_analysis.png", dpi=300, bbox_inches='tight')
    
    return fig

def export_results(df, stats_results, pattern_counts):
    """Export detailed results to files"""
    
    output_dir = Path("../../outputs/country_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Export Nigeria comparison data
    df.to_csv(output_dir / "nigeria_all_comparisons.csv", index=False)
    
    # Export summary statistics
    summary = {
        'pre_brexit_mean_sp': stats_results['pre_mean'],
        'post_brexit_mean_sp': stats_results['post_mean'],
        'sp_change': stats_results['post_mean'] - stats_results['pre_mean'],
        'pre_brexit_significance': stats_results['pre_p_value'],
        'post_brexit_significance': stats_results['post_p_value'],
        'change_significance': stats_results['change_p_value'],
        'total_comparisons': len(df),
        'fdr_significant_comparisons': df['fdr_significant'].sum()
    }
    
    with open(output_dir / "nigeria_analysis_summary.json", 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Results exported to:")
    print(f"   • nigeria_all_comparisons.csv")
    print(f"   • nigeria_analysis_summary.json")
    print(f"   • nigeria_disparity_analysis.png")

def main():
    """Main analysis function"""
    print("🇳🇬 NIGERIA DISPARITY ANALYSIS: Pre-Brexit vs Post-Brexit")
    print("="*80)
    print("Empirical Question: How did disparity outcomes for Nigeria change between models?")
    print("="*80)
    
    # Load data
    df = load_nigeria_comparisons()
    print(f"\nLoaded {len(df)} comparisons involving Nigeria")
    print(f"FDR-significant comparisons: {df['fdr_significant'].sum()}")
    
    # Analyze favorability by model (Sub-question 1)
    stats_results = analyze_nigeria_favorability_by_model(df)
    
    # Analyze pattern persistence (Sub-question 2)
    pattern_counts = analyze_pattern_persistence(df)
    
    # Create visualization
    fig = create_visualization(df, stats_results)
    
    # Export results
    export_results(df, stats_results, pattern_counts)
    
    # Final summary
    print(f"\n" + "="*80)
    print("🎯 FINAL CONCLUSIONS")
    print("="*80)
    
    pre_mean = stats_results['pre_mean']
    post_mean = stats_results['post_mean']
    
    print(f"\n1. Nigeria's +0.125 systematic favorability is:")
    if abs(pre_mean) > abs(post_mean):
        print(f"   PRIMARILY driven by Pre-Brexit model ({pre_mean:+.3f} vs {post_mean:+.3f})")
    else:
        print(f"   MORE pronounced in Post-Brexit model ({post_mean:+.3f} vs {pre_mean:+.3f})")
    
    favorable_disappeared = (df['pattern_type'] == 'DISAPPEARED_PRE_FAVORABLE').sum()
    favorable_persistent = (df['pattern_type'] == 'PERSISTENT_FAVORABLE').sum()
    
    print(f"\n2. Nigeria's favorable disparities:")
    if favorable_disappeared > favorable_persistent:
        print(f"   Were MOSTLY LOST post-Brexit ({favorable_disappeared} disappeared vs {favorable_persistent} persistent)")
    elif favorable_persistent > favorable_disappeared:
        print(f"   MOSTLY PERSISTED post-Brexit ({favorable_persistent} persistent vs {favorable_disappeared} disappeared)")
    else:
        print(f"   Show MIXED patterns (equal persistence and loss)")

if __name__ == "__main__":
    main() 