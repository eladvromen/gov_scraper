#!/usr/bin/env python3
"""
Comprehensive Statistics Analysis for Unified Fairness Dataframe
Provides full statistics for ALL comparisons, not just samples
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_unified_dataframe():
    """Load the unified fairness dataframe"""
    df_path = Path("../../../outputs/unified_analysis/unified_fairness_dataframe_topic_granular.csv")
    return pd.read_csv(df_path)

def comprehensive_statistics(df):
    """Generate comprehensive statistics for all data"""
    
    print("=" * 100)
    print("COMPREHENSIVE STATISTICS FOR ALL FAIRNESS COMPARISONS (TOPIC GRANULAR)")
    print("=" * 100)
    
    # Basic info
    print(f"Total comparisons: {len(df)}")
    print(f"Protected attributes: {df['protected_attribute'].nunique()}")
    print(f"Unique topics: {df['topic'].nunique()}")
    print(f"Unique group comparisons: {df['group_comparison'].nunique()}")
    print(f"Complete records (all 3 metrics): {df[['pre_brexit_model_statistical_parity', 'post_brexit_model_statistical_parity', 'equal_opportunity_models_gap']].notna().all(axis=1).sum()}")
    
    # Attribute breakdown
    print("\n" + "=" * 80)
    print("BREAKDOWN BY PROTECTED ATTRIBUTE")
    print("=" * 80)
    
    attr_counts = df['protected_attribute'].value_counts()
    for attr, count in attr_counts.items():
        print(f"{attr:12}: {count:3d} comparisons")
    
    print("\nTopic breakdown:")
    topic_counts = df['topic'].value_counts()
    for topic, count in topic_counts.items():
        print(f"{topic[:40]:40}: {count:3d} comparisons")
    
    # Statistical Parity Analysis
    print("\n" + "=" * 80)
    print("STATISTICAL PARITY ANALYSIS (ALL COMPARISONS)")
    print("=" * 80)
    
    pre_brexit_sp = df['pre_brexit_model_statistical_parity'].dropna()
    post_brexit_sp = df['post_brexit_model_statistical_parity'].dropna()
    sp_difference = df['models_sp_difference'].dropna()
    
    print("Pre-Brexit Statistical Parity:")
    print(f"  Mean: {pre_brexit_sp.mean():.6f}")
    print(f"  Median: {pre_brexit_sp.median():.6f}")
    print(f"  Std Dev: {pre_brexit_sp.std():.6f}")
    print(f"  Min: {pre_brexit_sp.min():.6f}")
    print(f"  Max: {pre_brexit_sp.max():.6f}")
    print(f"  Range: {pre_brexit_sp.max() - pre_brexit_sp.min():.6f}")
    
    print("\nPost-Brexit Statistical Parity:")
    print(f"  Mean: {post_brexit_sp.mean():.6f}")
    print(f"  Median: {post_brexit_sp.median():.6f}")
    print(f"  Std Dev: {post_brexit_sp.std():.6f}")
    print(f"  Min: {post_brexit_sp.min():.6f}")
    print(f"  Max: {post_brexit_sp.max():.6f}")
    print(f"  Range: {post_brexit_sp.max() - post_brexit_sp.min():.6f}")
    
    print("\nModels Difference (Post - Pre):")
    print(f"  Mean: {sp_difference.mean():.6f}")
    print(f"  Median: {sp_difference.median():.6f}")
    print(f"  Std Dev: {sp_difference.std():.6f}")
    print(f"  Min: {sp_difference.min():.6f}")
    print(f"  Max: {sp_difference.max():.6f}")
    print(f"  Range: {sp_difference.max() - sp_difference.min():.6f}")
    
    # Equal Opportunity Analysis
    print("\n" + "=" * 80)
    print("EQUAL OPPORTUNITY ANALYSIS (ALL COMPARISONS)")
    print("=" * 80)
    
    eo_gap = df['equal_opportunity_models_gap'].dropna()
    
    print("Equal Opportunity Gap:")
    print(f"  Mean: {eo_gap.mean():.6f}")
    print(f"  Median: {eo_gap.median():.6f}")
    print(f"  Std Dev: {eo_gap.std():.6f}")
    print(f"  Min: {eo_gap.min():.6f}")
    print(f"  Max: {eo_gap.max():.6f}")
    print(f"  Range: {eo_gap.max() - eo_gap.min():.6f}")
    
    # Significance Analysis
    print("\n" + "=" * 80)
    print("SIGNIFICANCE ANALYSIS (ALL COMPARISONS)")
    print("=" * 80)
    
    pre_sig = df['pre_brexit_significant'].sum()
    post_sig = df['post_brexit_significant'].sum()
    both_sig = df['both_models_significant'].sum()
    sig_change = df['significance_change'].sum()
    
    print(f"Pre-Brexit significant: {pre_sig:3d} ({pre_sig/len(df)*100:.1f}%)")
    print(f"Post-Brexit significant: {post_sig:3d} ({post_sig/len(df)*100:.1f}%)")
    print(f"Both models significant: {both_sig:3d} ({both_sig/len(df)*100:.1f}%)")
    print(f"Significance changed: {sig_change:3d} ({sig_change/len(df)*100:.1f}%)")
    
    # Detailed breakdown by significance patterns
    print("\nSignificance Patterns:")
    sig_patterns = df.groupby(['pre_brexit_significant', 'post_brexit_significant']).size()
    for (pre, post), count in sig_patterns.items():
        pre_label = "Significant" if pre else "Not Significant"
        post_label = "Significant" if post else "Not Significant"
        print(f"  Pre: {pre_label:15} | Post: {post_label:15} | Count: {count:3d}")
    
    # Magnitude Analysis
    print("\n" + "=" * 80)
    print("MAGNITUDE ANALYSIS (ALL COMPARISONS)")
    print("=" * 80)
    
    magnitude_counts = df['sp_difference_magnitude'].value_counts()
    print("Statistical Parity Difference Magnitude:")
    for magnitude, count in magnitude_counts.items():
        print(f"  {magnitude:12}: {count:3d} comparisons ({count/len(df)*100:.1f}%)")
    
    # Attribute-specific Analysis
    print("\n" + "=" * 80)
    print("ATTRIBUTE-SPECIFIC ANALYSIS")
    print("=" * 80)
    
    for attr in df['protected_attribute'].unique():
        attr_data = df[df['protected_attribute'] == attr]
        print(f"\n{attr.upper()} ATTRIBUTE ({len(attr_data)} comparisons):")
        
        # Statistical Parity by attribute
        attr_sp_diff = attr_data['models_sp_difference'].dropna()
        print(f"  SP Difference - Mean: {attr_sp_diff.mean():.6f}, Std: {attr_sp_diff.std():.6f}")
        
        # Equal Opportunity by attribute
        attr_eo = attr_data['equal_opportunity_models_gap'].dropna()
        print(f"  EO Gap - Mean: {attr_eo.mean():.6f}, Std: {attr_eo.std():.6f}")
        
        # Significance by attribute
        attr_pre_sig = attr_data['pre_brexit_significant'].sum()
        attr_post_sig = attr_data['post_brexit_significant'].sum()
        print(f"  Significance - Pre: {attr_pre_sig}/{len(attr_data)} ({attr_pre_sig/len(attr_data)*100:.1f}%), Post: {attr_post_sig}/{len(attr_data)} ({attr_post_sig/len(attr_data)*100:.1f}%)")
    
    # Topic-specific Analysis
    print("\n" + "=" * 80)
    print("TOPIC-SPECIFIC ANALYSIS")
    print("=" * 80)
    
    # Show top 10 topics by number of comparisons
    print("Top 10 topics by comparison count:")
    top_topics = df['topic'].value_counts().head(10)
    for topic, count in top_topics.items():
        topic_data = df[df['topic'] == topic]
        
        # Statistical Parity by topic
        topic_sp_diff = topic_data['models_sp_difference'].dropna()
        sp_mean = topic_sp_diff.mean() if len(topic_sp_diff) > 0 else 0
        
        # Equal Opportunity by topic
        topic_eo = topic_data['equal_opportunity_models_gap'].dropna()
        eo_mean = topic_eo.mean() if len(topic_eo) > 0 else 0
        
        # Significance by topic
        topic_pre_sig = topic_data['pre_brexit_significant'].sum()
        topic_post_sig = topic_data['post_brexit_significant'].sum()
        
        print(f"\n{topic[:35]:35} ({count:3d} comparisons):")
        print(f"  SP Diff Mean: {sp_mean:+.4f} | EO Gap Mean: {eo_mean:+.4f}")
        print(f"  Significance - Pre: {topic_pre_sig:3d} ({topic_pre_sig/count*100:.1f}%) | Post: {topic_post_sig:3d} ({topic_post_sig/count*100:.1f}%)")
    
    # Extreme Values Analysis
    print("\n" + "=" * 80)
    print("EXTREME VALUES ANALYSIS")
    print("=" * 80)
    
    # Largest positive and negative SP differences
    print("Largest Statistical Parity Differences:")
    top_positive = df.nlargest(5, 'models_sp_difference')[['group_comparison', 'protected_attribute', 'models_sp_difference']]
    print("\nTop 5 Positive (Post-Brexit > Pre-Brexit):")
    for _, row in top_positive.iterrows():
        print(f"  {row['group_comparison']:25} | {row['protected_attribute']:10} | {row['models_sp_difference']:+.6f}")
    
    top_negative = df.nsmallest(5, 'models_sp_difference')[['group_comparison', 'protected_attribute', 'models_sp_difference']]
    print("\nTop 5 Negative (Post-Brexit < Pre-Brexit):")
    for _, row in top_negative.iterrows():
        print(f"  {row['group_comparison']:25} | {row['protected_attribute']:10} | {row['models_sp_difference']:+.6f}")
    
    # Largest Equal Opportunity gaps
    print("\nLargest Equal Opportunity Gaps:")
    top_eo_positive = df.nlargest(5, 'equal_opportunity_models_gap')[['group_comparison', 'protected_attribute', 'equal_opportunity_models_gap']]
    print("\nTop 5 Positive EO Gaps:")
    for _, row in top_eo_positive.iterrows():
        print(f"  {row['group_comparison']:25} | {row['protected_attribute']:10} | {row['equal_opportunity_models_gap']:+.6f}")
    
    top_eo_negative = df.nsmallest(5, 'equal_opportunity_models_gap')[['group_comparison', 'protected_attribute', 'equal_opportunity_models_gap']]
    print("\nTop 5 Negative EO Gaps:")
    for _, row in top_eo_negative.iterrows():
        print(f"  {row['group_comparison']:25} | {row['protected_attribute']:10} | {row['equal_opportunity_models_gap']:+.6f}")
    
    # Sample Size Analysis
    print("\n" + "=" * 80)
    print("SAMPLE SIZE ANALYSIS")
    print("=" * 80)
    
    pre_group_sizes = df['pre_brexit_group_size'].dropna()
    post_group_sizes = df['post_brexit_group_size'].dropna()
    eo_sample_sizes = df['equal_opportunity_sample_size'].dropna()
    
    print("Pre-Brexit Group Sizes:")
    print(f"  Mean: {pre_group_sizes.mean():.1f}")
    print(f"  Median: {pre_group_sizes.median():.1f}")
    print(f"  Min: {pre_group_sizes.min()}")
    print(f"  Max: {pre_group_sizes.max()}")
    
    print("\nPost-Brexit Group Sizes:")
    print(f"  Mean: {post_group_sizes.mean():.1f}")
    print(f"  Median: {post_group_sizes.median():.1f}")
    print(f"  Min: {post_group_sizes.min()}")
    print(f"  Max: {post_group_sizes.max()}")
    
    print("\nEqual Opportunity Sample Sizes:")
    print(f"  Mean: {eo_sample_sizes.mean():.1f}")
    print(f"  Median: {eo_sample_sizes.median():.1f}")
    print(f"  Min: {eo_sample_sizes.min()}")
    print(f"  Max: {eo_sample_sizes.max()}")
    
    # Full Data Display
    print("\n" + "=" * 80)
    print("FULL DATASET PREVIEW")
    print("=" * 80)
    
    print("Core columns for ALL comparisons:")
    core_cols = ['group_comparison', 'protected_attribute', 'topic',
                'pre_brexit_model_statistical_parity', 
                'post_brexit_model_statistical_parity',
                'equal_opportunity_models_gap',
                'models_sp_difference',
                'pre_brexit_significant', 
                'post_brexit_significant']
    
    print(df[core_cols].to_string(index=False))

def main():
    """Main function"""
    df = load_unified_dataframe()
    comprehensive_statistics(df)

if __name__ == "__main__":
    main() 