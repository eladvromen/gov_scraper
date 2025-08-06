#!/usr/bin/env python3
"""
Add FDR-Corrected Significance to Fairness DataFrame
==================================================

Simple script to:
1. Load unified fairness dataframe
2. Add FDR-corrected significance columns from robust vectors
3. Print statistically significant results with topic and bias measurements
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path

def load_fdr_corrected_vectors():
    """Load FDR-corrected significance vectors"""
    print("Loading FDR-corrected significance vectors...")
    
    vectors_path = "../../outputs/statistical_validity/robust_significance_vectors.json"
    
    with open(vectors_path, 'r') as f:
        data = json.load(f)
    
    vectors = data['vectors']
    
    return {
        'pre_brexit_fdr': np.array(vectors['pre_brexit_sp_significance_fdr']),
        'post_brexit_fdr': np.array(vectors['post_brexit_sp_significance_fdr']),
        'both_significant_fdr': np.array(vectors['sp_both_significant_fdr']),
        'gained_significance_fdr': np.array(vectors['sp_gained_significance_fdr']),
        'lost_significance_fdr': np.array(vectors['sp_lost_significance_fdr']),
        'either_significant_fdr': np.array(vectors['sp_robust_either_significant'])
    }

def add_fdr_significance_columns(df, fdr_vectors):
    """Add FDR-corrected significance columns to dataframe"""
    print("Adding FDR-corrected significance columns...")
    
    # Verify lengths match
    if len(df) != len(fdr_vectors['pre_brexit_fdr']):
        raise ValueError(f"DataFrame length ({len(df)}) doesn't match FDR vector length ({len(fdr_vectors['pre_brexit_fdr'])})")
    
    # Add FDR-corrected columns
    df['pre_brexit_sp_significance_fdr'] = fdr_vectors['pre_brexit_fdr']
    df['post_brexit_sp_significance_fdr'] = fdr_vectors['post_brexit_fdr']
    df['both_significant_fdr'] = fdr_vectors['both_significant_fdr']
    df['gained_significance_fdr'] = fdr_vectors['gained_significance_fdr'] 
    df['lost_significance_fdr'] = fdr_vectors['lost_significance_fdr']
    df['corrected_significance'] = fdr_vectors['either_significant_fdr']  # Main column requested
    
    return df

def print_significant_comparisons(df):
    """Print FDR-significant comparisons with topic and bias measurements"""
    
    # Filter for FDR-significant comparisons
    significant_df = df[df['corrected_significance'] == True].copy()
    
    print(f"\n{'='*80}")
    print(f"FDR-CORRECTED SIGNIFICANT COMPARISONS")
    print(f"{'='*80}")
    print(f"Total FDR-significant comparisons: {len(significant_df)} out of {len(df)} ({len(significant_df)/len(df):.1%})")
    
    if len(significant_df) == 0:
        print("No comparisons remain significant after FDR correction!")
        return
    
    # Group by protected attribute
    print(f"\n📊 BREAKDOWN BY PROTECTED ATTRIBUTE:")
    attr_counts = significant_df.groupby('protected_attribute').size().sort_values(ascending=False)
    for attr, count in attr_counts.items():
        pct = count / len(significant_df) * 100
        print(f"   {attr:12s}: {count:3d} comparisons ({pct:5.1f}%)")
    
    # Group by topic  
    print(f"\n📋 BREAKDOWN BY TOPIC:")
    topic_counts = significant_df.groupby('topic').size().sort_values(ascending=False)
    for topic, count in topic_counts.head(10).items():
        pct = count / len(significant_df) * 100
        topic_short = topic[:50] + "..." if len(topic) > 50 else topic
        print(f"   {topic_short:53s}: {count:3d} ({pct:5.1f}%)")
    
    # Show detailed comparisons
    print(f"\n📝 DETAILED SIGNIFICANT COMPARISONS:")
    print(f"{'='*120}")
    print(f"{'Comparison':<25} | {'Attribute':<8} | {'Topic':<35} | {'Pre-SP':>8} | {'Post-SP':>8} | {'SP-Δ':>8} | {'Significance Pattern':<20}")
    print(f"{'-'*120}")
    
    # Sort by absolute bias change for most interesting results
    significant_df['abs_sp_change'] = abs(significant_df['models_sp_difference'])
    significant_df_sorted = significant_df.sort_values('abs_sp_change', ascending=False)
    
    for _, row in significant_df_sorted.iterrows():
        comparison = row['group_comparison'][:24]
        attribute = row['protected_attribute'][:7]
        topic = row['topic'][:34]
        pre_sp = row['pre_brexit_model_statistical_parity']
        post_sp = row['post_brexit_model_statistical_parity']
        sp_change = row['models_sp_difference']
        
        # Determine significance pattern
        if row['both_significant_fdr']:
            sig_pattern = "Both Significant"
        elif row['gained_significance_fdr']:
            sig_pattern = "Newly Emerged"
        elif row['lost_significance_fdr']:
            sig_pattern = "Disappeared"
        elif row['pre_brexit_sp_significance_fdr']:
            sig_pattern = "Pre-Brexit Only"
        elif row['post_brexit_sp_significance_fdr']:
            sig_pattern = "Post-Brexit Only"
        else:
            sig_pattern = "Unknown"
        
        print(f"{comparison:<25} | {attribute:<8} | {topic:<35} | {pre_sp:>8.3f} | {post_sp:>8.3f} | {sp_change:>+8.3f} | {sig_pattern:<20}")

def compare_original_vs_fdr(df):
    """Compare original vs FDR-corrected significance rates"""
    print(f"\n{'='*80}")
    print(f"ORIGINAL vs FDR-CORRECTED COMPARISON")
    print(f"{'='*80}")
    
    # Original significance
    orig_pre = df['pre_brexit_sp_significance'].fillna(False).sum()
    orig_post = df['post_brexit_sp_significance'].fillna(False).sum()
    orig_either = ((df['pre_brexit_sp_significance'].fillna(False)) | 
                   (df['post_brexit_sp_significance'].fillna(False))).sum()
    
    # FDR-corrected significance
    fdr_pre = df['pre_brexit_sp_significance_fdr'].sum()
    fdr_post = df['post_brexit_sp_significance_fdr'].sum()
    fdr_either = df['corrected_significance'].sum()
    
    total = len(df)
    
    print(f"📊 PRE-BREXIT MODEL:")
    print(f"   Original significant:     {orig_pre:4d} ({orig_pre/total:5.1%})")
    print(f"   FDR-corrected significant: {fdr_pre:4d} ({fdr_pre/total:5.1%})")
    print(f"   Reduction:                {orig_pre - fdr_pre:4d} ({(orig_pre - fdr_pre)/orig_pre:5.1%} of original)")
    
    print(f"\n📊 POST-BREXIT MODEL:")
    print(f"   Original significant:     {orig_post:4d} ({orig_post/total:5.1%})")
    print(f"   FDR-corrected significant: {fdr_post:4d} ({fdr_post/total:5.1%})")
    print(f"   Reduction:                {orig_post - fdr_post:4d} ({(orig_post - fdr_post)/orig_post:5.1%} of original)")
    
    print(f"\n📊 EITHER MODEL SIGNIFICANT:")
    print(f"   Original significant:     {orig_either:4d} ({orig_either/total:5.1%})")
    print(f"   FDR-corrected significant: {fdr_either:4d} ({fdr_either/total:5.1%})")
    print(f"   Reduction:                {orig_either - fdr_either:4d} ({(orig_either - fdr_either)/orig_either:5.1%} of original)")

def main():
    """Main execution function"""
    
    print("🎯 ADDING FDR-CORRECTED SIGNIFICANCE TO FAIRNESS DATAFRAME")
    print("="*80)
    
    # Load unified fairness dataframe
    df_path = "../../outputs/unified/unified_fairness_dataframe_topic_granular.csv"
    print(f"Loading fairness dataframe from: {df_path}")
    df = pd.read_csv(df_path)
    print(f"Loaded {len(df)} comparisons")
    
    # Load FDR-corrected vectors
    fdr_vectors = load_fdr_corrected_vectors()
    print(f"Loaded FDR vectors with {len(fdr_vectors['pre_brexit_fdr'])} elements")
    
    # Add FDR significance columns
    df = add_fdr_significance_columns(df, fdr_vectors)
    
    # Compare original vs FDR rates
    compare_original_vs_fdr(df)
    
    # Print significant comparisons
    print_significant_comparisons(df)
    
    # Save updated dataframe
    output_path = "../../outputs/unified/unified_fairness_dataframe_topic_granular_with_fdr.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Updated dataframe saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    result_df = main() 