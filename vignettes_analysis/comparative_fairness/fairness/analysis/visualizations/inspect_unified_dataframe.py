#!/usr/bin/env python3
"""
Inspect and Analyze Unified Fairness DataFrame
Provides detailed analysis of the unified fairness dataframe created.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def load_unified_dataframe():
    """Load the unified fairness dataframe"""
    
    df_path = Path("../../../outputs/unified_analysis/unified_fairness_dataframe.csv")
    
    if not df_path.exists():
        raise FileNotFoundError("Unified dataframe not found. Run create_unified_fairness_dataframe.py first.")
    
    df = pd.read_csv(df_path)
    return df

def create_user_requested_format(df: pd.DataFrame) -> pd.DataFrame:
    """Create the exact format requested by the user"""
    
    # Select only the core columns requested
    core_columns = [
        'group_comparison',
        'protected_attribute',
        'pre_brexit_model_statistical_parity',
        'post_brexit_model_statistical_parity',
        'equal_opportunity_models_gap'
    ]
    
    # Create the user-requested format
    user_df = df[core_columns].copy()
    
    # Add metadata columns
    user_df['pre_brexit_group_size'] = df['pre_brexit_group_size']
    user_df['post_brexit_group_size'] = df['post_brexit_group_size']
    user_df['pre_brexit_significance'] = df['pre_brexit_sp_significance']
    user_df['post_brexit_significance'] = df['post_brexit_sp_significance']
    user_df['equal_opportunity_sample_size'] = df['equal_opportunity_sample_size']
    
    return user_df

def analyze_raw_vectors(df: pd.DataFrame):
    """Analyze the raw vectors for model comparison as requested"""
    
    print("=" * 80)
    print("RAW VECTORS ANALYSIS")
    print("=" * 80)
    
    # Pre-Brexit Statistical Parity Vector
    pre_brexit_sp = df['pre_brexit_model_statistical_parity'].dropna()
    print(f"Pre-Brexit Statistical Parity Vector (n={len(pre_brexit_sp)}):")
    print(f"  Mean: {pre_brexit_sp.mean():.4f}")
    print(f"  Std:  {pre_brexit_sp.std():.4f}")
    print(f"  Min:  {pre_brexit_sp.min():.4f}")
    print(f"  Max:  {pre_brexit_sp.max():.4f}")
    print(f"  Values: {pre_brexit_sp.values}")
    
    # Post-Brexit Statistical Parity Vector
    post_brexit_sp = df['post_brexit_model_statistical_parity'].dropna()
    print(f"\nPost-Brexit Statistical Parity Vector (n={len(post_brexit_sp)}):")
    print(f"  Mean: {post_brexit_sp.mean():.4f}")
    print(f"  Std:  {post_brexit_sp.std():.4f}")
    print(f"  Min:  {post_brexit_sp.min():.4f}")
    print(f"  Max:  {post_brexit_sp.max():.4f}")
    print(f"  Values: {post_brexit_sp.values}")
    
    # Equal Opportunity Gap Vector
    eo_gap = df['equal_opportunity_models_gap'].dropna()
    print(f"\nEqual Opportunity Gap Vector (n={len(eo_gap)}):")
    print(f"  Mean: {eo_gap.mean():.4f}")
    print(f"  Std:  {eo_gap.std():.4f}")
    print(f"  Min:  {eo_gap.min():.4f}")
    print(f"  Max:  {eo_gap.max():.4f}")
    print(f"  Values: {eo_gap.values}")

def analyze_significance_vectors(df: pd.DataFrame):
    """Analyze significance vectors for model comparison"""
    
    print("\n" + "=" * 80)
    print("SIGNIFICANCE VECTORS ANALYSIS")
    print("=" * 80)
    
    # Pre-Brexit Significance Vector
    pre_brexit_sig = df['pre_brexit_sp_significance'].fillna(False)
    print(f"Pre-Brexit Significance Vector (n={len(pre_brexit_sig)}):")
    print(f"  Significant count: {pre_brexit_sig.sum()}")
    print(f"  Significance rate: {pre_brexit_sig.mean():.2%}")
    print(f"  Values: {pre_brexit_sig.values}")
    
    # Post-Brexit Significance Vector
    post_brexit_sig = df['post_brexit_sp_significance'].fillna(False)
    print(f"\nPost-Brexit Significance Vector (n={len(post_brexit_sig)}):")
    print(f"  Significant count: {post_brexit_sig.sum()}")
    print(f"  Significance rate: {post_brexit_sig.mean():.2%}")
    print(f"  Values: {post_brexit_sig.values}")
    
    # Significance Change Vector
    sig_change = pre_brexit_sig != post_brexit_sig
    print(f"\nSignificance Change Vector (n={len(sig_change)}):")
    print(f"  Changes count: {sig_change.sum()}")
    print(f"  Change rate: {sig_change.mean():.2%}")
    print(f"  Values: {sig_change.values}")

def analyze_comparative_vectors(df: pd.DataFrame):
    """Analyze comparative vectors for equal opportunity"""
    
    print("\n" + "=" * 80)
    print("COMPARATIVE VECTORS ANALYSIS")
    print("=" * 80)
    
    # Statistical Parity Difference Vector
    sp_diff = df['post_brexit_model_statistical_parity'] - df['pre_brexit_model_statistical_parity']
    sp_diff_clean = sp_diff.dropna()
    
    print(f"Statistical Parity Difference Vector (n={len(sp_diff_clean)}):")
    print(f"  Mean: {sp_diff_clean.mean():.4f}")
    print(f"  Std:  {sp_diff_clean.std():.4f}")
    print(f"  Min:  {sp_diff_clean.min():.4f}")
    print(f"  Max:  {sp_diff_clean.max():.4f}")
    print(f"  Values: {sp_diff_clean.values}")
    
    # Direction analysis
    positive_changes = (sp_diff_clean > 0).sum()
    negative_changes = (sp_diff_clean < 0).sum()
    no_changes = (sp_diff_clean == 0).sum()
    
    print(f"\nDirection Analysis:")
    print(f"  Positive changes (Post > Pre): {positive_changes}")
    print(f"  Negative changes (Post < Pre): {negative_changes}")
    print(f"  No changes (Post = Pre): {no_changes}")

def create_group_breakdown_analysis(df: pd.DataFrame):
    """Create detailed breakdown by protected attribute and group"""
    
    print("\n" + "=" * 80)
    print("GROUP BREAKDOWN ANALYSIS")
    print("=" * 80)
    
    for attribute in df['protected_attribute'].unique():
        if pd.isna(attribute):
            continue
            
        attr_df = df[df['protected_attribute'] == attribute]
        
        print(f"\n{attribute.upper()} ATTRIBUTE:")
        print(f"  Total comparisons: {len(attr_df)}")
        
        # Show each comparison
        for _, row in attr_df.iterrows():
            print(f"  {row['group_comparison']}:")
            print(f"    Pre-Brexit SP: {row['pre_brexit_model_statistical_parity']:.4f} (sig: {row['pre_brexit_sp_significance']})")
            print(f"    Post-Brexit SP: {row['post_brexit_model_statistical_parity']:.4f} (sig: {row['post_brexit_sp_significance']})")
            print(f"    Equal Opp Gap: {row['equal_opportunity_models_gap']:.4f}")
            print(f"    Sample sizes: Pre={row['pre_brexit_group_size']}, Post={row['post_brexit_group_size']}, EO={row['equal_opportunity_sample_size']}")

def create_conclusions_summary(df: pd.DataFrame):
    """Create the conclusions summary as requested by the user"""
    
    print("\n" + "=" * 80)
    print("CONCLUSIONS SUMMARY")
    print("=" * 80)
    
    # 1. Raw vectors comparison for models statistical parity
    print("1. RAW VECTORS COMPARISON FOR MODELS STATISTICAL PARITY:")
    
    pre_brexit_sp = df['pre_brexit_model_statistical_parity'].dropna()
    post_brexit_sp = df['post_brexit_model_statistical_parity'].dropna()
    
    print(f"   Pre-Brexit model shows {len(pre_brexit_sp)} statistical parity measurements")
    print(f"   Post-Brexit model shows {len(post_brexit_sp)} statistical parity measurements")
    print(f"   Average Pre-Brexit SP: {pre_brexit_sp.mean():.4f}")
    print(f"   Average Post-Brexit SP: {post_brexit_sp.mean():.4f}")
    print(f"   Difference (Post - Pre): {post_brexit_sp.mean() - pre_brexit_sp.mean():.4f}")
    
    # 2. Significance vector for model comparison
    print("\n2. SIGNIFICANCE VECTOR FOR MODEL COMPARISON:")
    
    pre_sig = df['pre_brexit_sp_significance'].fillna(False)
    post_sig = df['post_brexit_sp_significance'].fillna(False)
    
    print(f"   Pre-Brexit significant disparities: {pre_sig.sum()}/{len(pre_sig)} ({pre_sig.mean():.1%})")
    print(f"   Post-Brexit significant disparities: {post_sig.sum()}/{len(post_sig)} ({post_sig.mean():.1%})")
    print(f"   Both models significant: {(pre_sig & post_sig).sum()}")
    print(f"   Significance pattern changed: {(pre_sig != post_sig).sum()}")
    
    # 3. Comparative vector for equal opportunity gap
    print("\n3. COMPARATIVE VECTOR FOR EQUAL OPPORTUNITY GAP:")
    
    eo_gap = df['equal_opportunity_models_gap'].dropna()
    print(f"   Equal opportunity measurements: {len(eo_gap)}")
    print(f"   Average EO gap: {eo_gap.mean():.4f}")
    print(f"   Positive gaps (favoring protected group): {(eo_gap > 0).sum()}")
    print(f"   Negative gaps (favoring reference group): {(eo_gap < 0).sum()}")
    print(f"   Largest positive gap: {eo_gap.max():.4f}")
    print(f"   Largest negative gap: {eo_gap.min():.4f}")

def save_analysis_results(df: pd.DataFrame, user_df: pd.DataFrame):
    """Save analysis results to files"""
    
    output_dir = Path("../../../outputs/unified_analysis")
    
    # Save user-requested format
    user_df.to_csv(output_dir / "user_requested_format.csv", index=False)
    
    # Save vector analysis
    vectors = {
        'pre_brexit_statistical_parity': df['pre_brexit_model_statistical_parity'].dropna().tolist(),
        'post_brexit_statistical_parity': df['post_brexit_model_statistical_parity'].dropna().tolist(),
        'equal_opportunity_gap': df['equal_opportunity_models_gap'].dropna().tolist(),
        'pre_brexit_significance': df['pre_brexit_sp_significance'].fillna(False).tolist(),
        'post_brexit_significance': df['post_brexit_sp_significance'].fillna(False).tolist(),
        'statistical_parity_difference': (df['post_brexit_model_statistical_parity'] - df['pre_brexit_model_statistical_parity']).dropna().tolist()
    }
    
    with open(output_dir / "analysis_vectors.json", 'w') as f:
        json.dump(vectors, f, indent=2)
    
    print(f"\n✅ Files saved:")
    print(f"   - user_requested_format.csv")
    print(f"   - analysis_vectors.json")

def main():
    """Main analysis function"""
    
    print("=" * 80)
    print("UNIFIED FAIRNESS DATAFRAME INSPECTION")
    print("=" * 80)
    
    # Load dataframe
    df = load_unified_dataframe()
    
    # Create user-requested format
    user_df = create_user_requested_format(df)
    
    print(f"Loaded dataframe with {len(df)} records")
    print(f"User-requested format: {user_df.shape}")
    
    # Perform analyses
    analyze_raw_vectors(df)
    analyze_significance_vectors(df)
    analyze_comparative_vectors(df)
    create_group_breakdown_analysis(df)
    create_conclusions_summary(df)
    
    # Save results
    save_analysis_results(df, user_df)
    
    return df, user_df

if __name__ == "__main__":
    df, user_df = main() 