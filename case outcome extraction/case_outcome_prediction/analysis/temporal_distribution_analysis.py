#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
"""
Temporal Distribution Analysis - Before vs After 2017
====================================================

This script analyzes how case outcome distributions change over time by comparing
high-confidence predictions before 2017 vs 2017 and after.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def analyze_temporal_distribution():
    """Analyze temporal changes in case outcome distribution"""
    
    print("="*70)
    print("TEMPORAL ANALYSIS: HIGH CONFIDENCE PREDICTIONS")
    print("Time Period Comparison: Before 2017 vs 2017 and After")
    print("="*70)
    
    # Load high confidence predictions
    predictions_path = "../results/predictions/high_confidence_predictions.parquet"
    
    if not Path(predictions_path).exists():
        print(f"❌ High confidence predictions file not found: {predictions_path}")
        print("Please run the inference pipeline first (step 6)")
        return False
    
    df = pd.read_parquet(predictions_path)
    print(f"✓ Loaded {len(df):,} high confidence predictions")
    
    # Check if case_year column exists
    if 'case_year' not in df.columns:
        print("❌ case_year column not found in the data")
        print("Available columns:", list(df.columns))
        return False
    
    # Split data into two time periods
    before_2017 = df[df['case_year'] < 2017].copy()
    from_2017 = df[df['case_year'] >= 2017].copy()
    
    print(f"\n📊 Temporal Data Split:")
    print(f"  Before 2017: {len(before_2017):,} cases")
    print(f"  2017 and after: {len(from_2017):,} cases")
    
    if len(before_2017) == 0:
        print("❌ No cases found before 2017")
        return False
    
    if len(from_2017) == 0:
        print("❌ No cases found from 2017 onwards")
        return False
    
    # Get prediction distributions for both periods
    before_counts = before_2017['predicted_outcome'].value_counts()
    after_counts = from_2017['predicted_outcome'].value_counts()
    
    print(f"\n📈 Distribution Before 2017:")
    for outcome, count in before_counts.items():
        pct = count / len(before_2017) * 100
        print(f"  {outcome}: {count:,} cases ({pct:.1f}%)")
    
    print(f"\n📈 Distribution 2017 and After:")
    for outcome, count in after_counts.items():
        pct = count / len(from_2017) * 100
        print(f"  {outcome}: {count:,} cases ({pct:.1f}%)")
    
    return create_temporal_pie_charts(before_2017, from_2017, before_counts, after_counts)

def create_temporal_pie_charts(before_df, after_df, before_counts, after_counts):
    """Create side-by-side pie charts for temporal comparison"""
    
    # Set up the plot style
    plt.style.use('default')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Define consistent colors for outcomes
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    # Get all unique outcomes to ensure consistent colors
    all_outcomes = set(before_counts.index) | set(after_counts.index)
    color_map = {outcome: colors[i] for i, outcome in enumerate(sorted(all_outcomes))}
    
    # Create pie chart for Before 2017
    before_colors = [color_map[outcome] for outcome in before_counts.index]
    wedges1, texts1, autotexts1 = ax1.pie(
        before_counts.values,
        labels=before_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=before_colors,
        explode=[0.05] * len(before_counts),
        shadow=True,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    
    # Create pie chart for 2017 and After
    after_colors = [color_map[outcome] for outcome in after_counts.index]
    wedges2, texts2, autotexts2 = ax2.pie(
        after_counts.values,
        labels=after_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=after_colors,
        explode=[0.05] * len(after_counts),
        shadow=True,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    
    # Enhance appearance for both charts
    for autotexts in [autotexts1, autotexts2]:
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
    
    for texts in [texts1, texts2]:
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
    
    # Add titles
    ax1.set_title(
        f'Before 2017\n({len(before_df):,} high confidence cases)',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    
    ax2.set_title(
        f'2017 and After\n({len(after_df):,} high confidence cases)',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    
    # Add main title
    fig.suptitle(
        'Temporal Analysis: Case Outcome Distributions\n(High Confidence Predictions ≥80%)',
        fontsize=20,
        fontweight='bold',
        y=0.95
    )
    
    # Ensure pie charts are circular
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    
    # Add statistics below each chart
    before_stats = before_df['prediction_confidence'].describe()
    after_stats = after_df['prediction_confidence'].describe()
    
    before_stats_text = f"""Statistics Before 2017:
Cases: {len(before_df):,}
Mean Confidence: {before_stats['mean']:.3f}
Date Range: {before_df['case_year'].min()}-{before_df['case_year'].max()}"""
    
    after_stats_text = f"""Statistics 2017+:
Cases: {len(after_df):,}
Mean Confidence: {after_stats['mean']:.3f}
Date Range: {after_df['case_year'].min()}-{after_df['case_year'].max()}"""
    
    # Position statistics text below charts
    fig.text(0.25, 0.15, before_stats_text, fontsize=10, ha='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    fig.text(0.75, 0.15, after_stats_text, fontsize=10, ha='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.25)
    
    # Save the plot
    output_path = "../outputs/plots/temporal_distribution_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved temporal comparison: {output_path}")
    
    # Also save as PDF
    pdf_path = "../outputs/plots/temporal_distribution_comparison.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved PDF version: {pdf_path}")
    
    plt.show()
    
    return True

def create_difference_analysis(before_df, after_df):
    """Create detailed statistical comparison"""
    
    print(f"\n" + "="*70)
    print("DETAILED COMPARISON ANALYSIS")
    print("="*70)
    
    # Calculate percentage differences
    before_counts = before_df['predicted_outcome'].value_counts(normalize=True) * 100
    after_counts = after_df['predicted_outcome'].value_counts(normalize=True) * 100
    
    print("\nPercentage Distribution Comparison:")
    print("-" * 40)
    print(f"{'Outcome':<12} {'Before 2017':<12} {'2017+':<12} {'Difference':<12}")
    print("-" * 40)
    
    all_outcomes = set(before_counts.index) | set(after_counts.index)
    
    for outcome in sorted(all_outcomes):
        before_pct = before_counts.get(outcome, 0)
        after_pct = after_counts.get(outcome, 0)
        diff = after_pct - before_pct
        
        print(f"{outcome:<12} {before_pct:<12.1f} {after_pct:<12.1f} {diff:>+12.1f}")
    
    # Confidence comparison
    before_conf = before_df['prediction_confidence'].mean()
    after_conf = after_df['prediction_confidence'].mean()
    conf_diff = after_conf - before_conf
    
    print(f"\nConfidence Score Comparison:")
    print(f"  Before 2017: {before_conf:.4f}")
    print(f"  2017 and after: {after_conf:.4f}")
    print(f"  Difference: {conf_diff:>+.4f}")
    
    # Year range analysis
    print(f"\nTemporal Coverage:")
    print(f"  Before 2017: {before_df['case_year'].min()} - {before_df['case_year'].max()}")
    print(f"  2017 and after: {after_df['case_year'].min()} - {after_df['case_year'].max()}")
    
    return True

def main():
    """Main function to run temporal analysis"""
    print("🕒 TEMPORAL DISTRIBUTION ANALYSIS")
    print("="*70)
    
    # Load and analyze data
    if not analyze_temporal_distribution():
        return False
    
    # Load data again for detailed analysis
    predictions_path = "../results/predictions/high_confidence_predictions.parquet"
    df = pd.read_parquet(predictions_path)
    
    before_2017 = df[df['case_year'] < 2017].copy()
    from_2017 = df[df['case_year'] >= 2017].copy()
    
    # Create detailed comparison
    create_difference_analysis(before_2017, from_2017)
    
    print(f"\n" + "🎉 TEMPORAL ANALYSIS COMPLETE!")
    print("="*70)
    print("Files created:")
    print("  📊 temporal_distribution_comparison.png")
    print("  📄 temporal_distribution_comparison.pdf")
    print("\nKey Insights:")
    print("  ✓ Visual comparison of outcome distributions")
    print("  ✓ Statistical analysis of temporal changes")
    print("  ✓ Confidence score comparisons")
    print("\nCheck if there are significant changes between the two periods!")
    
    return True

if __name__ == "__main__":
    main() 