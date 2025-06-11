#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
"""
High Confidence Predictions - Label Distribution Analysis
========================================================

This script creates a pie chart showing the distribution of predicted outcomes
for high-confidence cases only (confidence >= 0.8).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def create_high_confidence_pie_chart():
    """Create pie chart for high confidence predictions"""
    
    print("="*60)
    print("HIGH CONFIDENCE PREDICTIONS - LABEL DISTRIBUTION")
    print("="*60)
    
    # Load high confidence predictions
    predictions_path = "../results/predictions/high_confidence_predictions.parquet"
    
    if not Path(predictions_path).exists():
        print(f"❌ High confidence predictions file not found: {predictions_path}")
        print("Please run the inference pipeline first (step 6)")
        return False
    
    df = pd.read_parquet(predictions_path)
    print(f"✓ Loaded {len(df):,} high confidence predictions")
    
    # Get prediction distribution
    pred_counts = df['predicted_outcome'].value_counts()
    total_cases = len(df)
    
    print(f"\nHigh confidence prediction distribution:")
    for outcome, count in pred_counts.items():
        pct = count / total_cases * 100
        print(f"  {outcome}: {count:,} cases ({pct:.1f}%)")
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define colors for each outcome
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        pred_counts.values,
        labels=pred_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors[:len(pred_counts)],
        explode=[0.05] * len(pred_counts),  # Slight separation between wedges
        shadow=True,
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    
    # Enhance the appearance
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)
        autotext.set_fontweight('bold')
    
    for text in texts:
        text.set_fontsize(14)
        text.set_fontweight('bold')
    
    # Add title
    plt.title(
        'Distribution of Predicted Outcomes\n(High Confidence Cases Only ≥80%)',
        fontsize=18,
        fontweight='bold',
        pad=20
    )
    
    # Add statistics box
    conf_stats = df['prediction_confidence'].describe()
    stats_text = f"""High Confidence Statistics:
    Total Cases: {total_cases:,}
    Min Confidence: {conf_stats['min']:.3f}
    Mean Confidence: {conf_stats['mean']:.3f}
    Max Confidence: {conf_stats['max']:.3f}"""
    
    # Position the statistics box
    ax.text(
        1.3, 0.5, stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='center',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8)
    )
    
    # Add confidence threshold annotation
    threshold_text = "Confidence Threshold: ≥80%"
    ax.text(
        1.3, 0.2, threshold_text,
        transform=ax.transAxes,
        fontsize=12,
        fontweight='bold',
        verticalalignment='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8)
    )
    
    # Ensure pie chart is circular
    ax.set_aspect('equal')
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    # Save the plot
    output_path = "../outputs/plots/high_confidence_predictions_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved pie chart: {output_path}")
    
    # Also save as PDF for high quality
    pdf_path = "../outputs/plots/high_confidence_predictions_distribution.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved PDF version: {pdf_path}")
    
    plt.show()
    
    # Print summary
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"📊 High confidence cases analyzed: {total_cases:,}")
    print(f"📈 Confidence range: {conf_stats['min']:.3f} - {conf_stats['max']:.3f}")
    print(f"📋 Outcome categories: {len(pred_counts)}")
    print(f"🎯 Most common outcome: {pred_counts.index[0]} ({pred_counts.iloc[0]:,} cases)")
    
    return True

def create_confidence_vs_outcome_plot():
    """Create additional analysis: confidence distribution by outcome"""
    
    print(f"\n" + "="*60)
    print("BONUS: CONFIDENCE DISTRIBUTION BY OUTCOME")
    print("="*60)
    
    # Load high confidence predictions
    predictions_path = "../results/predictions/high_confidence_predictions.parquet"
    df = pd.read_parquet(predictions_path)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Box plot of confidence by outcome
    sns.boxplot(data=df, x='predicted_outcome', y='prediction_confidence', ax=ax)
    
    # Enhance the plot
    ax.set_title('Prediction Confidence Distribution by Outcome\n(High Confidence Cases Only)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Outcome', fontsize=14, fontweight='bold')
    ax.set_ylabel('Prediction Confidence', fontsize=14, fontweight='bold')
    
    # Add horizontal line at 0.8 threshold
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, 
              label='Confidence Threshold (80%)')
    ax.legend()
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "../outputs/plots/confidence_by_outcome_boxplot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved confidence boxplot: {output_path}")
    
    plt.show()
    
    return True

def main():
    """Main function to run all analyses"""
    print("🎯 ANALYZING HIGH CONFIDENCE PREDICTIONS")
    print("="*60)
    
    # Create pie chart
    success1 = create_high_confidence_pie_chart()
    
    if success1:
        # Create bonus confidence analysis
        success2 = create_confidence_vs_outcome_plot()
        
        if success1 and success2:
            print(f"\n" + "🎉 ANALYSIS COMPLETE!")
            print("="*60)
            print("Files created:")
            print("  📊 high_confidence_predictions_distribution.png")
            print("  📄 high_confidence_predictions_distribution.pdf") 
            print("  📈 confidence_by_outcome_boxplot.png")
            print("\nCheck the outputs/plots/ directory for your visualizations!")
    
    return success1

if __name__ == "__main__":
    main() 