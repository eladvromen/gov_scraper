#!/usr/bin/env python3
"""
Systematic Country Bias Trends - Publication Plot
================================================

Creates a publication-quality bar plot showing mean statistical parity changes
by country with significance-based color coding.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path

def load_and_analyze_country_trends():
    """Load data and perform systematic country bias analysis"""
    print("Loading and analyzing systematic country bias trends...")
    
    # Load deduplicated data
    data_path = "../../outputs/vector_drift/deduplicated_fairness_comparisons.csv"
    df = pd.read_csv(data_path)
    
    # Extract FDR-significant country comparisons
    country_data = []
    
    for _, row in df.iterrows():
        label = row['comparison_label']
        
        if '(country)' in label:
            if row['pre_brexit_sp_significance'] or row['post_brexit_sp_significance']:
                
                # Parse label components
                comparison_part = label.split(' (')[0]
                remaining = label.split(' (')[1] if ' (' in label else ""
                
                topic = ""
                if '[' in remaining and ']' in remaining:
                    topic = remaining.split('[')[1].split(']')[0]
                
                parts = comparison_part.split('_vs_')
                if len(parts) == 2:
                    country1, country2 = parts
                    sp_change = row['sp_magnitude_difference']
                    
                    country_data.append({
                        'country1': country1,
                        'country2': country2,
                        'topic': topic,
                        'sp_change': sp_change,
                    })
    
    # Aggregate by country
    country_trends = defaultdict(list)
    
    for data in country_data:
        c1, c2 = data['country1'], data['country2']
        sp_change = data['sp_change']
        
        # For country1: positive sp_change means country1 is more favored
        country_trends[c1].append(sp_change)
        # For country2: negative sp_change means country2 is more favored  
        country_trends[c2].append(-sp_change)
    
    # Calculate statistics for each country
    country_stats = []
    for country, changes in country_trends.items():
        if len(changes) >= 3:  # Only countries with multiple comparisons
            mean_change = np.mean(changes)
            
            # One-sample t-test against zero
            t_stat, p_value = stats.ttest_1samp(changes, 0)
            
            # Determine significance category
            if p_value < 0.05:
                if mean_change < 0:
                    significance_category = 'Significant Disfavor'
                else:
                    significance_category = 'Significant Favor'
            else:
                significance_category = 'Non-significant'
            
            country_stats.append({
                'country': country,
                'mean_bias_change': mean_change,
                'p_value': p_value,
                'n_comparisons': len(changes),
                'significance_category': significance_category
            })
    
    return pd.DataFrame(country_stats)

def create_systematic_bias_plot(country_df):
    """Create publication-quality systematic bias plot"""
    print("Creating systematic country bias plot...")
    
    # Sort countries by mean bias change
    country_df_sorted = country_df.sort_values('mean_bias_change')
    
    # Set up the plot with publication style
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors based on significance
    color_map = {
        'Significant Disfavor': '#d62728',  # Red
        'Significant Favor': '#2ca02c',     # Green  
        'Non-significant': '#404040'        # Dark gray
    }
    
    colors = [color_map[cat] for cat in country_df_sorted['significance_category']]
    
    # Create the bar plot
    bars = ax.bar(range(len(country_df_sorted)), 
                  country_df_sorted['mean_bias_change'],
                  color=colors, 
                  alpha=0.8,
                  edgecolor='black',
                  linewidth=0.8)
    
    # Customize the plot
    ax.set_xlabel('Country', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Change in Statistical Parity (Δ SP)', fontsize=14, fontweight='bold')
    ax.set_title('Systematic Country Bias Trends: Post-Brexit vs Pre-Brexit Models\n' + 
                'Mean Statistical Parity Changes with Significance Testing',
                fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis labels
    ax.set_xticks(range(len(country_df_sorted)))
    ax.set_xticklabels(country_df_sorted['country'], rotation=45, ha='right')
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Add significance indicators (asterisks)
    for i, (_, row) in enumerate(country_df_sorted.iterrows()):
        if row['p_value'] < 0.01:
            significance_marker = '**'
        elif row['p_value'] < 0.05:
            significance_marker = '*'
        else:
            significance_marker = ''
        
        if significance_marker:
            y_pos = row['mean_bias_change']
            offset = 0.01 if y_pos >= 0 else -0.01
            ax.text(i, y_pos + offset, significance_marker, 
                   ha='center', va='bottom' if y_pos >= 0 else 'top',
                   fontsize=12, fontweight='bold')
    
    # Create custom legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=color_map['Significant Disfavor'], alpha=0.8, 
                     edgecolor='black', label='Significantly Disfavored (p < 0.05)'),
        plt.Rectangle((0,0),1,1, facecolor=color_map['Significant Favor'], alpha=0.8,
                     edgecolor='black', label='Significantly Favored (p < 0.05)'),
        plt.Rectangle((0,0),1,1, facecolor=color_map['Non-significant'], alpha=0.8,
                     edgecolor='black', label='Non-significant (p ≥ 0.05)')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.9)
    
    # Add grid for better readability
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add annotations for interpretation
    ax.text(0.02, 0.98, 'Positive values: More favored post-Brexit\nNegative values: Less favored post-Brexit',
           transform=ax.transAxes, fontsize=10, va='top', ha='left',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
    
    # Add sample size annotations
    for i, (_, row) in enumerate(country_df_sorted.iterrows()):
        ax.text(i, ax.get_ylim()[0] - 0.01, f'n={row["n_comparisons"]}',
               ha='center', va='top', fontsize=9, alpha=0.7)
    
    # Fine-tune layout
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("../../outputs/country_analysis")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "systematic_country_bias_trends.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / "systematic_country_bias_trends.pdf", 
                bbox_inches='tight', facecolor='white')
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("📊 SYSTEMATIC COUNTRY BIAS SUMMARY")
    print("="*70)
    
    significant_disfavor = country_df_sorted[country_df_sorted['significance_category'] == 'Significant Disfavor']
    significant_favor = country_df_sorted[country_df_sorted['significance_category'] == 'Significant Favor']
    non_significant = country_df_sorted[country_df_sorted['significance_category'] == 'Non-significant']
    
    print(f"\n🔴 SIGNIFICANTLY DISFAVORED COUNTRIES ({len(significant_disfavor)}):")
    for _, row in significant_disfavor.iterrows():
        print(f"   {row['country']:12} | Δ SP = {row['mean_bias_change']:+.3f} | p = {row['p_value']:.3f} | n = {row['n_comparisons']}")
    
    print(f"\n🟢 SIGNIFICANTLY FAVORED COUNTRIES ({len(significant_favor)}):")
    for _, row in significant_favor.iterrows():
        print(f"   {row['country']:12} | Δ SP = {row['mean_bias_change']:+.3f} | p = {row['p_value']:.3f} | n = {row['n_comparisons']}")
    
    print(f"\n⚫ NON-SIGNIFICANT COUNTRIES ({len(non_significant)}):")
    for _, row in non_significant.iterrows():
        print(f"   {row['country']:12} | Δ SP = {row['mean_bias_change']:+.3f} | p = {row['p_value']:.3f} | n = {row['n_comparisons']}")
    
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"   Total countries analyzed: {len(country_df_sorted)}")
    print(f"   Countries with significant trends: {len(significant_disfavor) + len(significant_favor)} ({(len(significant_disfavor) + len(significant_favor))/len(country_df_sorted)*100:.1f}%)")
    print(f"   Largest positive change: {country_df_sorted['mean_bias_change'].max():+.3f}")
    print(f"   Largest negative change: {country_df_sorted['mean_bias_change'].min():+.3f}")
    
    return country_df_sorted

def main():
    """Main function"""
    print("🌍 SYSTEMATIC COUNTRY BIAS TRENDS VISUALIZATION")
    print("="*70)
    
    # Load and analyze data
    country_df = load_and_analyze_country_trends()
    
    # Create visualization
    country_df_sorted = create_systematic_bias_plot(country_df)
    
    print(f"\n✅ Plot saved to: ../../outputs/country_analysis/")
    print("   - systematic_country_bias_trends.png (high resolution)")
    print("   - systematic_country_bias_trends.pdf (publication ready)")

if __name__ == "__main__":
    main() 