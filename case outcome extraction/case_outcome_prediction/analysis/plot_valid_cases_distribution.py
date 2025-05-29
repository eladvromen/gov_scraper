#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
"""
Plot distribution of valid cases to verify randomness of missing data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_valid_cases_distribution():
    print('ANALYZING DISTRIBUTION OF VALID CASES')
    print('='*50)
    
    # Load the processed inference data
    df = pd.read_parquet('processed_inference_data.parquet')
    
    print(f"Total valid cases loaded: {len(df):,}")
    
    # Basic info
    print(f"Date range: {df['case_year'].min()} - {df['case_year'].max()}")
    print(f"Countries: {df['country'].nunique()}")
    print(f"Status types: {df['status'].nunique()}")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("tab10")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Distribution of Valid Cases (Non-Null Decision Text)', fontsize=16, fontweight='bold')
    
    # 1. Cases by Year - Simple bar plot
    ax1 = axes[0, 0]
    year_counts = df['case_year'].value_counts().sort_index()
    year_counts.plot(kind='bar', ax=ax1, color='skyblue', alpha=0.8)
    ax1.set_title('Cases per Year')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of Cases')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Cases by Year and Status - Stacked bar plot
    ax2 = axes[0, 1]
    if 'status' in df.columns and df['status'].notna().sum() > 0:
        pivot_status = df.pivot_table(index='case_year', columns='status', aggfunc='size', fill_value=0)
        pivot_status.plot(kind='bar', stacked=True, ax=ax2, colormap='tab20')
        ax2.set_title('Cases per Year by Status')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Number of Cases')
        ax2.legend(title='Status', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)
    else:
        ax2.text(0.5, 0.5, 'Status data not available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Cases per Year by Status (No Data)')
    
    # 3. Cases by Year and Country - Top countries only
    ax3 = axes[1, 0]
    if 'country' in df.columns and df['country'].notna().sum() > 0:
        # Get top 10 countries
        top_countries = df['country'].value_counts().head(10).index
        df_top_countries = df[df['country'].isin(top_countries)]
        
        pivot_country = df_top_countries.pivot_table(index='case_year', columns='country', aggfunc='size', fill_value=0)
        pivot_country.plot(kind='bar', stacked=True, ax=ax3, colormap='tab20')
        ax3.set_title('Cases per Year by Country (Top 10)')
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Number of Cases')
        ax3.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.tick_params(axis='x', rotation=45)
    else:
        ax3.text(0.5, 0.5, 'Country data not available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Cases per Year by Country (No Data)')
    
    # 4. Text length distribution by year
    ax4 = axes[1, 1]
    if 'original_token_length' in df.columns:
        # Create year bins for better visualization
        years = sorted(df['case_year'].unique())
        year_bins = [years[i:i+3] for i in range(0, len(years), 3)]  # Group by 3-year periods
        
        box_data = []
        box_labels = []
        for bin_years in year_bins:
            bin_data = df[df['case_year'].isin(bin_years)]['original_token_length']
            if len(bin_data) > 0:
                box_data.append(bin_data)
                if len(bin_years) == 1:
                    box_labels.append(str(bin_years[0]))
                else:
                    box_labels.append(f"{min(bin_years)}-{max(bin_years)}")
        
        if box_data:
            ax4.boxplot(box_data, labels=box_labels)
            ax4.set_title('Text Length Distribution by Year Period')
            ax4.set_xlabel('Year Period')
            ax4.set_ylabel('Token Length')
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, 'Token length data not available', ha='center', va='center', transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, 'Token length data not available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Text Length Distribution (No Data)')
    
    plt.tight_layout()
    plt.savefig('valid_cases_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: valid_cases_distribution.png")
    
    # Create summary statistics
    print(f"\nSUMMARY STATISTICS:")
    print(f"-"*30)
    print(f"Cases by year:")
    year_stats = df['case_year'].value_counts().sort_index()
    for year, count in year_stats.items():
        pct = count / len(df) * 100
        print(f"  {year}: {count:,} cases ({pct:.1f}%)")
    
    if 'status' in df.columns:
        print(f"\nCases by status:")
        status_stats = df['status'].value_counts()
        for status, count in status_stats.head(10).items():
            pct = count / len(df) * 100
            print(f"  {status}: {count:,} cases ({pct:.1f}%)")
    
    if 'country' in df.columns:
        print(f"\nTop 10 countries:")
        country_stats = df['country'].value_counts()
        for country, count in country_stats.head(10).items():
            pct = count / len(df) * 100
            print(f"  {country}: {count:,} cases ({pct:.1f}%)")
    
    # Check for potential bias indicators
    print(f"\nBIAS ANALYSIS:")
    print(f"-"*30)
    
    # Year distribution uniformity
    year_counts = df['case_year'].value_counts()
    year_cv = year_counts.std() / year_counts.mean()
    print(f"Year distribution coefficient of variation: {year_cv:.3f}")
    if year_cv < 0.5:
        print("✓ Year distribution appears relatively uniform")
    else:
        print("⚠️  Year distribution shows some variation")
    
    # Text length consistency
    if 'original_token_length' in df.columns:
        text_length_by_year = df.groupby('case_year')['original_token_length'].mean()
        text_length_cv = text_length_by_year.std() / text_length_by_year.mean()
        print(f"Text length consistency across years (CV): {text_length_cv:.3f}")
        if text_length_cv < 0.3:
            print("✓ Text lengths consistent across years")
        else:
            print("⚠️  Text lengths vary significantly by year")
    
    plt.show()
    return df

if __name__ == "__main__":
    df = plot_valid_cases_distribution() 