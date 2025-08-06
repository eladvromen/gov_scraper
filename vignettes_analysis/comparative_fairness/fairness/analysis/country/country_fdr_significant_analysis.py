#!/usr/bin/env python3
"""
Country Disparity Analysis - FDR-Corrected Significance
=======================================================

Uses deduplicated data (707 comparisons) with FDR-corrected significance
to analyze country bias patterns between Pre-Brexit and Post-Brexit models.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_deduplicated_data():
    """Load deduplicated fairness data with FDR-corrected significance"""
    print("Loading deduplicated fairness data with FDR-corrected significance...")
    
    data_path = "../../outputs/vector_drift/deduplicated_fairness_comparisons.csv"
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} deduplicated comparisons")
    print(f"FDR significance rates: Pre={df['pre_brexit_sp_significance'].sum()} ({df['pre_brexit_sp_significance'].mean():.1%}), "
          f"Post={df['post_brexit_sp_significance'].sum()} ({df['post_brexit_sp_significance'].mean():.1%})")
    
    return df

def parse_comparison_labels(df):
    """Parse comparison labels to extract country, topic, and other metadata"""
    print("Parsing comparison labels...")
    
    parsed_data = []
    
    for _, row in df.iterrows():
        label = row['comparison_label']
        
        # Parse label format: "Group1_vs_Group2 (attribute) [topic]"
        try:
            # Split by parentheses to get attribute
            if '(' in label and ')' in label:
                main_part = label.split(' (')[0]  # "Group1_vs_Group2"
                remainder = label.split(' (')[1]   # "attribute) [topic]"
                
                attribute = remainder.split(')')[0]  # "attribute"
                topic = remainder.split('[')[1].split(']')[0] if '[' in remainder else "Unknown"
                
                # Extract groups
                if '_vs_' in main_part:
                    group1, group2 = main_part.split('_vs_')
                else:
                    group1, group2 = main_part, "Unknown"
                
                parsed_data.append({
                    'comparison_label': label,
                    'group1': group1,
                    'group2': group2, 
                    'protected_attribute': attribute,
                    'topic': topic,
                    'pre_brexit_sp': row['pre_brexit_sp_magnitude'],
                    'post_brexit_sp': row['post_brexit_sp_magnitude'],
                    'sp_change': row['sp_magnitude_difference'],
                    'pre_brexit_significant': row['pre_brexit_sp_significance'],
                    'post_brexit_significant': row['post_brexit_sp_significance'],
                    'gained_significance': row['sp_gained_significance'],
                    'lost_significance': row['sp_lost_significance'],
                    'both_significant': row['sp_both_significant'],
                    'neither_significant': row['sp_neither_significant']
                })
            else:
                # Fallback for unparseable labels
                parsed_data.append({
                    'comparison_label': label,
                    'group1': "Unknown", 
                    'group2': "Unknown",
                    'protected_attribute': "Unknown",
                    'topic': "Unknown",
                    'pre_brexit_sp': row['pre_brexit_sp_magnitude'],
                    'post_brexit_sp': row['post_brexit_sp_magnitude'], 
                    'sp_change': row['sp_magnitude_difference'],
                    'pre_brexit_significant': row['pre_brexit_sp_significance'],
                    'post_brexit_significant': row['post_brexit_sp_significance'],
                    'gained_significance': row['sp_gained_significance'],
                    'lost_significance': row['sp_lost_significance'],
                    'both_significant': row['sp_both_significant'],
                    'neither_significant': row['sp_neither_significant']
                })
                
        except Exception as e:
            print(f"Error parsing label '{label}': {e}")
            continue
    
    parsed_df = pd.DataFrame(parsed_data)
    print(f"Successfully parsed {len(parsed_df)} comparisons")
    
    return parsed_df

def analyze_country_comparisons(df):
    """Focus analysis on country comparisons only"""
    print(f"\n{'='*80}")
    print("COUNTRY DISPARITY ANALYSIS - FDR CORRECTED")
    print(f"{'='*80}")
    
    # Filter for country comparisons
    country_df = df[df['protected_attribute'] == 'country'].copy()
    print(f"Country comparisons: {len(country_df)} out of {len(df)} total ({len(country_df)/len(df):.1%})")
    
    if len(country_df) == 0:
        print("❌ No country comparisons found!")
        return None
    
    # Add FDR significance indicator
    country_df['fdr_significant'] = (country_df['pre_brexit_significant'] | 
                                   country_df['post_brexit_significant'])
    
    # Basic statistics
    print(f"\n📊 COUNTRY BIAS STATISTICS:")
    print(f"   Pre-Brexit significant:  {country_df['pre_brexit_significant'].sum():3d} ({country_df['pre_brexit_significant'].mean():.1%})")
    print(f"   Post-Brexit significant: {country_df['post_brexit_significant'].sum():3d} ({country_df['post_brexit_significant'].mean():.1%})")
    print(f"   Either model significant: {country_df['fdr_significant'].sum():3d} ({country_df['fdr_significant'].mean():.1%})")
    print(f"   Both models significant:  {country_df['both_significant'].sum():3d} ({country_df['both_significant'].mean():.1%})")
    print(f"   Gained significance:     {country_df['gained_significance'].sum():3d} ({country_df['gained_significance'].mean():.1%})")
    print(f"   Lost significance:       {country_df['lost_significance'].sum():3d} ({country_df['lost_significance'].mean():.1%})")
    
    return country_df

def print_significant_country_comparisons(country_df):
    """Print detailed significant country comparisons"""
    
    # Filter for FDR-significant comparisons
    significant_df = country_df[country_df['fdr_significant'] == True].copy()
    
    print(f"\n{'='*120}")
    print(f"FDR-SIGNIFICANT COUNTRY COMPARISONS")
    print(f"{'='*120}")
    print(f"Total FDR-significant country comparisons: {len(significant_df)}")
    
    if len(significant_df) == 0:
        print("❌ No country comparisons remain significant after FDR correction!")
        return
    
    # Group by topic
    print(f"\n📋 BREAKDOWN BY TOPIC:")
    topic_counts = significant_df.groupby('topic').size().sort_values(ascending=False)
    for topic, count in topic_counts.items():
        pct = count / len(significant_df) * 100
        topic_short = topic[:60] + "..." if len(topic) > 60 else topic
        print(f"   {topic_short:63s}: {count:2d} ({pct:4.1f}%)")
    
    # Show detailed comparisons sorted by absolute bias change
    print(f"\n📝 DETAILED SIGNIFICANT COUNTRY COMPARISONS:")
    print(f"{'='*140}")
    print(f"{'Country Comparison':<25} | {'Topic':<40} | {'Pre-SP':>8} | {'Post-SP':>8} | {'SP-Δ':>8} | {'Significance Pattern':<20}")
    print(f"{'-'*140}")
    
    # Sort by absolute bias change for most interesting results
    significant_df['abs_sp_change'] = abs(significant_df['sp_change'])
    significant_df_sorted = significant_df.sort_values('abs_sp_change', ascending=False)
    
    for _, row in significant_df_sorted.iterrows():
        comparison = f"{row['group1']} vs {row['group2']}"[:24]
        topic = row['topic'][:39]
        pre_sp = row['pre_brexit_sp']
        post_sp = row['post_brexit_sp']
        sp_change = row['sp_change']
        
        # Determine significance pattern
        if row['both_significant']:
            sig_pattern = "Both Significant"
        elif row['gained_significance']:
            sig_pattern = "Newly Emerged"
        elif row['lost_significance']:
            sig_pattern = "Disappeared"
        elif row['pre_brexit_significant']:
            sig_pattern = "Pre-Brexit Only"
        elif row['post_brexit_significant']:
            sig_pattern = "Post-Brexit Only"
        else:
            sig_pattern = "Unknown"
        
        print(f"{comparison:<25} | {topic:<40} | {pre_sp:>8.3f} | {post_sp:>8.3f} | {sp_change:>+8.3f} | {sig_pattern:<20}")

def analyze_country_pairs(country_df):
    """Analyze specific country pair relationships"""
    print(f"\n{'='*80}")
    print("COUNTRY PAIR ANALYSIS")
    print(f"{'='*80}")
    
    # Extract unique countries
    countries = set()
    for _, row in country_df.iterrows():
        countries.add(row['group1'])
        countries.add(row['group2'])
    
    countries = sorted(list(countries))
    print(f"Countries analyzed: {', '.join(countries)}")
    
    # Find most biased country pairs
    significant_df = country_df[country_df['fdr_significant'] == True]
    if len(significant_df) > 0:
        print(f"\n🔥 MOST EXTREME BIAS CHANGES:")
        extreme_changes = significant_df.nlargest(5, 'abs_sp_change')
        for i, (_, row) in enumerate(extreme_changes.iterrows(), 1):
            print(f"   {i}. {row['group1']} vs {row['group2']}: {row['sp_change']:+.3f} SP change")
            print(f"      Topic: {row['topic']}")
            print(f"      Pre: {row['pre_brexit_sp']:+.3f} → Post: {row['post_brexit_sp']:+.3f}")
            
def analyze_topics_by_country_bias(country_df):
    """Analyze which topics show most country bias"""
    print(f"\n{'='*80}")
    print("TOPIC-LEVEL COUNTRY BIAS ANALYSIS") 
    print(f"{'='*80}")
    
    # Group by topic and calculate bias statistics
    topic_stats = []
    
    for topic in country_df['topic'].unique():
        topic_data = country_df[country_df['topic'] == topic]
        
        stats = {
            'topic': topic,
            'total_comparisons': len(topic_data),
            'significant_comparisons': topic_data['fdr_significant'].sum(),
            'significance_rate': topic_data['fdr_significant'].mean(),
            'mean_abs_bias_change': topic_data['abs_sp_change'].mean(),
            'max_bias_change': topic_data['abs_sp_change'].max(),
            'persistent_bias': topic_data['both_significant'].sum(),
            'newly_emerged_bias': topic_data['gained_significance'].sum(),
            'disappeared_bias': topic_data['lost_significance'].sum()
        }
        topic_stats.append(stats)
    
    topic_stats_df = pd.DataFrame(topic_stats)
    topic_stats_df = topic_stats_df.sort_values('significance_rate', ascending=False)
    
    print(f"📊 TOPICS RANKED BY COUNTRY BIAS SIGNIFICANCE RATE:")
    print(f"{'Topic':<45} | {'Total':>5} | {'Sig':>3} | {'Rate':>6} | {'Avg|Δ|':>7} | {'Max|Δ|':>7} | {'P/N/D':>5}")
    print(f"{'-'*90}")
    
    for _, row in topic_stats_df.iterrows():
        topic_short = row['topic'][:44]
        total = int(row['total_comparisons'])
        sig = int(row['significant_comparisons'])
        rate = row['significance_rate']
        avg_change = row['mean_abs_bias_change']
        max_change = row['max_bias_change']
        persistent = int(row['persistent_bias'])
        newly = int(row['newly_emerged_bias'])
        disappeared = int(row['disappeared_bias'])
        
        print(f"{topic_short:<45} | {total:>5} | {sig:>3} | {rate:>5.1%} | {avg_change:>7.3f} | {max_change:>7.3f} | {persistent}/{newly}/{disappeared}")

def save_country_analysis_results(country_df, output_dir="../../outputs/country_analysis"):
    """Save country analysis results"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save full country dataframe
    country_df.to_csv(output_path / "country_comparisons_fdr_corrected.csv", index=False)
    
    # Save significant comparisons only
    significant_df = country_df[country_df['fdr_significant'] == True]
    significant_df.to_csv(output_path / "significant_country_comparisons_fdr.csv", index=False)
    
    # Create summary statistics
    summary = {
        'total_country_comparisons': len(country_df),
        'fdr_significant_comparisons': len(significant_df),
        'significance_rate': len(significant_df) / len(country_df),
        'pre_brexit_significant': country_df['pre_brexit_significant'].sum(),
        'post_brexit_significant': country_df['post_brexit_significant'].sum(),
        'both_significant': country_df['both_significant'].sum(),
        'gained_significance': country_df['gained_significance'].sum(),
        'lost_significance': country_df['lost_significance'].sum(),
        'topics_analyzed': country_df['topic'].nunique(),
        'countries_analyzed': len(set(country_df['group1'].tolist() + country_df['group2'].tolist())),
        'largest_bias_change': country_df['abs_sp_change'].max(),
        'mean_bias_change': country_df['abs_sp_change'].mean()
    }
    
    with open(output_path / "country_analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Country analysis results saved to: {output_path}")
    return summary

def main():
    """Main analysis function"""
    
    print("🌍 COUNTRY DISPARITY ANALYSIS - FDR CORRECTED")
    print("="*80)
    
    # Load deduplicated data with FDR-corrected significance
    df = load_deduplicated_data()
    
    # Parse comparison labels 
    parsed_df = parse_comparison_labels(df)
    
    # Focus on country comparisons
    country_df = analyze_country_comparisons(parsed_df)
    
    if country_df is not None:
        # Add absolute bias change for analysis
        country_df['abs_sp_change'] = abs(country_df['sp_change'])
        
        # Print significant comparisons
        print_significant_country_comparisons(country_df)
        
        # Analyze country pairs
        analyze_country_pairs(country_df)
        
        # Analyze topics
        analyze_topics_by_country_bias(country_df)
        
        # Save results
        summary = save_country_analysis_results(country_df)
        
        print(f"\n🎯 COUNTRY ANALYSIS SUMMARY:")
        print(f"   Total country comparisons: {summary['total_country_comparisons']}")
        print(f"   FDR-significant: {summary['fdr_significant_comparisons']} ({summary['significance_rate']:.1%})")
        print(f"   Countries analyzed: {summary['countries_analyzed']}")
        print(f"   Topics analyzed: {summary['topics_analyzed']}")
        print(f"   Largest bias change: {summary['largest_bias_change']:.3f}")
        
        return country_df
    
    return None

if __name__ == "__main__":
    result_df = main() 