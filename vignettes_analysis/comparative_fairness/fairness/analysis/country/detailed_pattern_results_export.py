#!/usr/bin/env python3
"""
Detailed Pattern Results Export
==============================

Generates comprehensive data-driven results tables for bias pattern 
emergence vs disappearance analysis for paper inclusion.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

def load_and_process_pattern_data():
    """Load and process pattern transition data with detailed categorization"""
    print("Loading detailed pattern transition data...")
    
    # Load deduplicated data
    data_path = "../../outputs/vector_drift/deduplicated_fairness_comparisons.csv"
    df = pd.read_csv(data_path)
    
    # Extract all country comparisons with pattern classification
    pattern_data = []
    
    for _, row in df.iterrows():
        label = row['comparison_label']
        
        if '(country)' in label:
            # Parse label components
            comparison_part = label.split(' (')[0]
            remaining = label.split(' (')[1] if ' (' in label else ""
            
            topic = ""
            if '[' in remaining and ']' in remaining:
                topic = remaining.split('[')[1].split(']')[0]
            
            parts = comparison_part.split('_vs_')
            if len(parts) == 2:
                country1, country2 = parts
                
                # Determine pattern type
                pre_sig = row['pre_brexit_sp_significance']
                post_sig = row['post_brexit_sp_significance']
                
                if pre_sig and post_sig:
                    pattern_type = 'PERSISTENT'
                elif not pre_sig and post_sig:
                    pattern_type = 'NEWLY_EMERGED'
                elif pre_sig and not post_sig:
                    pattern_type = 'DISAPPEARED'
                else:
                    continue  # Skip non-significant comparisons
                
                sp_change = row['sp_magnitude_difference']
                
                # Determine which country is favored
                if sp_change > 0:
                    favored_country = country1
                    disfavored_country = country2
                else:
                    favored_country = country2
                    disfavored_country = country1
                
                pattern_data.append({
                    'country1': country1,
                    'country2': country2,
                    'country_pair': f"{country1} vs {country2}",
                    'topic': topic,
                    'pattern_type': pattern_type,
                    'pre_sp': row['pre_brexit_sp_magnitude'],
                    'post_sp': row['post_brexit_sp_magnitude'],
                    'sp_change': sp_change,
                    'abs_sp_change': abs(sp_change),
                    'favored_country': favored_country,
                    'disfavored_country': disfavored_country,
                    'pre_significant': pre_sig,
                    'post_significant': post_sig
                })
    
    return pd.DataFrame(pattern_data)

def create_detailed_pattern_tables(df):
    """Create detailed tables for paper inclusion"""
    
    print("\nGenerating detailed pattern analysis tables...")
    
    # Table 1: Topic-Level Pattern Transitions
    print("\n" + "="*80)
    print("📊 TABLE 1: TOPIC-LEVEL BIAS PATTERN TRANSITIONS")
    print("="*80)
    
    topic_analysis = df.groupby(['topic', 'pattern_type']).size().unstack(fill_value=0)
    topic_analysis['total'] = topic_analysis.sum(axis=1)
    topic_analysis['net_change'] = topic_analysis.get('NEWLY_EMERGED', 0) - topic_analysis.get('DISAPPEARED', 0)
    topic_analysis = topic_analysis.sort_values('net_change', ascending=False)
    
    print("Topic-Level Pattern Transitions Summary:")
    print("(Positive net_change = more biased post-Brexit, Negative = less biased)")
    print()
    print(f"{'Topic':<40} {'Emerged':<8} {'Disappeared':<12} {'Persistent':<10} {'Total':<6} {'Net Δ':<6}")
    print("-" * 86)
    
    for topic, row in topic_analysis.iterrows():
        emerged = row.get('NEWLY_EMERGED', 0)
        disappeared = row.get('DISAPPEARED', 0)
        persistent = row.get('PERSISTENT', 0)
        total = row['total']
        net_change = row['net_change']
        
        # Truncate long topic names
        topic_short = topic[:37] + "..." if len(topic) > 40 else topic
        
        print(f"{topic_short:<40} {emerged:<8} {disappeared:<12} {persistent:<10} {total:<6} {net_change:+3}")
    
    # Table 2: Country Favorability Changes
    print(f"\n" + "="*80)
    print("📊 TABLE 2: COUNTRY FAVORABILITY PATTERN CHANGES")
    print("="*80)
    
    # Calculate favorability changes per country
    newly_emerged = df[df['pattern_type'] == 'NEWLY_EMERGED']
    disappeared = df[df['pattern_type'] == 'DISAPPEARED']
    
    # Count favorability patterns
    emerged_favorability = newly_emerged['favored_country'].value_counts()
    disappeared_favorability = disappeared['favored_country'].value_counts()
    
    # Combine all countries
    all_countries = set(emerged_favorability.index) | set(disappeared_favorability.index)
    favorability_changes = []
    
    for country in all_countries:
        emerged_count = emerged_favorability.get(country, 0)
        disappeared_count = disappeared_favorability.get(country, 0)
        net_favorability = emerged_count - disappeared_count
        
        favorability_changes.append({
            'country': country,
            'newly_favored': emerged_count,
            'lost_favorability': disappeared_count,
            'net_favorability_change': net_favorability
        })
    
    favorability_df = pd.DataFrame(favorability_changes).sort_values('net_favorability_change', ascending=False)
    
    print("Country Favorability Changes Summary:")
    print("(Positive = gained favorable bias patterns, Negative = lost favorable patterns)")
    print()
    print(f"{'Country':<12} {'Gained':<6} {'Lost':<6} {'Net Change':<11} {'Trend':<20}")
    print("-" * 55)
    
    for _, row in favorability_df.iterrows():
        country = row['country']
        gained = row['newly_favored']
        lost = row['lost_favorability']
        net_change = row['net_favorability_change']
        
        if net_change > 2:
            trend = "MAJOR GAIN"
        elif net_change > 0:
            trend = "Gain"
        elif net_change < -2:
            trend = "MAJOR LOSS"
        elif net_change < 0:
            trend = "Loss"
        else:
            trend = "No Change"
        
        print(f"{country:<12} {gained:<6} {lost:<6} {net_change:+3}         {trend:<20}")
    
    # Table 3: Specific Pattern Details
    print(f"\n" + "="*80)
    print("📊 TABLE 3: DETAILED PATTERN TRANSITION EXAMPLES")
    print("="*80)
    
    print("🆕 TOP NEWLY EMERGED BIAS PATTERNS:")
    print(f"{'Country Pair':<25} {'Topic':<35} {'Favored':<12} {'Magnitude':<9}")
    print("-" * 85)
    
    top_emerged = newly_emerged.nlargest(10, 'abs_sp_change')
    for _, row in top_emerged.iterrows():
        pair = row['country_pair']
        topic = row['topic'][:32] + "..." if len(row['topic']) > 35 else row['topic']
        favored = row['favored_country']
        magnitude = f"{row['abs_sp_change']:.3f}"
        
        print(f"{pair:<25} {topic:<35} {favored:<12} {magnitude:<9}")
    
    print(f"\n📉 TOP DISAPPEARED BIAS PATTERNS:")
    print(f"{'Country Pair':<25} {'Topic':<35} {'Favored':<12} {'Magnitude':<9}")
    print("-" * 85)
    
    top_disappeared = disappeared.nlargest(10, 'abs_sp_change')
    for _, row in top_disappeared.iterrows():
        pair = row['country_pair']
        topic = row['topic'][:32] + "..." if len(row['topic']) > 35 else row['topic']
        favored = row['favored_country']
        magnitude = f"{row['abs_sp_change']:.3f}"
        
        print(f"{pair:<25} {topic:<35} {favored:<12} {magnitude:<9}")
    
    # Statistical Summary
    print(f"\n" + "="*80)
    print("📊 STATISTICAL SUMMARY")
    print("="*80)
    
    total_comparisons = len(df)
    emerged_count = len(newly_emerged)
    disappeared_count = len(disappeared)
    persistent_count = len(df[df['pattern_type'] == 'PERSISTENT'])
    
    print(f"Total FDR-significant country comparisons: {total_comparisons}")
    print(f"  • Newly emerged patterns: {emerged_count} ({emerged_count/total_comparisons*100:.1f}%)")
    print(f"  • Disappeared patterns: {disappeared_count} ({disappeared_count/total_comparisons*100:.1f}%)")
    print(f"  • Persistent patterns: {persistent_count} ({persistent_count/total_comparisons*100:.1f}%)")
    print(f"  • Net pattern change: {emerged_count - disappeared_count:+d}")
    
    print(f"\nMagnitude Analysis:")
    print(f"  • Newly emerged bias magnitude: μ = {newly_emerged['abs_sp_change'].mean():.3f} ± {newly_emerged['abs_sp_change'].std():.3f}")
    print(f"  • Disappeared bias magnitude: μ = {disappeared['abs_sp_change'].mean():.3f} ± {disappeared['abs_sp_change'].std():.3f}")
    
    print(f"\nTopic Analysis:")
    topics_more_biased = len(topic_analysis[topic_analysis['net_change'] > 0])
    topics_less_biased = len(topic_analysis[topic_analysis['net_change'] < 0])
    print(f"  • Topics with increased bias patterns: {topics_more_biased}")
    print(f"  • Topics with decreased bias patterns: {topics_less_biased}")
    
    print(f"\nCountry Analysis:")
    countries_gained = len(favorability_df[favorability_df['net_favorability_change'] > 0])
    countries_lost = len(favorability_df[favorability_df['net_favorability_change'] < 0])
    print(f"  • Countries that gained favorability: {countries_gained}")
    print(f"  • Countries that lost favorability: {countries_lost}")
    
    return topic_analysis, favorability_df

def export_to_files(df, topic_analysis, favorability_df):
    """Export detailed results to CSV files"""
    
    output_dir = Path("../../outputs/country_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Export main pattern data
    df.to_csv(output_dir / "detailed_pattern_transitions.csv", index=False)
    
    # Export topic analysis
    topic_analysis.to_csv(output_dir / "topic_pattern_transitions.csv")
    
    # Export favorability analysis
    favorability_df.to_csv(output_dir / "country_favorability_changes.csv", index=False)
    
    print(f"\n✅ Detailed results exported to:")
    print(f"   • detailed_pattern_transitions.csv")
    print(f"   • topic_pattern_transitions.csv") 
    print(f"   • country_favorability_changes.csv")

def main():
    """Main function"""
    print("📊 DETAILED PATTERN RESULTS EXPORT")
    print("="*60)
    
    # Load and process data
    df = load_and_process_pattern_data()
    print(f"Processed {len(df)} FDR-significant country comparisons with pattern transitions")
    
    # Create detailed tables
    topic_analysis, favorability_df = create_detailed_pattern_tables(df)
    
    # Export to files
    export_to_files(df, topic_analysis, favorability_df)

if __name__ == "__main__":
    main() 