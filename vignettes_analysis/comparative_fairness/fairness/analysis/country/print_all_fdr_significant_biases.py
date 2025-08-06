#!/usr/bin/env python3
"""
Print All FDR-Significant Biases - Prettified Display
====================================================

Extract and display all FDR-significant bias comparisons in a beautiful,
readable format for human review.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_all_fdr_significant_comparisons():
    """Load all FDR-significant comparisons from deduplicated data"""
    print("Loading all FDR-significant comparisons...")
    
    data_path = "../../outputs/vector_drift/deduplicated_fairness_comparisons.csv"
    df = pd.read_csv(data_path)
    
    # Parse all comparison labels
    parsed_data = []
    
    for _, row in df.iterrows():
        label = row['comparison_label']
        
        # Parse label format: "Group1_vs_Group2 (attribute) [topic]"
        try:
            if '(' in label and ')' in label and '[' in label:
                main_part = label.split(' (')[0]
                remainder = label.split(' (')[1]
                attribute = remainder.split(')')[0]
                topic = remainder.split('[')[1].split(']')[0]
                
                if '_vs_' in main_part:
                    group1, group2 = main_part.split('_vs_')
                    
                    # Check if FDR significant (either model)
                    is_significant = (row['pre_brexit_sp_significance'] or row['post_brexit_sp_significance'])
                    
                    if is_significant:
                        parsed_data.append({
                            'comparison_label': label,
                            'group1': group1,
                            'group2': group2,
                            'protected_attribute': attribute,
                            'topic': topic,
                            'pre_brexit_sp': row['pre_brexit_sp_magnitude'],
                            'post_brexit_sp': row['post_brexit_sp_magnitude'],
                            'sp_change': row['sp_magnitude_difference'],
                            'abs_sp_change': abs(row['sp_magnitude_difference']),
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
    
    significant_df = pd.DataFrame(parsed_data)
    print(f"Found {len(significant_df)} FDR-significant comparisons")
    
    return significant_df

def categorize_significance_patterns(df):
    """Categorize significance patterns"""
    df = df.copy()
    
    # Create significance pattern categories
    def get_pattern(row):
        if row['both_significant']:
            return "🔄 PERSISTENT"
        elif row['gained_significance']:
            return "🆕 NEWLY EMERGED"
        elif row['lost_significance']:
            return "📉 DISAPPEARED"
        elif row['pre_brexit_significant']:
            return "📍 PRE-BREXIT ONLY"
        elif row['post_brexit_significant']:
            return "📍 POST-BREXIT ONLY"
        else:
            return "❓ UNKNOWN"
    
    df['significance_pattern'] = df.apply(get_pattern, axis=1)
    
    # Create bias direction categories
    def get_bias_direction(row):
        if abs(row['sp_change']) < 0.05:
            return "➡️ MINIMAL"
        elif row['sp_change'] > 0:
            return f"📈 FAVORS {row['group1']}"
        else:
            return f"📉 FAVORS {row['group2']}"
    
    df['bias_direction'] = df.apply(get_bias_direction, axis=1)
    
    # Create magnitude categories
    def get_magnitude(abs_change):
        if abs_change < 0.05:
            return "🔸 SMALL"
        elif abs_change < 0.1:
            return "🔹 MEDIUM"
        elif abs_change < 0.2:
            return "🔶 LARGE"
        else:
            return "🔴 VERY LARGE"
    
    df['magnitude_category'] = df['abs_sp_change'].apply(get_magnitude)
    
    return df

def print_summary_statistics(df):
    """Print summary statistics"""
    print(f"\n{'='*100}")
    print(f"📊 FDR-SIGNIFICANT BIAS SUMMARY STATISTICS")
    print(f"{'='*100}")
    
    print(f"🎯 TOTAL FDR-SIGNIFICANT COMPARISONS: {len(df)}")
    print(f"📈 Mean absolute bias change: {df['abs_sp_change'].mean():.3f}")
    print(f"📊 Median absolute bias change: {df['abs_sp_change'].median():.3f}")
    print(f"🔴 Largest bias change: {df['abs_sp_change'].max():.3f}")
    
    print(f"\n📋 BY PROTECTED ATTRIBUTE:")
    attr_counts = df.groupby('protected_attribute').size().sort_values(ascending=False)
    for attr, count in attr_counts.items():
        pct = count / len(df) * 100
        print(f"   {attr.upper():12s}: {count:3d} ({pct:5.1f}%)")
    
    print(f"\n🔄 BY SIGNIFICANCE PATTERN:")
    pattern_counts = df.groupby('significance_pattern').size().sort_values(ascending=False)
    for pattern, count in pattern_counts.items():
        pct = count / len(df) * 100
        print(f"   {pattern:18s}: {count:3d} ({pct:5.1f}%)")
    
    print(f"\n📏 BY MAGNITUDE:")
    magnitude_counts = df.groupby('magnitude_category').size().sort_values(ascending=False)
    for magnitude, count in magnitude_counts.items():
        pct = count / len(df) * 100
        print(f"   {magnitude:15s}: {count:3d} ({pct:5.1f}%)")

def print_top_biases_by_category(df):
    """Print top biases by different categories"""
    
    print(f"\n{'='*100}")
    print(f"🔥 TOP 10 LARGEST BIAS CHANGES")
    print(f"{'='*100}")
    
    top_biases = df.nlargest(10, 'abs_sp_change')
    
    print(f"{'Rank':<4} | {'Comparison':<25} | {'Attribute':<8} | {'Topic':<35} | {'SP-Δ':>8} | {'Pattern':<18}")
    print(f"{'-'*4}-+-{'-'*25}-+-{'-'*8}-+-{'-'*35}-+-{'-'*8}-+-{'-'*18}")
    
    for i, (_, row) in enumerate(top_biases.iterrows(), 1):
        comparison = f"{row['group1']} vs {row['group2']}"[:24]
        topic = row['topic'][:34]
        print(f"{i:>3}. | {comparison:<25} | {row['protected_attribute']:<8} | {topic:<35} | {row['sp_change']:>+7.3f} | {row['significance_pattern']}")

def print_detailed_comparisons_by_attribute(df):
    """Print detailed comparisons grouped by protected attribute"""
    
    for attribute in sorted(df['protected_attribute'].unique()):
        attr_df = df[df['protected_attribute'] == attribute].copy()
        attr_df = attr_df.sort_values('abs_sp_change', ascending=False)
        
        print(f"\n{'='*120}")
        print(f"🎯 {attribute.upper()} ATTRIBUTE - {len(attr_df)} FDR-SIGNIFICANT COMPARISONS")
        print(f"{'='*120}")
        
        print(f"{'Comparison':<25} | {'Topic':<40} | {'Pre-SP':>8} | {'Post-SP':>8} | {'SP-Δ':>8} | {'Pattern':<18} | {'Direction':<20}")
        print(f"{'-'*25}-+-{'-'*40}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*18}-+-{'-'*20}")
        
        for _, row in attr_df.iterrows():
            comparison = f"{row['group1']} vs {row['group2']}"[:24]
            topic = row['topic'][:39]
            pre_sp = row['pre_brexit_sp']
            post_sp = row['post_brexit_sp']
            sp_change = row['sp_change']
            pattern = row['significance_pattern']
            direction = row['bias_direction'][:19]
            
            print(f"{comparison:<25} | {topic:<40} | {pre_sp:>8.3f} | {post_sp:>8.3f} | {sp_change:>+8.3f} | {pattern:<18} | {direction:<20}")

def print_topic_analysis(df):
    """Print analysis by topic"""
    
    print(f"\n{'='*120}")
    print(f"📋 TOPIC-LEVEL ANALYSIS - FDR-SIGNIFICANT BIASES")
    print(f"{'='*120}")
    
    topic_stats = []
    
    for topic in df['topic'].unique():
        topic_data = df[df['topic'] == topic]
        
        stats = {
            'topic': topic,
            'count': len(topic_data),
            'attributes': topic_data['protected_attribute'].nunique(),
            'mean_abs_change': topic_data['abs_sp_change'].mean(),
            'max_abs_change': topic_data['abs_sp_change'].max(),
            'persistent': topic_data['both_significant'].sum(),
            'newly_emerged': topic_data['gained_significance'].sum(),
            'disappeared': topic_data['lost_significance'].sum()
        }
        topic_stats.append(stats)
    
    topic_stats_df = pd.DataFrame(topic_stats)
    topic_stats_df = topic_stats_df.sort_values('count', ascending=False)
    
    print(f"{'Topic':<45} | {'Count':>5} | {'Attrs':>5} | {'Mean|Δ|':>8} | {'Max|Δ|':>8} | {'Persistent':>9} | {'Emerged':>7} | {'Disappeared':>10}")
    print(f"{'-'*45}-+-{'-'*5}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*7}-+-{'-'*10}")
    
    for _, row in topic_stats_df.iterrows():
        topic_short = row['topic'][:44]
        count = int(row['count'])
        attrs = int(row['attributes'])
        mean_change = row['mean_abs_change']
        max_change = row['max_abs_change']
        persistent = int(row['persistent'])
        emerged = int(row['newly_emerged'])
        disappeared = int(row['disappeared'])
        
        print(f"{topic_short:<45} | {count:>5} | {attrs:>5} | {mean_change:>8.3f} | {max_change:>8.3f} | {persistent:>9} | {emerged:>7} | {disappeared:>10}")

def save_results(df):
    """Save the prettified results"""
    output_dir = Path("../../outputs/country_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    df.to_csv(output_dir / "all_fdr_significant_biases_detailed.csv", index=False)
    
    # Save summary by attribute
    summary_by_attr = df.groupby('protected_attribute').agg({
        'abs_sp_change': ['count', 'mean', 'max', 'std'],
        'both_significant': 'sum',
        'gained_significance': 'sum',
        'lost_significance': 'sum'
    }).round(3)
    
    summary_by_attr.to_csv(output_dir / "fdr_significant_summary_by_attribute.csv")
    
    print(f"\n✅ Results saved to: {output_dir}")

def main():
    """Main function to display all FDR-significant biases"""
    
    print("🎨 PRETTIFIED FDR-SIGNIFICANT BIAS DISPLAY")
    print("="*100)
    
    # Load data
    df = load_all_fdr_significant_comparisons()
    
    if len(df) == 0:
        print("❌ No FDR-significant comparisons found!")
        return
    
    # Categorize patterns
    df = categorize_significance_patterns(df)
    
    # Print analyses
    print_summary_statistics(df)
    print_top_biases_by_category(df)
    print_detailed_comparisons_by_attribute(df)
    print_topic_analysis(df)
    
    # Save results
    save_results(df)
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"Total FDR-significant biases displayed: {len(df)}")
    
    return df

if __name__ == "__main__":
    result_df = main() 