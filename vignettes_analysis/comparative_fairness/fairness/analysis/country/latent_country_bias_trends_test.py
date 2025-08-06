#!/usr/bin/env python3
"""
Latent Country Bias Trends Test
==============================

Statistical test to identify systematic bias patterns in FDR-significant
country comparisons across topics:

1. Systematic bias direction changes (countries becoming consistently more/less favored)
2. Persistent bias amplification (existing biases getting stronger)  
3. Cross-topic consistency patterns (same country pairs biased across multiple topics)
4. Geopolitical clustering effects
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from pathlib import Path

def load_fdr_significant_country_data():
    """Load FDR-significant country comparisons"""
    print("Loading FDR-significant country comparisons...")
    
    data_path = "../../outputs/vector_drift/deduplicated_fairness_comparisons.csv"
    df = pd.read_csv(data_path)
    
    # Filter for country comparisons that are FDR-significant
    country_data = []
    
    for _, row in df.iterrows():
        # Parse comparison label format: "Country1_vs_Country2 (attribute) [topic]"
        label = row['comparison_label']
        
        # Check if it's a country comparison
        if '(country)' in label:
            # Check if significant in either model
            if row['pre_brexit_sp_significance'] or row['post_brexit_sp_significance']:
                
                # Extract components from label
                # Split by ' (' to separate comparison from attribute
                comparison_part = label.split(' (')[0]
                remaining = label.split(' (')[1] if ' (' in label else ""
                
                # Extract topic from [topic] format
                topic = ""
                if '[' in remaining and ']' in remaining:
                    topic = remaining.split('[')[1].split(']')[0]
                
                # Parse comparison
                parts = comparison_part.split('_vs_')
                if len(parts) == 2:
                    country1, country2 = parts
                    
                    # Calculate sp_change from magnitude difference
                    sp_change = row['sp_magnitude_difference'] 
                    
                    country_data.append({
                        'country1': country1,
                        'country2': country2,
                        'topic': topic,
                        'pre_sp': row['pre_brexit_sp_magnitude'],
                        'post_sp': row['post_brexit_sp_magnitude'],
                        'sp_change': sp_change,
                        'pre_significant': row['pre_brexit_sp_significance'],
                        'post_significant': row['post_brexit_sp_significance'],
                        'pattern': get_significance_pattern(row),
                        'direction': get_bias_direction(sp_change)
                    })
    
    return pd.DataFrame(country_data)

def get_significance_pattern(row):
    """Determine significance pattern"""
    pre_sig = row['pre_brexit_sp_significance']
    post_sig = row['post_brexit_sp_significance']
    
    if pre_sig and post_sig:
        return 'PERSISTENT'
    elif not pre_sig and post_sig:
        return 'NEWLY_EMERGED'
    elif pre_sig and not post_sig:
        return 'DISAPPEARED'
    else:
        return 'NEITHER'  # Shouldn't happen in our filtered data

def get_bias_direction(sp_change):
    """Categorize bias direction change"""
    if abs(sp_change) < 0.05:
        return 'MINIMAL'
    elif sp_change > 0:
        return 'FAVORS_COUNTRY1'
    else:
        return 'FAVORS_COUNTRY2'

def test_systematic_bias_trends(df):
    """Test for systematic bias trends across topics"""
    print("\n" + "="*80)
    print("🔍 SYSTEMATIC BIAS TRENDS ANALYSIS")
    print("="*80)
    
    # 1. Country-level aggregation: Are certain countries systematically favored/disfavored?
    print("\n📊 1. SYSTEMATIC COUNTRY FAVORABILITY TRENDS")
    print("-" * 50)
    
    country_trends = defaultdict(list)
    
    # Collect all bias changes for each country (as both country1 and country2)
    for _, row in df.iterrows():
        c1, c2 = row['country1'], row['country2']
        sp_change = row['sp_change']
        
        # For country1: positive sp_change means country1 is more favored
        country_trends[c1].append(sp_change)
        # For country2: negative sp_change means country2 is more favored  
        country_trends[c2].append(-sp_change)
    
    # Calculate systematic trends
    country_bias_summary = []
    for country, changes in country_trends.items():
        if len(changes) >= 3:  # Only countries with multiple comparisons
            mean_change = np.mean(changes)
            median_change = np.median(changes)
            consistency = np.sum(np.sign(changes) == np.sign(mean_change)) / len(changes)
            
            # One-sample t-test: is the mean significantly different from 0?
            t_stat, p_value = stats.ttest_1samp(changes, 0)
            
            country_bias_summary.append({
                'country': country,
                'n_comparisons': len(changes),
                'mean_bias_change': mean_change,
                'median_bias_change': median_change,
                'consistency_rate': consistency,
                't_statistic': t_stat,
                'p_value': p_value,
                'trend_strength': 'STRONG' if abs(mean_change) > 0.1 and p_value < 0.05 else 
                                'MODERATE' if abs(mean_change) > 0.05 else 'WEAK'
            })
    
    country_summary_df = pd.DataFrame(country_bias_summary).sort_values('mean_bias_change')
    
    print("Top Systematically DISFAVORED countries (negative bias):")
    disfavored = country_summary_df.head(5)
    for _, row in disfavored.iterrows():
        direction = "📉 DISFAVORED" if row['mean_bias_change'] < 0 else "📈 FAVORED"
        print(f"  {row['country']:12} | {row['mean_bias_change']:+.3f} | {row['consistency_rate']:.1%} consistency | p={row['p_value']:.3f} | {row['trend_strength']}")
    
    print("\nTop Systematically FAVORED countries (positive bias):")
    favored = country_summary_df.tail(5)
    for _, row in favored.iterrows():
        direction = "📉 DISFAVORED" if row['mean_bias_change'] < 0 else "📈 FAVORED"
        print(f"  {row['country']:12} | {row['mean_bias_change']:+.3f} | {row['consistency_rate']:.1%} consistency | p={row['p_value']:.3f} | {row['trend_strength']}")
    
    # 2. Country pair consistency: Same pairs biased across multiple topics?
    print(f"\n📊 2. CROSS-TOPIC COUNTRY PAIR CONSISTENCY")
    print("-" * 50)
    
    pair_consistency = defaultdict(list)
    for _, row in df.iterrows():
        pair_key = f"{row['country1']}_vs_{row['country2']}"
        pair_consistency[pair_key].append({
            'topic': row['topic'],
            'sp_change': row['sp_change'],
            'pattern': row['pattern'],
            'direction': row['direction']
        })
    
    # Analyze pairs with multiple topics
    multi_topic_pairs = {k: v for k, v in pair_consistency.items() if len(v) >= 2}
    
    print(f"Found {len(multi_topic_pairs)} country pairs with bias across multiple topics:")
    
    for pair, topics_data in sorted(multi_topic_pairs.items(), 
                                   key=lambda x: len(x[1]), reverse=True):
        if len(topics_data) >= 2:
            changes = [t['sp_change'] for t in topics_data]
            directions = [t['direction'] for t in topics_data]
            patterns = [t['pattern'] for t in topics_data]
            
            # Consistency metrics
            direction_consistency = len(set(directions)) == 1 if len(directions) > 1 else True
            mean_change = np.mean(changes)
            
            print(f"\n  🎯 {pair} ({len(topics_data)} topics):")
            print(f"     Mean bias change: {mean_change:+.3f}")
            print(f"     Direction consistency: {'✅ CONSISTENT' if direction_consistency else '❌ INCONSISTENT'}")
            
            for topic_data in topics_data:
                print(f"     • {topic_data['topic'][:30]:30} | {topic_data['sp_change']:+.3f} | {topic_data['pattern']}")
    
    # 3. Bias amplification: Persistent biases getting stronger
    print(f"\n📊 3. BIAS AMPLIFICATION ANALYSIS")
    print("-" * 50)
    
    persistent_biases = df[df['pattern'] == 'PERSISTENT'].copy()
    if len(persistent_biases) > 0:
        print(f"Analyzing {len(persistent_biases)} persistent biases for amplification...")
        
        # Calculate amplification (absolute bias getting stronger)
        persistent_biases['pre_abs'] = abs(persistent_biases['pre_sp'])
        persistent_biases['post_abs'] = abs(persistent_biases['post_sp'])
        persistent_biases['amplification'] = persistent_biases['post_abs'] - persistent_biases['pre_abs']
        
        # Test if amplification is systematic
        amplification_values = persistent_biases['amplification'].values
        t_stat, p_value = stats.ttest_1samp(amplification_values, 0)
        
        print(f"Mean amplification: {np.mean(amplification_values):+.3f}")
        print(f"Amplification t-test: t={t_stat:.3f}, p={p_value:.3f}")
        
        if p_value < 0.05:
            if np.mean(amplification_values) > 0:
                print("✅ SIGNIFICANT: Persistent biases are systematically AMPLIFYING")
            else:
                print("✅ SIGNIFICANT: Persistent biases are systematically WEAKENING")
        else:
            print("❌ No significant systematic amplification/weakening")
        
        # Show top amplifications
        top_amplified = persistent_biases.nlargest(5, 'amplification')
        print("\nTop 5 amplified persistent biases:")
        for _, row in top_amplified.iterrows():
            print(f"  {row['country1']} vs {row['country2']:12} | {row['topic'][:25]:25} | {row['amplification']:+.3f}")
    
    # 4. Pattern transition analysis
    print(f"\n📊 4. BIAS PATTERN TRANSITIONS")
    print("-" * 50)
    
    pattern_counts = df['pattern'].value_counts()
    print("Pattern distribution:")
    for pattern, count in pattern_counts.items():
        pct = count / len(df) * 100
        print(f"  {pattern:15} | {count:3d} ({pct:.1f}%)")
    
    return country_summary_df, multi_topic_pairs, persistent_biases

def create_visualizations(df, country_summary_df):
    """Create visualizations for bias trends"""
    print(f"\n📊 Creating bias trend visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Country Bias Trends Analysis - FDR Significant Comparisons', fontsize=16, fontweight='bold')
    
    # 1. Country systematic bias trends
    ax1 = axes[0, 0]
    top_countries = pd.concat([
        country_summary_df.head(5),  # Most disfavored
        country_summary_df.tail(5)   # Most favored
    ])
    
    colors = ['red' if x < 0 else 'blue' for x in top_countries['mean_bias_change']]
    bars = ax1.barh(range(len(top_countries)), top_countries['mean_bias_change'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(top_countries)))
    ax1.set_yticklabels(top_countries['country'])
    ax1.set_xlabel('Mean Bias Change')
    ax1.set_title('Systematic Country Favorability Trends')
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Add significance indicators
    for i, (_, row) in enumerate(top_countries.iterrows()):
        if row['p_value'] < 0.05:
            ax1.text(row['mean_bias_change'] + (0.01 if row['mean_bias_change'] > 0 else -0.01), 
                    i, '*', ha='left' if row['mean_bias_change'] > 0 else 'right', 
                    va='center', fontsize=12, fontweight='bold')
    
    # 2. Bias pattern distribution
    ax2 = axes[0, 1]
    pattern_counts = df['pattern'].value_counts()
    colors_pattern = ['lightcoral', 'lightblue', 'lightgreen']
    wedges, texts, autotexts = ax2.pie(pattern_counts.values, labels=pattern_counts.index, 
                                      autopct='%1.1f%%', colors=colors_pattern)
    ax2.set_title('Bias Pattern Distribution')
    
    # 3. Bias change magnitude by topic
    ax3 = axes[1, 0]
    topic_stats = df.groupby('topic').agg({
        'sp_change': ['mean', 'std', 'count']
    }).round(3)
    topic_stats.columns = ['mean_change', 'std_change', 'count']
    topic_stats = topic_stats[topic_stats['count'] >= 2].sort_values('mean_change')
    
    if len(topic_stats) > 0:
        y_pos = range(len(topic_stats))
        ax3.barh(y_pos, topic_stats['mean_change'], 
                xerr=topic_stats['std_change'], alpha=0.7, color='orange')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([t[:25] + '...' if len(t) > 25 else t for t in topic_stats.index])
        ax3.set_xlabel('Mean Bias Change')
        ax3.set_title('Average Bias Change by Topic')
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3)
    
    # 4. Amplification analysis for persistent biases
    ax4 = axes[1, 1]
    persistent_biases = df[df['pattern'] == 'PERSISTENT'].copy()
    if len(persistent_biases) > 0:
        persistent_biases['amplification'] = abs(persistent_biases['post_sp']) - abs(persistent_biases['pre_sp'])
        
        ax4.hist(persistent_biases['amplification'], bins=15, alpha=0.7, color='purple', edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Change')
        ax4.axvline(x=persistent_biases['amplification'].mean(), color='orange', 
                   linestyle='-', linewidth=2, label=f'Mean: {persistent_biases["amplification"].mean():.3f}')
        ax4.set_xlabel('Bias Amplification (|Post| - |Pre|)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Persistent Bias Amplification')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No persistent biases found', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Persistent Bias Amplification')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("../../outputs/country_analysis")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "country_bias_trends_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function"""
    print("🌍 LATENT COUNTRY BIAS TRENDS TEST")
    print("="*80)
    
    # Load data
    df = load_fdr_significant_country_data()
    print(f"Loaded {len(df)} FDR-significant country comparisons")
    
    # Run systematic bias trends analysis
    country_summary_df, multi_topic_pairs, persistent_biases = test_systematic_bias_trends(df)
    
    # Create visualizations
    create_visualizations(df, country_summary_df)
    
    # Summary conclusions
    print("\n" + "="*80)
    print("🎯 KEY CONCLUSIONS FROM LATENT BIAS TRENDS ANALYSIS")
    print("="*80)
    
    # Statistical significance test for overall bias direction
    all_changes = df['sp_change'].values
    t_stat, p_value = stats.ttest_1samp(all_changes, 0)
    
    print(f"\n📊 OVERALL BIAS TREND:")
    print(f"   Mean bias change across all comparisons: {np.mean(all_changes):+.3f}")
    print(f"   One-sample t-test vs 0: t={t_stat:.3f}, p={p_value:.3f}")
    
    if p_value < 0.05:
        direction = "MORE FAVORABLE" if np.mean(all_changes) > 0 else "LESS FAVORABLE"
        print(f"   ✅ SIGNIFICANT: Post-Brexit model is systematically {direction} in country comparisons")
    else:
        print(f"   ❌ No significant overall bias direction change")
    
    print(f"\n📈 SYSTEMATIC COUNTRY EFFECTS:")
    strong_trends = country_summary_df[
        (abs(country_summary_df['mean_bias_change']) > 0.1) & 
        (country_summary_df['p_value'] < 0.05)
    ]
    
    if len(strong_trends) > 0:
        print(f"   Found {len(strong_trends)} countries with strong systematic bias trends")
    else:
        print(f"   No countries show strong systematic bias trends")
    
    print(f"\n🔄 CROSS-TOPIC CONSISTENCY:")
    consistent_pairs = [pair for pair, data in multi_topic_pairs.items() 
                       if len(set([d['direction'] for d in data])) == 1 and len(data) >= 2]
    print(f"   {len(consistent_pairs)} country pairs show consistent bias direction across topics")
    
    print(f"\n📊 Results saved to: ../../outputs/country_analysis/")

if __name__ == "__main__":
    main() 