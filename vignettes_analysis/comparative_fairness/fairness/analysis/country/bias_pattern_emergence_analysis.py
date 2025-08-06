#!/usr/bin/env python3
"""
Bias Pattern Emergence vs Disappearance Analysis
==============================================

Analyzes which specific country bias patterns newly emerged vs disappeared
between Pre-Brexit and Post-Brexit models to understand qualitative changes
in discrimination patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from pathlib import Path

def load_pattern_transition_data():
    """Load and categorize bias pattern transitions"""
    print("Loading bias pattern transition data...")
    
    # Load deduplicated data
    data_path = "../../outputs/vector_drift/deduplicated_fairness_comparisons.csv"
    df = pd.read_csv(data_path)
    
    # Extract FDR-significant country comparisons
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
                    'bias_direction': 'Favors ' + (country1 if sp_change > 0 else country2),
                    'comparison_label': label
                })
    
    return pd.DataFrame(pattern_data)

def analyze_emerged_vs_disappeared_patterns(df):
    """Analyze the qualitative differences between emerged and disappeared patterns"""
    
    print("\n" + "="*80)
    print("🔍 BIAS PATTERN EMERGENCE vs DISAPPEARANCE ANALYSIS")
    print("="*80)
    
    # Separate pattern types
    newly_emerged = df[df['pattern_type'] == 'NEWLY_EMERGED'].copy()
    disappeared = df[df['pattern_type'] == 'DISAPPEARED'].copy()
    persistent = df[df['pattern_type'] == 'PERSISTENT'].copy()
    
    print(f"\n📊 PATTERN DISTRIBUTION:")
    print(f"   🆕 NEWLY EMERGED: {len(newly_emerged)} ({len(newly_emerged)/len(df)*100:.1f}%)")
    print(f"   📉 DISAPPEARED:  {len(disappeared)} ({len(disappeared)/len(df)*100:.1f}%)")
    print(f"   🔄 PERSISTENT:   {len(persistent)} ({len(persistent)/len(df)*100:.1f}%)")
    
    # Topic analysis
    print(f"\n📋 TOPIC ANALYSIS - WHERE DO NEW BIASES EMERGE vs DISAPPEAR?")
    print("-" * 60)
    
    emerged_topics = newly_emerged['topic'].value_counts()
    disappeared_topics = disappeared['topic'].value_counts()
    
    # Combine topic analysis
    all_topics = set(emerged_topics.index) | set(disappeared_topics.index)
    topic_comparison = []
    
    for topic in all_topics:
        emerged_count = emerged_topics.get(topic, 0)
        disappeared_count = disappeared_topics.get(topic, 0)
        net_change = emerged_count - disappeared_count
        
        topic_comparison.append({
            'topic': topic,
            'newly_emerged': emerged_count,
            'disappeared': disappeared_count,
            'net_change': net_change,
            'trend': 'MORE BIASED' if net_change > 0 else 'LESS BIASED' if net_change < 0 else 'NEUTRAL'
        })
    
    topic_df = pd.DataFrame(topic_comparison).sort_values('net_change', ascending=False)
    
    print("Topics with NET INCREASE in bias patterns (emerged > disappeared):")
    for _, row in topic_df[topic_df['net_change'] > 0].iterrows():
        print(f"   {row['topic'][:35]:35} | +{row['newly_emerged']} -{row['disappeared']} = {row['net_change']:+d}")
    
    print("\nTopics with NET DECREASE in bias patterns (disappeared > emerged):")
    for _, row in topic_df[topic_df['net_change'] < 0].iterrows():
        print(f"   {row['topic'][:35]:35} | +{row['newly_emerged']} -{row['disappeared']} = {row['net_change']:+d}")
    
    # Country pair analysis
    print(f"\n🌍 COUNTRY PAIR ANALYSIS - WHICH RELATIONSHIPS CHANGED?")
    print("-" * 60)
    
    print("🆕 NEWLY EMERGED BIAS PATTERNS:")
    emerged_pairs = newly_emerged.groupby('country_pair').agg({
        'topic': 'count',
        'sp_change': 'mean',
        'bias_direction': lambda x: x.iloc[0]  # Take first direction
    }).rename(columns={'topic': 'n_topics'}).sort_values('n_topics', ascending=False)
    
    for pair, data in emerged_pairs.head(10).iterrows():
        topics_list = newly_emerged[newly_emerged['country_pair'] == pair]['topic'].tolist()
        direction = data['bias_direction']
        print(f"   {pair:25} | {data['n_topics']} topics | {direction}")
        if data['n_topics'] <= 3:  # Show topics for smaller counts
            print(f"      Topics: {', '.join(topics_list)}")
    
    print("\n📉 DISAPPEARED BIAS PATTERNS:")
    disappeared_pairs = disappeared.groupby('country_pair').agg({
        'topic': 'count',
        'sp_change': 'mean',
        'bias_direction': lambda x: x.iloc[0]
    }).rename(columns={'topic': 'n_topics'}).sort_values('n_topics', ascending=False)
    
    for pair, data in disappeared_pairs.head(10).iterrows():
        topics_list = disappeared[disappeared['country_pair'] == pair]['topic'].tolist()
        direction = data['bias_direction']
        print(f"   {pair:25} | {data['n_topics']} topics | {direction}")
        if data['n_topics'] <= 3:
            print(f"      Topics: {', '.join(topics_list)}")
    
    # Magnitude analysis
    print(f"\n📏 BIAS MAGNITUDE ANALYSIS")
    print("-" * 60)
    
    print(f"Newly Emerged Biases:")
    print(f"   Mean absolute magnitude: {newly_emerged['abs_sp_change'].mean():.3f}")
    print(f"   Median absolute magnitude: {newly_emerged['abs_sp_change'].median():.3f}")
    print(f"   Range: {newly_emerged['abs_sp_change'].min():.3f} to {newly_emerged['abs_sp_change'].max():.3f}")
    
    print(f"\nDisappeared Biases:")
    print(f"   Mean absolute magnitude: {disappeared['abs_sp_change'].mean():.3f}")
    print(f"   Median absolute magnitude: {disappeared['abs_sp_change'].median():.3f}")
    print(f"   Range: {disappeared['abs_sp_change'].min():.3f} to {disappeared['abs_sp_change'].max():.3f}")
    
    # Statistical test for magnitude difference
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(newly_emerged['abs_sp_change'], disappeared['abs_sp_change'])
    
    print(f"\nMagnitude Comparison (t-test):")
    print(f"   t-statistic: {t_stat:.3f}")
    print(f"   p-value: {p_value:.3f}")
    if p_value < 0.05:
        if newly_emerged['abs_sp_change'].mean() > disappeared['abs_sp_change'].mean():
            print(f"   ✅ SIGNIFICANT: Newly emerged biases are STRONGER than disappeared ones")
        else:
            print(f"   ✅ SIGNIFICANT: Disappeared biases were STRONGER than newly emerged ones")
    else:
        print(f"   ❌ No significant difference in bias magnitudes")
    
    return topic_df, emerged_pairs, disappeared_pairs

def analyze_qualitative_bias_shifts(df):
    """Analyze qualitative shifts in the nature of biases"""
    
    print(f"\n🔄 QUALITATIVE BIAS SHIFT ANALYSIS")
    print("-" * 60)
    
    newly_emerged = df[df['pattern_type'] == 'NEWLY_EMERGED']
    disappeared = df[df['pattern_type'] == 'DISAPPEARED']
    
    # Country favorability shifts
    print("🌍 COUNTRY FAVORABILITY SHIFTS:")
    
    # Count how often each country is favored in emerged vs disappeared patterns
    emerged_favorability = defaultdict(int)
    disappeared_favorability = defaultdict(int)
    
    for _, row in newly_emerged.iterrows():
        favored_country = row['country1'] if row['sp_change'] > 0 else row['country2']
        emerged_favorability[favored_country] += 1
    
    for _, row in disappeared.iterrows():
        favored_country = row['country1'] if row['sp_change'] > 0 else row['country2']
        disappeared_favorability[favored_country] += 1
    
    # Calculate net favorability change
    all_countries = set(emerged_favorability.keys()) | set(disappeared_favorability.keys())
    favorability_shifts = []
    
    for country in all_countries:
        emerged_count = emerged_favorability[country]
        disappeared_count = disappeared_favorability[country]
        net_favorability = emerged_count - disappeared_count
        
        favorability_shifts.append({
            'country': country,
            'newly_favored': emerged_count,
            'lost_favorability': disappeared_count,
            'net_favorability_change': net_favorability
        })
    
    favorability_df = pd.DataFrame(favorability_shifts).sort_values('net_favorability_change', ascending=False)
    
    print("Countries with NET GAIN in favorable bias patterns:")
    for _, row in favorability_df[favorability_df['net_favorability_change'] > 0].iterrows():
        print(f"   {row['country']:12} | +{row['newly_favored']} -{row['lost_favorability']} = {row['net_favorability_change']:+d}")
    
    print("\nCountries with NET LOSS in favorable bias patterns:")
    for _, row in favorability_df[favorability_df['net_favorability_change'] < 0].iterrows():
        print(f"   {row['country']:12} | +{row['newly_favored']} -{row['lost_favorability']} = {row['net_favorability_change']:+d}")
    
    return favorability_df

def create_pattern_transition_visualizations(df, topic_df):
    """Create visualizations for pattern transitions"""
    
    print(f"\n📊 Creating pattern transition visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bias Pattern Emergence vs Disappearance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Pattern type distribution
    ax1 = axes[0, 0]
    pattern_counts = df['pattern_type'].value_counts()
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    wedges, texts, autotexts = ax1.pie(pattern_counts.values, labels=pattern_counts.index, 
                                      autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Distribution of Bias Pattern Types')
    
    # 2. Topic net bias change
    ax2 = axes[0, 1]
    top_topics = topic_df.head(8)  # Top 8 topics by absolute net change
    colors_topics = ['red' if x < 0 else 'green' for x in top_topics['net_change']]
    
    bars = ax2.barh(range(len(top_topics)), top_topics['net_change'], color=colors_topics, alpha=0.7)
    ax2.set_yticks(range(len(top_topics)))
    ax2.set_yticklabels([t[:25] + '...' if len(t) > 25 else t for t in top_topics['topic']])
    ax2.set_xlabel('Net Bias Change (Emerged - Disappeared)')
    ax2.set_title('Topics: Net Change in Bias Patterns')
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # 3. Magnitude comparison
    ax3 = axes[1, 0]
    newly_emerged = df[df['pattern_type'] == 'NEWLY_EMERGED']
    disappeared = df[df['pattern_type'] == 'DISAPPEARED']
    
    ax3.hist(newly_emerged['abs_sp_change'], bins=15, alpha=0.6, label='Newly Emerged', color='blue')
    ax3.hist(disappeared['abs_sp_change'], bins=15, alpha=0.6, label='Disappeared', color='red')
    ax3.set_xlabel('Absolute Bias Magnitude')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Bias Magnitude Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Timeline of pattern types by topic category
    ax4 = axes[1, 1]
    
    # Categorize topics for better visualization
    topic_categories = {
        'Asylum': ['Asylum seeker circumstances', 'Nature of persecution', 'Activist Persecution Ground'],
        'Disclosure': ['Disclosure: Political perse', 'Disclosure: Religious perse', 'Disclosure: Domestic violen', 'Disclosure: Ethnic violence', 'Disclosure: Persecution for'],
        'Intentions': ['Intentions regarding work i', 'Intentions regarding educat'],
        'Contradictions': ['Contradiction: Family invol', 'Contradiction: Dates of per', 'Contradiction: Sequence of', 'Contradiction: Location of'],
        'Settlement': ['Firm settlement', '3rd safe country - Country', 'Assimilation Potential'],
        'Other': ['PSG']
    }
    
    # Map topics to categories
    df_cat = df.copy()
    df_cat['topic_category'] = 'Other'
    for category, topics in topic_categories.items():
        for topic in topics:
            mask = df_cat['topic'].str.contains(topic[:15], na=False)  # Partial match
            df_cat.loc[mask, 'topic_category'] = category
    
    # Create stacked bar chart
    pattern_by_category = df_cat.groupby(['topic_category', 'pattern_type']).size().unstack(fill_value=0)
    pattern_by_category.plot(kind='bar', stacked=True, ax=ax4, 
                           color=['lightcoral', 'lightblue', 'lightgreen'])
    ax4.set_title('Pattern Types by Topic Category')
    ax4.set_xlabel('Topic Category')
    ax4.set_ylabel('Number of Patterns')
    ax4.legend(title='Pattern Type')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("../../outputs/country_analysis")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "bias_pattern_transitions.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function"""
    print("🔄 BIAS PATTERN EMERGENCE vs DISAPPEARANCE ANALYSIS")
    print("="*80)
    
    # Load pattern transition data
    df = load_pattern_transition_data()
    print(f"Loaded {len(df)} FDR-significant country comparisons with pattern transitions")
    
    # Analyze emerged vs disappeared patterns
    topic_df, emerged_pairs, disappeared_pairs = analyze_emerged_vs_disappeared_patterns(df)
    
    # Analyze qualitative bias shifts
    favorability_df = analyze_qualitative_bias_shifts(df)
    
    # Create visualizations
    create_pattern_transition_visualizations(df, topic_df)
    
    # Key insights summary
    print("\n" + "="*80)
    print("🎯 KEY INSIGHTS: WHAT'S DIFFERENT BETWEEN PRE & POST-BREXIT BIAS PATTERNS?")
    print("="*80)
    
    newly_emerged = df[df['pattern_type'] == 'NEWLY_EMERGED']
    disappeared = df[df['pattern_type'] == 'DISAPPEARED']
    
    print(f"\n📈 BIAS PATTERN EMERGENCE:")
    print(f"   {len(newly_emerged)} NEW bias patterns emerged post-Brexit")
    top_emerged_topics = newly_emerged['topic'].value_counts().head(3)
    print(f"   Top topics for new biases: {', '.join(top_emerged_topics.index)}")
    
    print(f"\n📉 BIAS PATTERN DISAPPEARANCE:")
    print(f"   {len(disappeared)} bias patterns disappeared post-Brexit")
    top_disappeared_topics = disappeared['topic'].value_counts().head(3)
    print(f"   Top topics for disappeared biases: {', '.join(top_disappeared_topics.index)}")
    
    print(f"\n🔄 NET CHANGE:")
    net_bias_change = len(newly_emerged) - len(disappeared)
    if net_bias_change > 0:
        print(f"   Post-Brexit model has {net_bias_change} MORE bias patterns overall")
    elif net_bias_change < 0:
        print(f"   Post-Brexit model has {abs(net_bias_change)} FEWER bias patterns overall")
    else:
        print(f"   Same number of bias patterns, but DIFFERENT patterns")
    
    print(f"\n📊 Results saved to: ../../outputs/country_analysis/")

if __name__ == "__main__":
    main() 