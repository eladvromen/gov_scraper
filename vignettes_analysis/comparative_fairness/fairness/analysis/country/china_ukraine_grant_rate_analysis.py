#!/usr/bin/env python3
"""
China vs Ukraine Grant Rate Analysis
===================================

Deep dive into China-Ukraine bias patterns with grant rate visualizations
for FDR-significant topics.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_raw_data():
    """Load raw tagged records to calculate grant rates"""
    print("Loading raw tagged records...")
    
    data_path = "../../../data/processed/tagged_records.json"
    with open(data_path, 'r') as f:
        records = json.load(f)
    
    print(f"Loaded {len(records)} raw records")
    return records

def load_china_ukraine_significant_topics():
    """Load FDR-significant China vs Ukraine comparisons"""
    print("Loading FDR-significant China vs Ukraine comparisons...")
    
    # Load deduplicated data
    data_path = "../../outputs/vector_drift/deduplicated_fairness_comparisons.csv"
    df = pd.read_csv(data_path)
    
    # Parse and filter for China vs Ukraine FDR-significant comparisons
    china_ukraine_data = []
    
    for _, row in df.iterrows():
        label = row['comparison_label']
        
        # Parse label format: "Group1_vs_Group2 (attribute) [topic]"
        if '(' in label and ')' in label and '[' in label:
            main_part = label.split(' (')[0]
            remainder = label.split(' (')[1]
            attribute = remainder.split(')')[0]
            topic = remainder.split('[')[1].split(']')[0]
            
            if '_vs_' in main_part:
                group1, group2 = main_part.split('_vs_')
                
                # Check if this is China vs Ukraine (either direction) and FDR significant
                is_china_ukraine = ((group1 == 'China' and group2 == 'Ukraine') or 
                                   (group1 == 'Ukraine' and group2 == 'China'))
                is_country = (attribute == 'country')
                is_significant = (row['pre_brexit_sp_significance'] or row['post_brexit_sp_significance'])
                
                if is_china_ukraine and is_country and is_significant:
                    china_ukraine_data.append({
                        'topic': topic,
                        'group1': group1,
                        'group2': group2,
                        'pre_brexit_sp': row['pre_brexit_sp_magnitude'],
                        'post_brexit_sp': row['post_brexit_sp_magnitude'],
                        'sp_change': row['sp_magnitude_difference'],
                        'pre_brexit_significant': row['pre_brexit_sp_significance'],
                        'post_brexit_significant': row['post_brexit_sp_significance'],
                        'gained_significance': row['sp_gained_significance'],
                        'lost_significance': row['sp_lost_significance'],
                        'both_significant': row['sp_both_significant']
                    })
    
    china_ukraine_df = pd.DataFrame(china_ukraine_data)
    print(f"Found {len(china_ukraine_df)} FDR-significant China vs Ukraine comparisons")
    
    return china_ukraine_df

def calculate_grant_rates(records, significant_topics):
    """Calculate grant rates for China and Ukraine by topic and model"""
    print("Calculating grant rates for China and Ukraine...")
    
    # Filter records for China and Ukraine
    china_ukraine_records = [
        r for r in records 
        if r.get('protected_attributes', {}).get('country') in ['China', 'Ukraine']
    ]
    
    print(f"Found {len(china_ukraine_records)} China/Ukraine records")
    
    # Calculate grant rates by country, topic, and model
    grant_rates = {}
    
    for record in china_ukraine_records:
        country = record.get('protected_attributes', {}).get('country')
        topic = record.get('topic')
        model = record.get('model')
        decision = record.get('decision', '').upper()
        
        # Only process records for topics that are FDR-significant
        if topic in significant_topics and country and model:
            key = (country, topic, model)
            
            if key not in grant_rates:
                grant_rates[key] = {'grants': 0, 'total': 0}
            
            grant_rates[key]['total'] += 1
            if decision == 'GRANT':
                grant_rates[key]['grants'] += 1
    
    # Convert to grant rate percentages
    final_grant_rates = {}
    for (country, topic, model), data in grant_rates.items():
        if data['total'] > 0:
            rate = data['grants'] / data['total']
            final_grant_rates[(country, topic, model)] = {
                'grant_rate': rate,
                'grants': data['grants'],
                'total': data['total']
            }
    
    print(f"Calculated grant rates for {len(final_grant_rates)} country-topic-model combinations")
    return final_grant_rates

def create_grant_rate_visualization(grant_rates, significant_df):
    """Create bar plot visualization of grant rates"""
    print("Creating grant rate visualization...")
    
    # Prepare data for plotting
    plot_data = []
    
    topics = significant_df['topic'].unique()
    countries = ['China', 'Ukraine']
    models = ['pre_brexit', 'post_brexit']
    
    for topic in topics:
        for country in countries:
            for model in models:
                key = (country, topic, model)
                if key in grant_rates:
                    data = grant_rates[key]
                    plot_data.append({
                        'topic': topic,
                        'country': country,
                        'model': model,
                        'grant_rate': data['grant_rate'] * 100,  # Convert to percentage
                        'grants': data['grants'],
                        'total': data['total'],
                        'sample_size': data['total']
                    })
                else:
                    # Add zero entry if no data
                    plot_data.append({
                        'topic': topic,
                        'country': country,
                        'model': model,
                        'grant_rate': 0,
                        'grants': 0,
                        'total': 0,
                        'sample_size': 0
                    })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create the visualization
    plt.style.use('default')
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('China vs Ukraine: Grant Rates Across FDR-Significant Topics\n(Pre-Brexit vs Post-Brexit Models)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    countries_colors = {'China': ['#d62728', '#ff7f7f'], 'Ukraine': ['#1f77b4', '#aec7e8']}
    
    for idx, country in enumerate(countries):
        ax = axes[idx]
        
        # Filter data for this country
        country_data = plot_df[plot_df['country'] == country]
        
        # Create grouped bar plot
        topics_list = sorted(country_data['topic'].unique())
        
        x_pos = np.arange(len(topics_list))
        width = 0.35
        
        pre_rates = []
        post_rates = []
        pre_samples = []
        post_samples = []
        
        for topic in topics_list:
            pre_data = country_data[(country_data['topic'] == topic) & 
                                  (country_data['model'] == 'pre_brexit')]
            post_data = country_data[(country_data['topic'] == topic) & 
                                   (country_data['model'] == 'post_brexit')]
            
            pre_rate = pre_data['grant_rate'].iloc[0] if len(pre_data) > 0 else 0
            post_rate = post_data['grant_rate'].iloc[0] if len(post_data) > 0 else 0
            pre_sample = pre_data['sample_size'].iloc[0] if len(pre_data) > 0 else 0
            post_sample = post_data['sample_size'].iloc[0] if len(post_data) > 0 else 0
            
            pre_rates.append(pre_rate)
            post_rates.append(post_rate)
            pre_samples.append(pre_sample)
            post_samples.append(post_sample)
        
        # Create bars
        bars1 = ax.bar(x_pos - width/2, pre_rates, width, 
                      label='Pre-Brexit', color=countries_colors[country][0], alpha=0.8)
        bars2 = ax.bar(x_pos + width/2, post_rates, width,
                      label='Post-Brexit', color=countries_colors[country][1], alpha=0.8)
        
        # Add sample size annotations
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            if pre_samples[i] > 0:
                ax.annotate(f'n={pre_samples[i]}', 
                           xy=(bar1.get_x() + bar1.get_width()/2, bar1.get_height()),
                           xytext=(0, 3), textcoords='offset points', 
                           ha='center', va='bottom', fontsize=8)
            if post_samples[i] > 0:
                ax.annotate(f'n={post_samples[i]}', 
                           xy=(bar2.get_x() + bar2.get_width()/2, bar2.get_height()),
                           xytext=(0, 3), textcoords='offset points', 
                           ha='center', va='bottom', fontsize=8)
        
        # Customize axes
        ax.set_title(f'{country} Grant Rates', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('Grant Rate (%)', fontsize=12)
        ax.set_xlabel('Topic', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([topic[:25] + '...' if len(topic) > 25 else topic 
                           for topic in topics_list], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(max(pre_rates + post_rates, default=0) * 1.1, 10))
        
        # Add bias direction annotations
        for i, topic in enumerate(topics_list):
            topic_bias = significant_df[significant_df['topic'] == topic]
            if len(topic_bias) > 0:
                bias_info = topic_bias.iloc[0]
                
                # Determine if China is favored or disfavored relative to Ukraine
                if bias_info['group1'] == 'China':
                    sp_value = bias_info['post_brexit_sp']
                    bias_direction = "Favored" if sp_value > 0 else "Disfavored"
                else:  # Ukraine vs China
                    sp_value = -bias_info['post_brexit_sp']  # Flip for China perspective
                    bias_direction = "Favored" if sp_value > 0 else "Disfavored"
                
                if country == 'China':
                    ax.annotate(f'{bias_direction}', 
                               xy=(i, max(pre_rates[i], post_rates[i]) + 1),
                               ha='center', va='bottom', fontsize=8, 
                               color='green' if bias_direction == 'Favored' else 'red',
                               fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("../../outputs/country_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / "china_ukraine_grant_rates_fdr_significant.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return plot_df

def print_detailed_analysis(significant_df, grant_rates):
    """Print detailed analysis of China vs Ukraine patterns"""
    print(f"\n{'='*80}")
    print("CHINA vs UKRAINE DETAILED ANALYSIS")
    print(f"{'='*80}")
    
    print(f"📊 FDR-SIGNIFICANT TOPICS: {len(significant_df)}")
    
    for _, row in significant_df.iterrows():
        topic = row['topic']
        
        print(f"\n🔍 TOPIC: {topic}")
        print(f"   Statistical Parity Change: {row['sp_change']:+.3f}")
        print(f"   Pre-Brexit SP: {row['pre_brexit_sp']:+.3f} | Post-Brexit SP: {row['post_brexit_sp']:+.3f}")
        
        # Determine direction
        if row['group1'] == 'China':
            if row['post_brexit_sp'] > 0:
                print(f"   ✅ China FAVORED over Ukraine in Post-Brexit model")
            else:
                print(f"   ❌ China DISFAVORED compared to Ukraine in Post-Brexit model")
        else:
            if row['post_brexit_sp'] > 0:
                print(f"   ✅ Ukraine FAVORED over China in Post-Brexit model") 
            else:
                print(f"   ❌ Ukraine DISFAVORED compared to China in Post-Brexit model")
        
        # Show grant rates
        china_pre = grant_rates.get(('China', topic, 'pre_brexit'), {})
        china_post = grant_rates.get(('China', topic, 'post_brexit'), {})
        ukraine_pre = grant_rates.get(('Ukraine', topic, 'pre_brexit'), {})
        ukraine_post = grant_rates.get(('Ukraine', topic, 'post_brexit'), {})
        
        print(f"   📈 Grant Rates:")
        if china_pre:
            print(f"      China Pre-Brexit:  {china_pre['grant_rate']:.1%} ({china_pre['grants']}/{china_pre['total']})")
        if china_post:
            print(f"      China Post-Brexit: {china_post['grant_rate']:.1%} ({china_post['grants']}/{china_post['total']})")
        if ukraine_pre:
            print(f"      Ukraine Pre-Brexit:  {ukraine_pre['grant_rate']:.1%} ({ukraine_pre['grants']}/{ukraine_pre['total']})")
        if ukraine_post:
            print(f"      Ukraine Post-Brexit: {ukraine_post['grant_rate']:.1%} ({ukraine_post['grants']}/{ukraine_post['total']})")
        
        # Significance pattern
        if row['both_significant']:
            pattern = "🔄 PERSISTENT BIAS"
        elif row['gained_significance']:
            pattern = "🆕 NEWLY EMERGED BIAS"
        elif row['lost_significance']:
            pattern = "📉 DISAPPEARED BIAS"
        else:
            pattern = "❓ UNKNOWN PATTERN"
        
        print(f"   {pattern}")

def main():
    """Main analysis function"""
    
    print("🇨🇳🇺🇦 CHINA vs UKRAINE GRANT RATE ANALYSIS")
    print("="*80)
    
    # Load data
    records = load_raw_data()
    significant_df = load_china_ukraine_significant_topics()
    
    if len(significant_df) == 0:
        print("❌ No FDR-significant China vs Ukraine comparisons found!")
        return
    
    # Get list of significant topics
    significant_topics = significant_df['topic'].unique().tolist()
    print(f"📋 Analyzing {len(significant_topics)} FDR-significant topics:")
    for i, topic in enumerate(significant_topics, 1):
        print(f"   {i}. {topic}")
    
    # Calculate grant rates
    grant_rates = calculate_grant_rates(records, significant_topics)
    
    # Create visualization
    plot_df = create_grant_rate_visualization(grant_rates, significant_df)
    
    # Print detailed analysis
    print_detailed_analysis(significant_df, grant_rates)
    
    # Save data
    output_dir = Path("../../outputs/country_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_df.to_csv(output_dir / "china_ukraine_grant_rates_data.csv", index=False)
    significant_df.to_csv(output_dir / "china_ukraine_significant_comparisons.csv", index=False)
    
    print(f"\n✅ Analysis complete! Results saved to {output_dir}")
    
    return significant_df, grant_rates, plot_df

if __name__ == "__main__":
    results = main() 