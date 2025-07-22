#!/usr/bin/env python3
"""
FINAL CORRECTED: Age group SP and grant rates with proper model assignments
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

def load_grant_rates():
    """Load grant rates with correct model assignments"""
    print("Loading raw data to calculate grant rates...")
    
    with open('../../../data/processed/tagged_records.json', 'r') as f:
        records = json.load(f)
    
    # Target topics
    target_keywords = ['asylum seeker circumstances', '3rd safe country', 'activist persecution ground']
    
    # Filter records for target topics
    filtered_records = []
    for record in records:
        topic = record.get('topic', '').lower()
        if any(keyword.lower() in topic for keyword in target_keywords):
            filtered_records.append(record)
    
    print(f"Found {len(filtered_records)} records for target topics")
    
    # Calculate grant rates by age, topic, and model
    grant_rates = {}
    
    for record in filtered_records:
        model = record.get('model', '')
        topic = record.get('topic', '')
        protected_attrs = record.get('protected_attributes', {})
        age = protected_attrs.get('age') if protected_attrs else None
        decision = record.get('decision', '').upper()
        
        if model and topic and age is not None:
            key = (model, topic, age)
            if key not in grant_rates:
                grant_rates[key] = {'grants': 0, 'total': 0}
            
            grant_rates[key]['total'] += 1
            if decision == 'GRANT':
                grant_rates[key]['grants'] += 1
    
    print(f"Calculated grant rates for {len(grant_rates)} model-topic-age combinations")
    
    # Convert to grant rate percentages
    final_grant_rates = {}
    for (model, topic, age), data in grant_rates.items():
        if data['total'] > 0:
            final_grant_rates[(model, topic, age)] = data['grants'] / data['total']
    
    return final_grant_rates

def main():
    print("🎯 FINAL CORRECTED AGE GROUP DATA EXTRACTION")
    print("=" * 60)
    
    # Load existing unified fairness data
    df = pd.read_csv('../../outputs/unified/unified_fairness_dataframe_topic_granular.csv')
    age_df = df[df['protected_attribute'] == 'age'].copy()
    
    # Filter for target topics
    target_keywords = ['asylum seeker circumstances', '3rd safe country', 'activist persecution ground']
    target_data = []
    
    for keyword in target_keywords:
        matches = age_df[age_df['topic'].str.lower().str.contains(keyword.lower(), na=False)]
        if len(matches) > 0:
            target_data.append(matches)
    
    final_df = pd.concat(target_data, ignore_index=True)
    
    # Filter for FMR significant cases
    significant_df = final_df[
        (final_df['pre_brexit_sp_significance'] == True) | 
        (final_df['post_brexit_sp_significance'] == True)
    ].copy()
    
    print(f"📊 Found {len(significant_df)} FMR significant age comparisons")
    
    # Load grant rates
    grant_rates = load_grant_rates()
    
    # Extract the results
    print("\n" + "=" * 80)
    print("FINAL RESULTS: AGE GROUP SP VALUES & GRANT RATES")
    print("=" * 80)
    
    for topic in significant_df['topic'].unique():
        topic_data = significant_df[significant_df['topic'] == topic]
        
        print(f"\n📊 Topic: {topic}")
        print("-" * 60)
        
        # Show SP values
        print("\nStatistical Parity Values (relative to age 40):")
        print("Age Group     | Pre-Brexit SP | Post-Brexit SP | Significant")
        print("-" * 65)
        
        for _, row in topic_data.iterrows():
            age_group = row['group_comparison']
            pre_sp = row['pre_brexit_model_statistical_parity']
            post_sp = row['post_brexit_model_statistical_parity']
            pre_sig = "✓" if row['pre_brexit_sp_significance'] else "✗"
            post_sig = "✓" if row['post_brexit_sp_significance'] else "✗"
            
            print(f"{age_group:<13} | {pre_sp:>12.4f} | {post_sp:>13.4f} | {pre_sig}/{post_sig}")
        
        # Show grant rates for this topic
        print("\nGrant Rates by Age:")
        print("Age | Pre-Brexit Rate | Post-Brexit Rate | Difference")
        print("-" * 52)
        
        # Extract ages from comparisons
        ages = set()
        for _, row in topic_data.iterrows():
            comp = row['group_comparison']
            if '_vs_' in comp:
                age1, age2 = comp.split('_vs_')
                try:
                    ages.add(int(age1))
                    ages.add(int(age2))
                except:
                    pass
        
        for age in sorted(ages):
            pre_rate = grant_rates.get(('pre_brexit', topic, age), 0)
            post_rate = grant_rates.get(('post_brexit', topic, age), 0)
            difference = post_rate - pre_rate
            
            print(f"{age:>3} | {pre_rate:>14.4f} | {post_rate:>15.4f} | {difference:>+9.4f}")
    
    # Create summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    
    summary_data = []
    
    for topic in significant_df['topic'].unique():
        topic_data = significant_df[significant_df['topic'] == topic]
        
        # Extract ages
        ages = set()
        for _, row in topic_data.iterrows():
            comp = row['group_comparison']
            if '_vs_' in comp:
                age1, age2 = comp.split('_vs_')
                try:
                    ages.add(int(age1))
                    ages.add(int(age2))
                except:
                    pass
        
        for age in sorted(ages):
            pre_rate = grant_rates.get(('pre_brexit', topic, age), 0)
            post_rate = grant_rates.get(('post_brexit', topic, age), 0)
            
            # Find SP value for this age vs 40
            sp_pre = None
            sp_post = None
            for _, row in topic_data.iterrows():
                comp = row['group_comparison']
                if f"{age}_vs_40" == comp:
                    sp_pre = row['pre_brexit_model_statistical_parity']
                    sp_post = row['post_brexit_model_statistical_parity']
                elif f"40_vs_{age}" == comp:
                    sp_pre = -row['pre_brexit_model_statistical_parity']
                    sp_post = -row['post_brexit_model_statistical_parity']
            
            if sp_pre is not None:
                summary_data.append({
                    'Topic': topic,
                    'Age': age,
                    'Pre_Brexit_Grant_Rate': pre_rate,
                    'Post_Brexit_Grant_Rate': post_rate,
                    'Grant_Rate_Change': post_rate - pre_rate,
                    'Pre_Brexit_SP': sp_pre,
                    'Post_Brexit_SP': sp_post,
                    'SP_Change': sp_post - sp_pre
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    print("Topic | Age | Pre-Grant | Post-Grant | Grant-Δ | Pre-SP | Post-SP | SP-Δ")
    print("-" * 75)
    
    for _, row in summary_df.iterrows():
        topic_short = row['Topic'][:20] + "..." if len(row['Topic']) > 20 else row['Topic']
        print(f"{topic_short:<20} | {row['Age']:>3} | {row['Pre_Brexit_Grant_Rate']:>8.3f} | {row['Post_Brexit_Grant_Rate']:>9.3f} | {row['Grant_Rate_Change']:>+6.3f} | {row['Pre_Brexit_SP']:>+5.3f} | {row['Post_Brexit_SP']:>+6.3f} | {row['SP_Change']:>+5.3f}")
    
    # Create line plot visualization
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATION: Grant Rate by Age Group Across Topics")
    print("=" * 80)
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Define age groups and topics
    age_groups = [12, 25, 40, 70]
    topics = significant_df['topic'].unique()
    
    # Color palette for topics
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    for i, topic in enumerate(topics):
        pre_rates = []
        post_rates = []
        
        for age in age_groups:
            pre_rate = grant_rates.get(('pre_brexit', topic, age), 0) * 100  # Convert to percentage
            post_rate = grant_rates.get(('post_brexit', topic, age), 0) * 100
            pre_rates.append(pre_rate)
            post_rates.append(post_rate)
        
        # Plot pre-Brexit line (dashed)
        plt.plot(age_groups, pre_rates, 
                color=colors[i], linestyle='--', linewidth=2.5, 
                marker='o', markersize=8, alpha=0.8,
                label=f'{topic} (Pre-Brexit)')
        
        # Plot post-Brexit line (solid)
        plt.plot(age_groups, post_rates, 
                color=colors[i], linestyle='-', linewidth=2.5,
                marker='s', markersize=8, alpha=0.9,
                label=f'{topic} (Post-Brexit)')
    
    # Customize the plot
    plt.xlabel('Age Group', fontsize=14, fontweight='bold')
    plt.ylabel('Grant Rate (%)', fontsize=14, fontweight='bold')
    plt.title('Grant Rate by Age Group Across Topics\n(Pre-Brexit vs Post-Brexit Models)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis ticks
    plt.xticks(age_groups, [f'Age {age}' for age in age_groups])
    
    # Set y-axis to percentage
    plt.ylim(0, 100)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Tight layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot
    plot_filename = '/data/shil6369/gov_scraper/vignettes_analysis/comparative_fairness/fairness/analysis/age/age_grant_rates_by_topic_comparison.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Visualization saved as: {plot_filename}")
    
    # Save results
    output_file = 'final_age_sp_grant_rates.csv'
    summary_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")
    
    return summary_df

if __name__ == "__main__":
    results = main() 