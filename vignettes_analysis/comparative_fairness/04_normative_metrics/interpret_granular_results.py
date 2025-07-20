#!/usr/bin/env python3

import pandas as pd
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def interpret_granular_results():
    """Extract and interpret the granular topic divergence results"""
    
    # Load data like in the main script
    field_df = pd.read_csv('../outputs/grant_rate_analysis/grant_rate_analysis_by_vignette_fields_enhanced.csv')
    topic_df = pd.read_csv('../outputs/grant_rate_analysis/topic_tendencies_analysis_enhanced.csv')

    # Create 13D topic vector
    exclude_patterns = [
        'Disclosure: Political persecution & sexual violence',
        'Disclosure: Religious persecution & mental health', 
        'Disclosure: Domestic violence & criminal threats',
        'Disclosure: Ethnic violence & family separation',
        'Disclosure: Persecution for sexual orientation & mental health crisis',
        'Contradiction: Dates of persecution',
        'Contradiction: Persecutor identity confusion', 
        'Contradiction: Location of harm',
        'Contradiction: Family involvement in the persecution',
        'Contradiction: Sequence of events'
    ]
    topic_13d = topic_df[~topic_df['topic'].isin(exclude_patterns)].copy()

    print('=== GRANULAR TOPIC DIVERGENCE ANALYSIS RESULTS ===')
    print()
    
    results = []

    # Calculate topic-specific field divergences
    for _, topic_row in topic_13d.iterrows():
        topic_name = topic_row['topic']
        
        # Get field subset for this topic
        if topic_name == 'Disclosure':
            disclosure_patterns = [
                'Disclosure: Political persecution & sexual violence',
                'Disclosure: Religious persecution & mental health', 
                'Disclosure: Domestic violence & criminal threats',
                'Disclosure: Ethnic violence & family separation',
                'Disclosure: Persecution for sexual orientation & mental health crisis'
            ]
            topic_fields = field_df[field_df['topic'].isin(disclosure_patterns)].copy()
        elif topic_name == 'Contradiction':
            contradiction_patterns = [
                'Contradiction: Dates of persecution',
                'Contradiction: Persecutor identity confusion', 
                'Contradiction: Location of harm',
                'Contradiction: Family involvement in the persecution',
                'Contradiction: Sequence of events'
            ]
            topic_fields = field_df[field_df['topic'].isin(contradiction_patterns)].copy()
        else:
            topic_fields = field_df[field_df['topic'] == topic_name].copy()
        
        if len(topic_fields) == 0:
            continue
            
        # Calculate metrics
        topic_level_div = abs(topic_row['topic_tendency_difference'])
        field_count = len(topic_fields)
        sig_rate = topic_fields['statistical_significance'].mean()
        mean_diff = topic_fields['cross_model_difference'].mean()
        
        # Calculate field-level cosine distance if sufficient dimensions
        if len(topic_fields) >= 2:
            pre_sub = topic_fields['pre_brexit_normalized'].values
            post_sub = topic_fields['post_brexit_normalized'].values
            cos_sim = cosine_similarity(pre_sub.reshape(1, -1), post_sub.reshape(1, -1))[0][0]
            field_level_div = 1 - cos_sim
            
            # Determine pattern
            if field_level_div > topic_level_div + 0.05:
                pattern = "Field > Topic (Granular Disagreement)"
            elif topic_level_div > field_level_div + 0.05:
                pattern = "Topic > Field (Weighting Differences)"
            else:
                pattern = "Balanced (Consistent Disagreement)"
            
            result = {
                'topic': topic_name,
                'topic_div': topic_level_div,
                'field_div': field_level_div,
                'field_count': field_count,
                'sig_rate': sig_rate,
                'mean_diff': mean_diff,
                'pattern': pattern,
                'ratio': topic_level_div / field_level_div if field_level_div > 0 else float('inf')
            }
            
            print(f'{topic_name}:')
            print(f'  Topic-Level Divergence: {topic_level_div:.3f}')
            print(f'  Field-Level Divergence: {field_level_div:.3f}')
            print(f'  Field Count: {field_count}')
            print(f'  Significance Rate: {sig_rate:.1%}')
            print(f'  Mean Difference: {mean_diff:+.3f}')
            print(f'  Pattern: {pattern}')
            print()
            
        else:
            result = {
                'topic': topic_name,
                'topic_div': topic_level_div,
                'field_div': 'N/A',
                'field_count': field_count,
                'sig_rate': sig_rate,
                'mean_diff': mean_diff,
                'pattern': 'Single Field Topic',
                'ratio': 'N/A'
            }
            
            print(f'{topic_name}:')
            print(f'  Topic-Level Divergence: {topic_level_div:.3f}')
            print(f'  Field-Level Divergence: N/A (only {field_count} field)')
            print(f'  Significance Rate: {sig_rate:.1%}')
            print()
        
        results.append(result)
    
    # Summary insights
    print('\n=== KEY INTERPRETATIONS ===')
    
    # Find highest divergences
    valid_results = [r for r in results if isinstance(r['field_div'], float)]
    
    if valid_results:
        highest_field = max(valid_results, key=lambda x: x['field_div'])
        highest_topic = max(valid_results, key=lambda x: x['topic_div'])
        highest_combined = max(valid_results, key=lambda x: x['topic_div'] + x['field_div'])
        
        print(f"\n1. HIGHEST FIELD-LEVEL DIVERGENCE:")
        print(f"   {highest_field['topic']}: {highest_field['field_div']:.3f}")
        print(f"   → Models disagree most on individual field decisions within this topic")
        
        print(f"\n2. HIGHEST TOPIC-LEVEL DIVERGENCE:")
        print(f"   {highest_topic['topic']}: {highest_topic['topic_div']:.3f}")
        print(f"   → Models have most different overall approaches to this topic")
        
        print(f"\n3. HIGHEST COMBINED DIVERGENCE:")
        print(f"   {highest_combined['topic']}: {highest_combined['topic_div'] + highest_combined['field_div']:.3f}")
        print(f"   → This topic drives the most overall interpretive drift")
        
        # Pattern analysis
        field_dominant = [r for r in valid_results if "Field > Topic" in r['pattern']]
        topic_dominant = [r for r in valid_results if "Topic > Field" in r['pattern']]
        balanced = [r for r in valid_results if "Balanced" in r['pattern']]
        
        print(f"\n4. DIVERGENCE PATTERNS:")
        print(f"   • Field-Dominant (Granular Disagreement): {len(field_dominant)} topics")
        if field_dominant:
            print(f"     - {', '.join([r['topic'][:25] for r in field_dominant])}")
        print(f"   • Topic-Dominant (Weighting Differences): {len(topic_dominant)} topics")
        if topic_dominant:
            print(f"     - {', '.join([r['topic'][:25] for r in topic_dominant])}")
        print(f"   • Balanced (Consistent Disagreement): {len(balanced)} topics")
        
        print(f"\n5. WHAT THIS MEANS:")
        if len(field_dominant) > len(topic_dominant):
            print("   → Models primarily disagree on individual field decisions")
            print("   → This suggests different micro-level decision patterns")
        elif len(topic_dominant) > len(field_dominant):
            print("   → Models primarily disagree on how to weight/combine field evidence")
            print("   → This suggests different macro-level reasoning approaches")
        else:
            print("   → Mixed pattern: both granular and systematic differences exist")

if __name__ == "__main__":
    interpret_granular_results() 