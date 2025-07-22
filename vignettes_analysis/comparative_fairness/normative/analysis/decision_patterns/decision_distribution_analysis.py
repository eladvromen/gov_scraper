#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

def load_completion_data():
    """Load the raw completion data to analyze actual decisions"""
    
    # Load the enhanced analysis data
    field_df = pd.read_csv('../outputs/grant_rate_analysis/grant_rate_analysis_by_vignette_fields_enhanced.csv')
    topic_df = pd.read_csv('../outputs/grant_rate_analysis/topic_tendencies_analysis_enhanced.csv')
    
    return field_df, topic_df

def get_divergent_topics():
    """Return the three most divergent topics for detailed analysis"""
    return {
        'disclosure': {
            'name': 'Disclosure',
            'subtopics': [
                'Disclosure: Political persecution & sexual violence',
                'Disclosure: Religious persecution & mental health', 
                'Disclosure: Domestic violence & criminal threats',
                'Disclosure: Ethnic violence & family separation',
                'Disclosure: Persecution for sexual orientation & mental health crisis'
            ],
            'divergence_type': 'Field-Dominant (Granular Disagreement)',
            'field_div': 1.164,
            'topic_div': 0.038
        },
        'asylum_circumstances': {
            'name': 'Asylum seeker circumstances',
            'subtopics': ['Asylum seeker circumstances'],
            'divergence_type': 'Topic-Dominant (Weighting Differences)',
            'field_div': 0.109,
            'topic_div': 0.626
        },
        'safe_country': {
            'name': '3rd safe country - Country safety definition',
            'subtopics': ['3rd safe country - Country safety definition'],
            'divergence_type': 'Combined High Divergence',
            'field_div': 0.908,
            'topic_div': 0.354
        }
    }

def analyze_topic_decision_patterns(field_df, topic_info):
    """Analyze decision patterns for a specific topic"""
    
    topic_name = topic_info['name']
    subtopics = topic_info['subtopics']
    
    # Get all fields for this topic
    if len(subtopics) == 1 and subtopics[0] == topic_name:
        # Regular topic
        topic_fields = field_df[field_df['topic'] == topic_name].copy()
    else:
        # Aggregated topic (like Disclosure)
        topic_fields = field_df[field_df['topic'].isin(subtopics)].copy()
    
    if len(topic_fields) == 0:
        return None
    
    # Create decision pattern analysis
    pattern_data = []
    
    for _, row in topic_fields.iterrows():
        field_name = row['field_value']
        pre_rate = row['pre_brexit_raw_rate']
        post_rate = row['post_brexit_raw_rate']
        pre_normalized = row['pre_brexit_normalized']
        post_normalized = row['post_brexit_normalized']
        difference = row['cross_model_difference']
        significant = row['statistical_significance']
        p_value = row['p_value']
        
        # Categorize the disagreement pattern
        if abs(difference) < 0.1:
            pattern = "Agreement"
        elif difference > 0.3:
            pattern = "Strong Post-Brexit Favor"
        elif difference > 0.1:
            pattern = "Moderate Post-Brexit Favor"
        elif difference < -0.3:
            pattern = "Strong Pre-Brexit Favor"
        elif difference < -0.1:
            pattern = "Moderate Pre-Brexit Favor"
        else:
            pattern = "Slight Disagreement"
        
        pattern_data.append({
            'field': field_name,
            'subtopic': row['topic'],
            'pre_brexit_rate': pre_rate,
            'post_brexit_rate': post_rate,
            'pre_normalized': pre_normalized,
            'post_normalized': post_normalized,
            'difference': difference,
            'pattern': pattern,
            'significant': significant,
            'p_value': p_value
        })
    
    return pd.DataFrame(pattern_data)

def get_field_mappings():
    """Get meaningful abbreviations for field labels by topic"""
    return {
        # 3rd safe country mappings (quotes removed by pandas when reading CSV)
        'severe violence and no recognized asylum procedures, poor healthcare and education systems': 'Severe violence, no systems',
        'ongoing low-level violence and functioning but discriminatory and unequal asylum, healthcare and education systems': 'Low-level violence, unequal systems',
        'generally stable security situation and functioning and accessible asylum, healthcare and education systems': 'Stable security, accessible systems',
        'strong rule of law and high national security and functioning and accessible asylum, healthcare and education systems': 'Rule of law, full system access',
        
        # Asylum seeker circumstances mappings (note: lowercase first letter from CSV data)
        'chronic depression worsened by social isolation': 'Chronic depression',
        'stress and anxiety due to unemployment and lack of support': 'Stress from unemployment',
        'emotional dependency on a partner currently residing in the UK': 'Emotional dependency (partner in UK)',
        'long-term health condition with poor treatment access in home country': 'Long-term condition, poor care access',
        
        # Disclosure mappings (exact case from CSV data)
        'Disclosure: Political persecution & sexual violence': 'Political persecn. & sexual violence',
        'Disclosure: Religious persecution & mental health': 'Religious persecn. & mental health',
        'Disclosure: Domestic violence & criminal threats': 'Domestic violence & threats',
        'Disclosure: Ethnic violence & family separation': 'Ethnic violence & separation',
        'Disclosure: Persecution for sexual orientation & mental health crisis': 'LGBT+ persecn. & mental health'
    }

def create_multiline_labels(labels, max_chars_per_line=25):
    """Create multi-line labels for better readability"""
    
    multiline_labels = []
    for label in labels:
        if len(label) <= max_chars_per_line:
            multiline_labels.append(label)
        else:
            # Try to break at logical points
            words = label.split()
            line1 = ""
            line2 = ""
            
            # Build first line
            for word in words:
                if len(line1 + " " + word) <= max_chars_per_line:
                    line1 += (" " + word) if line1 else word
                else:
                    # Rest goes to second line
                    remaining_words = words[words.index(word):]
                    line2 = " ".join(remaining_words)
                    break
            
            # If line2 is too long, truncate it
            if len(line2) > max_chars_per_line:
                line2 = line2[:max_chars_per_line-3] + "..."
            
            multiline_labels.append(f"{line1}\n{line2}")
    
    return multiline_labels

def create_meaningful_short_labels(fields):
    """Create meaningful short labels using topic-specific mappings"""
    
    field_mappings = get_field_mappings()
    short_labels = []
    
    for field in fields:
        if field in field_mappings:
            mapped = field_mappings[field]
            # For right-side heatmaps, further truncate if still too long to prevent overlaps
            if len(mapped) > 18:
                short_labels.append(mapped[:15] + '...')
            else:
                short_labels.append(mapped)
        else:
            # Fallback to truncation for unmapped fields
            short_labels.append(field[:12] + '...' if len(field) > 12 else field)
    
    return short_labels

def create_meaningful_full_labels(fields):
    """Create meaningful labels for main heatmap - use mappings if available, otherwise multi-line"""
    
    field_mappings = get_field_mappings()
    full_labels = []
    
    for field in fields:
        if field in field_mappings:
            # Use meaningful mapping
            mapped_label = field_mappings[field]
            # If mapped label is still long, make it multi-line
            full_labels.append(create_multiline_labels([mapped_label], max_chars_per_line=25)[0])
        else:
            # Fallback to multi-line wrapping for unmapped fields
            full_labels.append(create_multiline_labels([field], max_chars_per_line=20)[0])
    
    return full_labels

def create_decision_distribution_heatmap(pattern_df, topic_info, save_path):
    """Create detailed heatmap showing decision distributions"""
    
    topic_name = topic_info['name']
    divergence_type = topic_info['divergence_type']
    
    # Prepare data for heatmap
    fields = pattern_df['field'].tolist()
    
    # Create meaningful labels for main heatmap (with multi-line if needed)
    field_labels_full = create_meaningful_full_labels(fields)
    
    # Create meaningful short labels for side heatmaps
    field_labels_short = create_meaningful_short_labels(fields)
    
    # Create matrix data
    heatmap_data = []
    for _, row in pattern_df.iterrows():
        heatmap_data.append([row['pre_brexit_rate'], row['post_brexit_rate']])
    
    heatmap_array = np.array(heatmap_data)
    
    # Create figure with original layout (left: main heatmap, right: diff + sig stacked)
    fig = plt.figure(figsize=(20, 12))
    
    # Main heatmap spanning left side (like original but without pie chart and text)
    ax1 = plt.subplot(2, 3, (1, 4))  # Spans positions 1 and 4 (left side, both rows)
    
    # Main heatmap (Pre-Brexit and Post-Brexit rates)
    im = ax1.imshow(heatmap_array.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels with full 2-line labels
    ax1.set_xticks(range(len(fields)))
    ax1.set_xticklabels(field_labels_full, rotation=45, ha='right', fontsize=9)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Pre-Brexit Model', 'Post-Brexit Model'], fontsize=12, fontweight='bold')
    ax1.set_title(f'Grant Rate Distributions: {topic_name}\n{divergence_type}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('Grant Rate', fontsize=12, fontweight='bold')
    
    # Add value annotations
    for i in range(len(fields)):
        for j in range(2):
            value = heatmap_array[i, j]
            color = 'white' if value < 0.5 else 'black'
            ax1.text(i, j, f'{value:.2f}', ha='center', va='center', 
                    color=color, fontweight='bold', fontsize=8)
    
    # Difference heatmap (top right)
    ax2 = plt.subplot(2, 3, 2)
    differences = pattern_df['difference'].values.reshape(1, -1)
    im2 = ax2.imshow(differences, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax2.set_xticks(range(len(fields)))
    ax2.set_xticklabels(field_labels_short, rotation=45, ha='right', fontsize=8)
    ax2.set_yticks([0])
    ax2.set_yticklabels(['Difference\n(Post - Pre)'], fontsize=9, fontweight='bold')
    ax2.set_title('Cross-Model Differences', fontsize=12, fontweight='bold')
    
    # Add difference value annotations
    for i in range(len(fields)):
        value = differences[0, i]
        color = 'white' if abs(value) > 0.5 else 'black'
        ax2.text(i, 0, f'{value:+.2f}', ha='center', va='center', 
                color=color, fontweight='bold', fontsize=7)
    
    plt.colorbar(im2, ax=ax2, shrink=0.6)
    
    # Significance indicators (bottom right)
    ax3 = plt.subplot(2, 3, 5)
    significance = pattern_df['significant'].astype(int).values.reshape(1, -1)
    im4 = ax3.imshow(significance, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax3.set_xticks(range(len(fields)))
    ax3.set_xticklabels(field_labels_short, rotation=45, ha='right', fontsize=8)
    ax3.set_yticks([0])
    ax3.set_yticklabels(['Statistical\nSignificance'], fontsize=10, fontweight='bold')
    ax3.set_title('Significance Indicators', fontsize=12, fontweight='bold')
    
    # Add significance annotations
    for i in range(len(fields)):
        value = significance[0, i]
        text = 'SIG' if value == 1 else 'NS'
        color = 'white' if value == 1 else 'black'
        ax3.text(i, 0, text, ha='center', va='center', 
                color=color, fontweight='bold', fontsize=7)
    
    plt.colorbar(im4, ax=ax3, shrink=0.6)
    
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(hspace=0.4, wspace=0.3, left=0.08, right=0.95, top=0.92, bottom=0.2)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    plt.show()
    plt.close()

def create_cross_topic_comparison():
    """Create a comparison view across all three divergent topics"""
    
    field_df, topic_df = load_completion_data()
    divergent_topics = get_divergent_topics()
    
    print("=== DECISION DISTRIBUTION ANALYSIS ===")
    print("Analyzing the three most divergent topics...")
    print()
    
    # Analyze each topic
    for topic_key, topic_info in divergent_topics.items():
        print(f"Analyzing {topic_info['name']}...")
        
        pattern_df = analyze_topic_decision_patterns(field_df, topic_info)
        if pattern_df is not None:
            save_path = f"../outputs/rq1_visualizations/decision_distribution_{topic_key}.png"
            create_decision_distribution_heatmap(pattern_df, topic_info, save_path)
            
            print(f"✓ Generated: decision_distribution_{topic_key}.png")
            print(f"  - {len(pattern_df)} fields analyzed")
            print(f"  - {pattern_df['significant'].sum()} statistically significant differences")
            print(f"  - Mean difference: {pattern_df['difference'].mean():+.3f}")
            print()
    
    print("=== QUALITATIVE INSIGHTS ===")
    print("These visualizations reveal:")
    print("• Actual grant rate distributions for each field")
    print("• Cross-model difference patterns") 
    print("• Statistical significance indicators")
    print("• Disagreement pattern classifications")
    print()
    print("Use these to understand WHERE and HOW models disagree granularly!")

if __name__ == "__main__":
    create_cross_topic_comparison() 