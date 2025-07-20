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
        
        pattern_data.append({
            'field': field_name,
            'subtopic': row['topic'],
            'pre_brexit_rate': pre_rate,
            'post_brexit_rate': post_rate,
            'pre_normalized': pre_normalized,
            'post_normalized': post_normalized,
            'difference': difference,
            'significant': significant,
            'p_value': p_value
        })
    
    return pd.DataFrame(pattern_data)

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

def create_combined_decision_distribution():
    """Create combined heatmap with all three topics side by side"""
    
    print("Creating combined decision distribution analysis...")
    
    field_df, topic_df = load_completion_data()
    divergent_topics = get_divergent_topics()
    
    # Analyze all topics
    topic_data = {}
    for topic_key, topic_info in divergent_topics.items():
        pattern_df = analyze_topic_decision_patterns(field_df, topic_info)
        if pattern_df is not None:
            topic_data[topic_key] = {
                'pattern_df': pattern_df,
                'topic_info': topic_info
            }
    
    # Create the combined figure
    # 4 rows (Pre-Brexit, Post-Brexit, Difference, Significance) x 3 columns (topics)
    fig, axes = plt.subplots(4, 3, figsize=(24, 16))
    
    topic_keys = ['disclosure', 'asylum_circumstances', 'safe_country']
    topic_titles = [
        'Disclosure\n(Field-Dominant Divergence)',
        'Asylum Seeker Circumstances\n(Topic-Dominant Divergence)', 
        'Safe Country Definition\n(Combined High Divergence)'
    ]
    
    # Process each topic
    for col_idx, (topic_key, title) in enumerate(zip(topic_keys, topic_titles)):
        if topic_key not in topic_data:
            continue
            
        pattern_df = topic_data[topic_key]['pattern_df']
        topic_info = topic_data[topic_key]['topic_info']
        
        fields = pattern_df['field'].tolist()
        
        # Create multi-line field labels
        field_labels = create_multiline_labels(fields, max_chars_per_line=20)
        
        # Prepare data arrays
        pre_rates = pattern_df['pre_brexit_rate'].values.reshape(1, -1)
        post_rates = pattern_df['post_brexit_rate'].values.reshape(1, -1)
        differences = pattern_df['difference'].values.reshape(1, -1)
        significance = pattern_df['significant'].astype(int).values.reshape(1, -1)
        
        # Row 0: Pre-Brexit rates
        ax = axes[0, col_idx]
        im1 = ax.imshow(pre_rates, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(fields)))
        ax.set_xticklabels(field_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticks([0])
        ax.set_yticklabels(['Pre-Brexit\nModel'], fontsize=10, fontweight='bold')
        
        # Add value annotations
        for i in range(len(fields)):
            value = pre_rates[0, i]
            color = 'white' if value < 0.5 else 'black'
            ax.text(i, 0, f'{value:.2f}', ha='center', va='center', 
                   color=color, fontweight='bold', fontsize=8)
        
        # Add title only to top row
        if col_idx == 1:  # Middle column
            ax.set_title('Pre-Brexit Grant Rates', fontsize=14, fontweight='bold', pad=20)
        
        # Row 1: Post-Brexit rates  
        ax = axes[1, col_idx]
        im2 = ax.imshow(post_rates, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(fields)))
        ax.set_xticklabels(field_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticks([0])
        ax.set_yticklabels(['Post-Brexit\nModel'], fontsize=10, fontweight='bold')
        
        # Add value annotations
        for i in range(len(fields)):
            value = post_rates[0, i]
            color = 'white' if value < 0.5 else 'black'
            ax.text(i, 0, f'{value:.2f}', ha='center', va='center', 
                   color=color, fontweight='bold', fontsize=8)
        
        if col_idx == 1:  # Middle column
            ax.set_title('Post-Brexit Grant Rates', fontsize=14, fontweight='bold', pad=20)
        
        # Row 2: Differences
        ax = axes[2, col_idx]
        im3 = ax.imshow(differences, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(fields)))
        ax.set_xticklabels(field_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticks([0])
        ax.set_yticklabels(['Difference\n(Post - Pre)'], fontsize=10, fontweight='bold')
        
        # Add difference value annotations
        for i in range(len(fields)):
            value = differences[0, i]
            color = 'white' if abs(value) > 0.5 else 'black'
            ax.text(i, 0, f'{value:+.2f}', ha='center', va='center', 
                   color=color, fontweight='bold', fontsize=8)
        
        if col_idx == 1:  # Middle column
            ax.set_title('Cross-Model Differences', fontsize=14, fontweight='bold', pad=20)
        
        # Row 3: Statistical significance
        ax = axes[3, col_idx]
        im4 = ax.imshow(significance, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(fields)))
        ax.set_xticklabels(field_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticks([0])
        ax.set_yticklabels(['Statistical\nSignificance'], fontsize=10, fontweight='bold')
        
        # Add significance annotations
        for i in range(len(fields)):
            value = significance[0, i]
            text = 'SIG' if value == 1 else 'NS'
            color = 'white' if value == 1 else 'black'
            ax.text(i, 0, text, ha='center', va='center', 
                   color=color, fontweight='bold', fontsize=8)
        
        if col_idx == 1:  # Middle column
            ax.set_title('Statistical Significance', fontsize=14, fontweight='bold', pad=20)
        
        # Add topic title to leftmost column
        if col_idx == 0:
            # Add topic name as y-axis label spanning all rows for this column
            fig.text(0.02, 0.5, title, rotation=90, va='center', ha='center',
                    fontsize=16, fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        elif col_idx == 1:
            fig.text(0.42, 0.5, title, rotation=90, va='center', ha='center',
                    fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
        elif col_idx == 2:
            fig.text(0.82, 0.5, title, rotation=90, va='center', ha='center',
                    fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # Add colorbars
    # Colorbar for grant rates (rows 0 and 1)
    cbar1 = fig.colorbar(im1, ax=axes[:2, :].ravel().tolist(), shrink=0.6, aspect=30, pad=0.01)
    cbar1.set_label('Grant Rate', fontsize=12, fontweight='bold')
    
    # Colorbar for differences (row 2)
    cbar2 = fig.colorbar(im3, ax=axes[2, :].ravel().tolist(), shrink=0.6, aspect=15, pad=0.01)
    cbar2.set_label('Difference (Post - Pre)', fontsize=12, fontweight='bold')
    
    # Colorbar for significance (row 3)
    cbar3 = fig.colorbar(im4, ax=axes[3, :].ravel().tolist(), shrink=0.6, aspect=15, pad=0.01)
    cbar3.set_label('Statistically Significant', fontsize=12, fontweight='bold')
    cbar3.set_ticks([0, 1])
    cbar3.set_ticklabels(['No', 'Yes'])
    
    # Overall title
    fig.suptitle('Decision Distribution Analysis: Three Most Divergent Topics\n' +
                'Heatmaps showing grant rates, differences, and statistical significance across case characteristics',
                fontsize=18, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0.08, 0.02, 0.98, 0.95])
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Save the plot
    save_path = '../outputs/rq1_visualizations/combined_decision_distribution_heatmaps.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.4)
    
    print(f"✓ Generated combined decision distribution plot: {save_path}")
    print(f"✓ Shows heatmaps for: Pre-Brexit rates, Post-Brexit rates, Differences, Statistical significance")
    print(f"✓ Full field labels displayed with multi-line wrapping")
    
    plt.show()
    plt.close()

if __name__ == "__main__":
    create_combined_decision_distribution() 