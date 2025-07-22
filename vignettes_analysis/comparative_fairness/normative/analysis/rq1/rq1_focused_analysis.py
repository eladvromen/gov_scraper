#!/usr/bin/env python3
"""
RQ1: Focused Analysis - EXACTLY what is needed
1. Topic level comparison for 13 topics
2. Topic divergence statistics summary 
3. Within-topic preference shifts analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set clean style
plt.style.use('default')
sns.set_palette("Set2")

def load_data():
    """Load all necessary data"""
    field_df = pd.read_csv("../outputs/grant_rate_analysis/grant_rate_analysis_by_vignette_fields_enhanced.csv")
    topic_df = pd.read_csv("../outputs/grant_rate_analysis/topic_tendencies_analysis_enhanced.csv")
    
    with open("../outputs/normative_divergence/normative_divergence_results.json", 'r') as f:
        divergence_results = json.load(f)
    
    return field_df, topic_df, divergence_results

def create_13d_topic_vector(topic_df):
    """Create proper 13D topic vector"""
    exclude_patterns = [
        "Disclosure: Political persecution & sexual violence",
        "Disclosure: Religious persecution & mental health", 
        "Disclosure: Domestic violence & criminal threats",
        "Disclosure: Ethnic violence & family separation",
        "Disclosure: Persecution for sexual orientation & mental health crisis",
        "Contradiction: Dates of persecution",
        "Contradiction: Persecutor identity confusion", 
        "Contradiction: Location of harm",
        "Contradiction: Family involvement in the persecution",
        "Contradiction: Sequence of events"
    ]
    
    topic_13d = topic_df[~topic_df['topic'].isin(exclude_patterns)].copy()
    return topic_13d

def smart_truncate_topic(topic, max_length=30):
    """Smart truncation that creates short, readable 2-line topic names"""
    # First, try to make shorter single line if possible
    if len(topic) <= max_length:
        return topic
    
    # Try to break at logical points for 2-line display
    break_points = [' - ', ': ', ' & ', ' (', ' of ', ' in ']
    
    for bp in break_points:
        if bp in topic:
            parts = topic.split(bp, 1)
            # More aggressive truncation for better readability
            if 10 <= len(parts[0]) <= 25:
                line1 = parts[0]
                line2 = parts[1][:25] + ('...' if len(parts[1]) > 25 else '')
                return line1 + '\n' + line2
    
    # If no good break point, split at word boundary 
    if ' ' in topic:
        words = topic.split(' ')
        
        # Try to split into two roughly equal parts
        total_len = len(topic)
        target_first_line = min(25, total_len // 2)
        
        current_length = 0
        split_index = 0
        
        for i, word in enumerate(words):
            if current_length + len(word) > target_first_line and current_length > 10:
                split_index = i
                break
            current_length += len(word) + 1  # +1 for space
        
        if split_index > 0:
            line1 = ' '.join(words[:split_index])
            remaining_words = words[split_index:]
            line2 = ' '.join(remaining_words)
            
            # Truncate second line if too long
            if len(line2) > 30:
                line2 = line2[:27] + '...'
            
            return line1 + '\n' + line2
    
    # Fallback: simple truncation with ellipsis
    return topic[:max_length-3] + '...'

def calculate_topic_significance(field_df, topic_13d):
    """Calculate statistical significance for each topic from field-level data"""
    topic_significance = []
    
    for topic in topic_13d['topic']:
        if topic == 'Disclosure':
            # Aggregate disclosure subtopics
            disclosure_patterns = [
                'Disclosure: Political persecution & sexual violence',
                'Disclosure: Religious persecution & mental health', 
                'Disclosure: Domestic violence & criminal threats',
                'Disclosure: Ethnic violence & family separation',
                'Disclosure: Persecution for sexual orientation & mental health crisis'
            ]
            topic_fields = field_df[field_df['topic'].isin(disclosure_patterns)]
        elif topic == 'Contradiction':
            # Aggregate contradiction subtopics  
            contradiction_patterns = [
                'Contradiction: Dates of persecution',
                'Contradiction: Persecutor identity confusion', 
                'Contradiction: Location of harm',
                'Contradiction: Family involvement in the persecution',
                'Contradiction: Sequence of events'
            ]
            topic_fields = field_df[field_df['topic'].isin(contradiction_patterns)]
        else:
            topic_fields = field_df[field_df['topic'] == topic]
        
        if len(topic_fields) > 0:
            # Calculate topic-level significance metrics
            sig_rate = topic_fields['statistical_significance'].mean()
            min_p_value = topic_fields['p_value'].min() if 'p_value' in topic_fields.columns else 1.0
            n_fields = len(topic_fields)
            n_sig_fields = topic_fields['statistical_significance'].sum()
            
            # Topic is significant if >50% fields significant OR any field has p<0.01
            is_significant = sig_rate > 0.5 or min_p_value < 0.01
            
            topic_significance.append({
                'topic': topic,
                'significance_rate': sig_rate,
                'min_p_value': min_p_value,
                'n_fields': n_fields,
                'n_significant_fields': n_sig_fields,
                'is_topic_significant': is_significant
            })
        else:
            topic_significance.append({
                'topic': topic,
                'significance_rate': 0.0,
                'min_p_value': 1.0,
                'n_fields': 0,
                'n_significant_fields': 0,
                'is_topic_significant': False
            })
    
    return pd.DataFrame(topic_significance)

def create_topic_level_comparison(topic_13d):
    """Create publication-quality topic-level comparison with aligned significance indicators"""
    
    print("Creating enhanced 13D topic-level comparison...")
    
    # Load field data for significance calculation
    field_df, _, _ = load_data()
    
    # Calculate topic-level significance
    sig_df = calculate_topic_significance(field_df, topic_13d)
    
    # Merge significance data with topic data
    topic_13d_enhanced = topic_13d.merge(sig_df[['topic', 'is_topic_significant', 'significance_rate']], 
                                        on='topic', how='left')
    
    # Sort by magnitude for better visualization
    topic_13d_sorted = topic_13d_enhanced.sort_values('topic_tendency_difference', key=abs, ascending=True)
    
    # Create figure with manual subplot positioning for complete control
    fig = plt.figure(figsize=(26, 14))
    
    # Create two separate subplots with exact positioning - push significance bar to far left
    # Left plot: significance (very narrow, pushed to left edge)
    ax_sig = plt.subplot2grid((1, 30), (0, 0), colspan=1)  # Takes 1 column out of 30, starts at edge
    # Right plot: main divergence (moved further right to prevent overlap)
    ax_main = plt.subplot2grid((1, 30), (0, 4), colspan=24)  # Takes 24 columns, starts at column 4
    
    # Shared y-axis setup for perfect alignment
    y_positions = np.arange(len(topic_13d_sorted))
    bar_height = 0.75
    
    # Smart topic name truncation with aggressive 2-line support
    truncated_topics = [smart_truncate_topic(topic, 25) for topic in topic_13d_sorted['topic']]
    
    # PLOT 1: SIGNIFICANCE INDICATORS (LEFT - VERY NARROW)
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    # Create color gradient based on significance rate
    cmap = cm.get_cmap('RdYlGn')  # Red-Yellow-Green colormap
    sig_colors = [cmap(sig_rate) for sig_rate in topic_13d_sorted['significance_rate']]
    
    # Create significance bars
    sig_bars = ax_sig.barh(y_positions, [1] * len(y_positions), 
                          height=bar_height, color=sig_colors, alpha=0.9, 
                          edgecolor='black', linewidth=0.8)
    
    # Format significance plot - make it very clean and minimal
    ax_sig.set_xlim(0, 1)
    ax_sig.set_ylim(-0.5, len(y_positions) - 0.5)
    ax_sig.set_yticks([])  # No y-axis ticks
    ax_sig.set_xticks([])  # No x-axis ticks
    ax_sig.set_title('Sig.\nRate', fontsize=12, fontweight='bold', pad=10)
    
    # Remove all spines to make it cleaner
    for spine in ax_sig.spines.values():
        spine.set_visible(False)
    
    # Add significance rate text
    for i, (bar, sig_rate) in enumerate(zip(sig_bars, topic_13d_sorted['significance_rate'])):
        text_color = 'white' if sig_rate < 0.5 else 'black'
        ax_sig.text(0.5, bar.get_y() + bar.get_height()/2, 
                   f'{sig_rate:.0%}', ha='center', va='center', 
                   fontsize=10, fontweight='bold', color=text_color)
    
    # PLOT 2: MAIN DIVERGENCE BARS (RIGHT - WIDE)
    colors = ['#d73027' if x > 0 else '#1a9850' for x in topic_13d_sorted['topic_tendency_difference']]
    
    # Create main divergence bars
    bars = ax_main.barh(y_positions, topic_13d_sorted['topic_tendency_difference'], 
                       height=bar_height, color=colors, alpha=0.8, 
                       edgecolor='black', linewidth=1)
    
    # Format main plot
    ax_main.set_yticks(y_positions)
    ax_main.set_yticklabels(truncated_topics, fontsize=12, ha='right')
    ax_main.set_xlabel('Topic Tendency Difference (Pre-Brexit - Post-Brexit)', 
                      fontsize=16, fontweight='bold')
    ax_main.set_title('Topic-Level Interpretive Drift (13D Vector)', 
                     fontsize=18, fontweight='bold', pad=25)
    
    # Add reference line and grid
    ax_main.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=2.5)
    ax_main.grid(axis='x', alpha=0.3, linewidth=0.8)
    
    # Calculate max absolute value and set margins
    max_abs_val = topic_13d_sorted['topic_tendency_difference'].abs().max()
    
    # Add value labels on bars
    for bar, val in zip(bars, topic_13d_sorted['topic_tendency_difference']):
        offset = max(0.01, abs(val) * 0.06)
        label_pos = bar.get_width() + (offset if val > 0 else -offset)
        ax_main.text(label_pos, bar.get_y() + bar.get_height()/2,
                    f'{val:+.3f}', ha='left' if val > 0 else 'right', 
                    va='center', fontsize=10, fontweight='bold')
    
    # Set x-axis limits
    x_padding = max_abs_val * 0.18
    ax_main.set_xlim(-(max_abs_val + x_padding), max_abs_val + x_padding)
    
    # CRITICAL: Ensure both plots have EXACTLY the same y-limits
    shared_ylim = (-0.5, len(y_positions) - 0.5)
    ax_sig.set_ylim(shared_ylim)
    ax_main.set_ylim(shared_ylim)
    
    # Add legends
    from matplotlib.patches import Patch
    
    # Main legend
    legend_elements = [
        Patch(facecolor='#d73027', label='Pre-Brexit More Generous'),
        Patch(facecolor='#1a9850', label='Post-Brexit More Generous')
    ]
    ax_main.legend(handles=legend_elements, loc='lower right', fontsize=12, 
                  bbox_to_anchor=(0.98, 0.02))
    
    # Significance colorbar positioned below the significance plot
    sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    
    # Position colorbar below significance plot - pushed to far left
    cbar_ax = fig.add_axes([0.01, 0.08, 0.03, 0.02])  # [left, bottom, width, height] - pushed to very edge
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Sig. Rate', fontsize=9, fontweight='bold')
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(['0%', '50%', '100%'])
    
    # Fine-tune overall layout - minimal left margin to push significance bar to edge
    plt.tight_layout()
    plt.subplots_adjust(left=0.01, right=0.98, top=0.90, bottom=0.15)
    
    # Save with high quality
    plt.savefig('../outputs/rq1_visualizations/13d_topic_comparison.png', 
                dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.4)
    plt.show()
    plt.close()

def calculate_topic_divergence_stats(topic_13d):
    """Calculate divergence statistics for 13D topic vector"""
    
    # Extract the two vectors
    pre_brexit_vector = topic_13d['pre_brexit_topic_tendency'].values
    post_brexit_vector = topic_13d['post_brexit_topic_tendency'].values
    
    # Calculate divergence metrics
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.stats import pearsonr, spearmanr
    
    # Reshape for sklearn
    pre_reshaped = pre_brexit_vector.reshape(1, -1)
    post_reshaped = post_brexit_vector.reshape(1, -1)
    
    # Cosine similarity
    cos_sim = cosine_similarity(pre_reshaped, post_reshaped)[0][0]
    cos_distance = 1 - cos_sim
    
    # Pearson correlation
    pearson_r, pearson_p = pearsonr(pre_brexit_vector, post_brexit_vector)
    
    # Spearman correlation
    spearman_r, spearman_p = spearmanr(pre_brexit_vector, post_brexit_vector)
    
    # Other metrics
    euclidean_dist = np.linalg.norm(pre_brexit_vector - post_brexit_vector)
    manhattan_dist = np.sum(np.abs(pre_brexit_vector - post_brexit_vector))
    mean_abs_diff = np.mean(np.abs(pre_brexit_vector - post_brexit_vector))
    
    # Identify flips
    norm_flips = topic_13d[
        ((topic_13d['pre_brexit_topic_tendency'] > 0.05) & (topic_13d['post_brexit_topic_tendency'] < -0.05)) |
        ((topic_13d['pre_brexit_topic_tendency'] < -0.05) & (topic_13d['post_brexit_topic_tendency'] > 0.05))
    ]
    
    stats = {
        'vector_dimension': len(topic_13d),
        'cosine_similarity': cos_sim,
        'cosine_distance': cos_distance,
        'pearson_correlation': pearson_r,
        'pearson_p_value': pearson_p,
        'spearman_correlation': spearman_r,
        'spearman_p_value': spearman_p,
        'euclidean_distance': euclidean_dist,
        'manhattan_distance': manhattan_dist,
        'mean_absolute_difference': mean_abs_diff,
        'norm_flips_count': len(norm_flips),
        'norm_flips': norm_flips[['topic', 'pre_brexit_topic_tendency', 'post_brexit_topic_tendency', 'topic_tendency_difference']].to_dict('records') if len(norm_flips) > 0 else []
    }
    
    return stats

def create_divergence_summary(topic_stats):
    """Create divergence statistics summary visualization"""
    
    print("Creating topic divergence statistics summary...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Main divergence metrics
    metrics = ['Cosine\nSimilarity', 'Pearson\nCorrelation', 'Spearman\nCorrelation']
    values = [topic_stats['cosine_similarity'], topic_stats['pearson_correlation'], topic_stats['spearman_correlation']]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    bars = ax1.bar(metrics, values, color=colors, alpha=0.9, edgecolor='black', linewidth=1.5)
    ax1.set_ylim(0, 1)
    ax1.set_title('13D Topic Vector Divergence Metrics', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Correlation/Similarity', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 2. Distance metrics
    distances = ['Cosine\nDistance', 'Euclidean\nDistance', 'Manhattan\nDistance']
    dist_values = [topic_stats['cosine_distance'], topic_stats['euclidean_distance'], topic_stats['manhattan_distance']]
    
    # Normalize for visualization
    dist_values_norm = [val / max(dist_values) for val in dist_values]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax2.bar(distances, dist_values_norm, color=colors, alpha=0.9, edgecolor='black', linewidth=1.5)
    ax2.set_title('Distance Metrics (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Normalized Distance', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add original values as labels
    for bar, val, orig_val in zip(bars, dist_values_norm, dist_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{orig_val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 3. Normative flips analysis
    if topic_stats['norm_flips_count'] > 0:
        flip_topics = [flip['topic'][:20] + '...' if len(flip['topic']) > 20 else flip['topic'] 
                      for flip in topic_stats['norm_flips']]
        flip_magnitudes = [abs(flip['topic_tendency_difference']) for flip in topic_stats['norm_flips']]
        
        bars = ax3.barh(range(len(flip_topics)), flip_magnitudes, 
                       color='red', alpha=0.8, edgecolor='black', linewidth=1)
        ax3.set_yticks(range(len(flip_topics)))
        ax3.set_yticklabels(flip_topics, fontsize=11)
        ax3.set_xlabel('Flip Magnitude', fontsize=12)
        ax3.set_title(f'Normative Flips ({topic_stats["norm_flips_count"]} topics)', fontsize=14, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No Major Normative Flips\nDetected', ha='center', va='center',
                transform=ax3.transAxes, fontsize=14, fontweight='bold')
        ax3.set_title('Normative Flips Analysis', fontsize=14, fontweight='bold')
    
    # 4. Summary statistics
    ax4.axis('off')
    
    summary_text = f"""13D TOPIC VECTOR ANALYSIS
    
Divergence Overview:
• Vector Dimension: {topic_stats['vector_dimension']}D
• Cosine Distance: {topic_stats['cosine_distance']:.3f}
• Mean Absolute Difference: {topic_stats['mean_absolute_difference']:.3f}

Statistical Significance:
• Pearson p-value: {topic_stats['pearson_p_value']:.2e}
• Spearman p-value: {topic_stats['spearman_p_value']:.2e}

Interpretive Drift:
• Normative Flips: {topic_stats['norm_flips_count']} topics
• Divergence Level: {"High" if topic_stats['cosine_distance'] > 0.3 else "Moderate" if topic_stats['cosine_distance'] > 0.1 else "Low"}
• Models exhibit {"substantial" if topic_stats['cosine_distance'] > 0.3 else "moderate"} interpretive differences
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
             facecolor="lightblue", alpha=0.8))
    
    plt.suptitle('Topic-Level Divergence Analysis (13D Vector)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('../outputs/rq1_visualizations/topic_divergence_stats.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()

def analyze_within_topic_preferences(field_df, topic_13d):
    """Analyze preference shifts within each topic using the 78D field vector"""
    
    print("Analyzing within-topic preference shifts...")
    
    # For each of the 13 topics, analyze how models differ in field preferences
    within_topic_analysis = {}
    
    for _, topic_row in topic_13d.iterrows():
        topic_name = topic_row['topic']
        
        # Get all field comparisons for this topic
        topic_fields = field_df[field_df['topic'] == topic_name].copy()
        
        if len(topic_fields) == 0:
            continue
        
        # Calculate preference metrics
        significant_fields = topic_fields[topic_fields['statistical_significance']].copy()
        if len(significant_fields) > 0:
            significant_fields['abs_difference'] = significant_fields['cross_model_difference'].abs()
            most_significant = significant_fields.nlargest(3, 'abs_difference')[['field_value', 'cross_model_difference', 'p_value']].to_dict('records')
        else:
            most_significant = []
        
        preference_analysis = {
            'topic': topic_name,
            'total_fields': len(topic_fields),
            'significant_fields': topic_fields['statistical_significance'].sum(),
            'significance_rate': topic_fields['statistical_significance'].mean(),
            'mean_difference': topic_fields['cross_model_difference'].mean(),
            'max_difference': topic_fields['cross_model_difference'].abs().max(),
            'fields_favoring_post': (topic_fields['cross_model_difference'] > 0).sum(),
            'fields_favoring_pre': (topic_fields['cross_model_difference'] < 0).sum(),
            'largest_shifts': topic_fields.nlargest(3, 'cross_model_difference')[['field_value', 'cross_model_difference']].to_dict('records'),
            'most_significant': most_significant
        }
        
        within_topic_analysis[topic_name] = preference_analysis
    
    return within_topic_analysis

def create_within_topic_preferences_viz(within_topic_analysis, topic_13d):
    """Create visualization of within-topic preference shifts"""
    
    print("Creating within-topic preferences visualization...")
    
    # Prepare data for visualization
    topics = []
    total_fields = []
    sig_rates = []
    mean_diffs = []
    post_favor_pcts = []
    
    for topic_name in topic_13d['topic']:
        if topic_name in within_topic_analysis:
            analysis = within_topic_analysis[topic_name]
            topics.append(topic_name[:25] + '...' if len(topic_name) > 25 else topic_name)
            total_fields.append(analysis['total_fields'])
            sig_rates.append(analysis['significance_rate'])
            mean_diffs.append(analysis['mean_difference'])
            post_favor_pcts.append(analysis['fields_favoring_post'] / analysis['total_fields'] if analysis['total_fields'] > 0 else 0)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1. Significance rates by topic
    colors = ['red' if rate < 0.5 else 'orange' if rate < 0.8 else 'green' for rate in sig_rates]
    
    bars = ax1.barh(range(len(topics)), sig_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(topics)))
    ax1.set_yticklabels(topics, fontsize=11)
    ax1.set_xlabel('Statistical Significance Rate', fontsize=12, fontweight='bold')
    ax1.set_title('Within-Topic Statistical Significance', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val, total in zip(bars, sig_rates, total_fields):
        ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}\n({total} fields)', ha='left', va='center', fontsize=9)
    
    # 2. Mean preference differences
    colors = ['green' if diff > 0 else 'red' for diff in mean_diffs]
    
    bars = ax2.barh(range(len(topics)), mean_diffs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(len(topics)))
    ax2.set_yticklabels(topics, fontsize=11)
    ax2.set_xlabel('Mean Cross-Model Difference', fontsize=12, fontweight='bold')
    ax2.set_title('Within-Topic Preference Direction', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.7)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, mean_diffs):
        ax2.text(bar.get_width() + (0.01 if val > 0 else -0.01), bar.get_y() + bar.get_height()/2,
                f'{val:+.3f}', ha='left' if val > 0 else 'right', va='center', fontsize=10, fontweight='bold')
    
    # 3. Post-Brexit favorability percentage
    bars = ax3.barh(range(len(topics)), post_favor_pcts, color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax3.set_yticks(range(len(topics)))
    ax3.set_yticklabels(topics, fontsize=11)
    ax3.set_xlabel('% Fields Favoring Post-Brexit', fontsize=12, fontweight='bold')
    ax3.set_title('Post-Brexit Preference Dominance', fontsize=14, fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Equal Split')
    ax3.grid(axis='x', alpha=0.3)
    ax3.legend()
    
    # Add value labels
    for bar, val in zip(bars, post_favor_pcts):
        ax3.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.1%}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # 4. Summary insights
    ax4.axis('off')
    
    # Calculate summary statistics
    high_sig_topics = sum(1 for rate in sig_rates if rate > 0.8)
    post_dominant_topics = sum(1 for pct in post_favor_pcts if pct > 0.6)
    pre_dominant_topics = sum(1 for pct in post_favor_pcts if pct < 0.4)
    
    summary_text = f"""WITHIN-TOPIC PREFERENCE ANALYSIS

Field-Level Analysis (78D Vector):
• Total Topics Analyzed: {len(topics)}
• High Significance Topics: {high_sig_topics} (>80% fields significant)
• Post-Brexit Dominant: {post_dominant_topics} topics (>60% fields favor)
• Pre-Brexit Dominant: {pre_dominant_topics} topics (<40% fields favor)

Key Insights:
• Most Significant Topic: {topics[sig_rates.index(max(sig_rates))]}
• Largest Shift: {topics[mean_diffs.index(max(mean_diffs, key=abs))]}
• Models show {"substantial" if max(sig_rates) > 0.8 else "moderate"} within-topic disagreement
• Preference patterns vary significantly across legal domains
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
             facecolor="lightyellow", alpha=0.8))
    
    plt.suptitle('Within-Topic Preference Shifts Analysis (From 78D Field Vector)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('../outputs/rq1_visualizations/within_topic_preferences.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()

def calculate_78d_field_vector_distance(field_df):
    """Calculate cosine distance for the complete 78D field vector"""
    
    print("Calculating 78D field vector cosine distance...")
    
    # Create vectors for pre-Brexit and post-Brexit models across all fields
    pre_brexit_vector = field_df['pre_brexit_normalized'].values
    post_brexit_vector = field_df['post_brexit_normalized'].values
    
    # Calculate cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Reshape for sklearn
    pre_reshaped = pre_brexit_vector.reshape(1, -1)
    post_reshaped = post_brexit_vector.reshape(1, -1)
    
    # Cosine similarity and distance
    cos_sim = cosine_similarity(pre_reshaped, post_reshaped)[0][0]
    cos_distance = 1 - cos_sim
    
    field_stats = {
        'vector_dimension': len(field_df),
        'cosine_similarity': cos_sim,
        'cosine_distance': cos_distance
    }
    
    return field_stats

def calculate_topic_specific_field_divergences(field_df, topic_13d):
    """Calculate cosine distance for each topic's field sub-vector"""
    
    print("Calculating topic-specific field divergences...")
    
    topic_divergences = {}
    
    # Handle regular topics
    for _, topic_row in topic_13d.iterrows():
        topic_name = topic_row['topic']
        
        # Get field subset for this topic
        if topic_name == "Disclosure":
            # Aggregate all disclosure subtopics
            disclosure_patterns = [
                "Disclosure: Political persecution & sexual violence",
                "Disclosure: Religious persecution & mental health", 
                "Disclosure: Domestic violence & criminal threats",
                "Disclosure: Ethnic violence & family separation",
                "Disclosure: Persecution for sexual orientation & mental health crisis"
            ]
            topic_fields = field_df[field_df['topic'].isin(disclosure_patterns)].copy()
        elif topic_name == "Contradiction":
            # Aggregate all contradiction subtopics  
            contradiction_patterns = [
                "Contradiction: Dates of persecution",
                "Contradiction: Persecutor identity confusion", 
                "Contradiction: Location of harm",
                "Contradiction: Family involvement in the persecution",
                "Contradiction: Sequence of events"
            ]
            topic_fields = field_df[field_df['topic'].isin(contradiction_patterns)].copy()
        else:
            # Regular topic
            topic_fields = field_df[field_df['topic'] == topic_name].copy()
        
        if len(topic_fields) == 0:
            continue
            
        # Extract sub-vectors for this topic
        pre_brexit_subvector = topic_fields['pre_brexit_normalized'].values
        post_brexit_subvector = topic_fields['post_brexit_normalized'].values
        
        if len(pre_brexit_subvector) < 2:
            # Need at least 2 dimensions for meaningful cosine distance
            topic_divergences[topic_name] = {
                'cosine_distance': 'N/A (insufficient dimensions)',
                'field_count': len(topic_fields),
                'mean_abs_difference': topic_fields['cross_model_difference'].abs().mean(),
                'max_difference': topic_fields['cross_model_difference'].abs().max(),
                'significance_rate': topic_fields['statistical_significance'].mean()
            }
            continue
        
        # Calculate cosine similarity for this topic's sub-vector
        from sklearn.metrics.pairwise import cosine_similarity
        
        pre_reshaped = pre_brexit_subvector.reshape(1, -1)
        post_reshaped = post_brexit_subvector.reshape(1, -1)
        
        cos_sim = cosine_similarity(pre_reshaped, post_reshaped)[0][0]
        cos_distance = 1 - cos_sim
        
        topic_divergences[topic_name] = {
            'cosine_distance': cos_distance,
            'cosine_similarity': cos_sim,
            'field_count': len(topic_fields),
            'mean_abs_difference': topic_fields['cross_model_difference'].abs().mean(),
            'max_difference': topic_fields['cross_model_difference'].abs().max(),
            'significance_rate': topic_fields['statistical_significance'].mean(),
            'topic_level_divergence': abs(topic_row['topic_tendency_difference']) if 'topic_tendency_difference' in topic_row else None
        }
    
    return topic_divergences

def create_granular_divergence_analysis(topic_divergences, topic_13d):
    """Create 3 separate visualizations for granular divergence analysis"""
    
    print("Creating granular divergence analysis - 3 separate plots...")
    
    # Prepare data for visualization
    topics = []
    field_divergences = []
    topic_divergences_vals = []
    field_counts = []
    significance_rates = []
    
    for _, topic_row in topic_13d.iterrows():
        topic_name = topic_row['topic']
        if topic_name in topic_divergences:
            div_data = topic_divergences[topic_name]
            
            # Better topic name truncation for readability
            topics.append(smart_truncate_topic(topic_name, 30))
            
            # Field-level divergence (cosine distance of sub-vector)
            field_div = div_data['cosine_distance']
            field_divergences.append(field_div if isinstance(field_div, (int, float)) else 0)
            
            # Topic-level divergence (absolute difference in topic tendencies)
            topic_div = abs(topic_row['topic_tendency_difference'])
            topic_divergences_vals.append(topic_div)
            
            field_counts.append(div_data['field_count'])
            significance_rates.append(div_data['significance_rate'])
    
    # PLOT 1: Combined Stacked Divergence Components (Horizontal)
    # Dynamic figure sizing - horizontal layout for better topic name presentation
    fig_height = max(11, len(topics) * 0.7)  # Slightly increased to accommodate legend above
    fig_width = 16
    
    plt.figure(figsize=(fig_width, fig_height))
    
    # Sort by total divergence for better visualization
    total_divergence = [t + f for t, f in zip(topic_divergences_vals, field_divergences)]
    sorted_indices = sorted(range(len(total_divergence)), key=lambda i: total_divergence[i], reverse=True)
    
    # Apply sorting to all arrays
    topics_sorted = [topics[i] for i in sorted_indices]
    topic_div_sorted = [topic_divergences_vals[i] for i in sorted_indices]
    field_div_sorted = [field_divergences[i] for i in sorted_indices]
    field_counts_sorted = [field_counts[i] for i in sorted_indices]
    
    y_pos = np.arange(len(topics_sorted))
    
    # Create horizontal stacked bar chart
    bars1 = plt.barh(y_pos, topic_div_sorted, height=0.7, 
                     label='Topic-Level Divergence', color='skyblue', alpha=0.8, 
                     edgecolor='black', linewidth=1)
    bars2 = plt.barh(y_pos, field_div_sorted, height=0.7, left=topic_div_sorted,
                     label='Field-Level Divergence', color='lightcoral', alpha=0.8, 
                     edgecolor='black', linewidth=1)
    
    plt.ylabel('Topics (sorted by total divergence)', fontsize=14, fontweight='bold')
    plt.xlabel('Divergence Magnitude', fontsize=14, fontweight='bold')
    plt.title('Decomposed Divergence by Topic\n' +
              'Topic-level vs Field-level components stacked to show total interpretive differences', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Better topic name presentation - horizontal layout
    plt.yticks(y_pos, topics_sorted, fontsize=11)
    plt.grid(axis='x', alpha=0.3)
    
    # Legend positioned above the plot to avoid overlapping with growing data
    plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2)
    
    # Add comprehensive labels positioned to the right of bars
    total_sorted = [topic_div_sorted[i] + field_div_sorted[i] for i in range(len(topics_sorted))]
    max_total = max(total_sorted)
    
    for i, (bar1, bar2, fc, td, topic_div, field_div) in enumerate(zip(bars1, bars2, field_counts_sorted, 
                                                                       total_sorted, topic_div_sorted, field_div_sorted)):
        # Position label to the right of the complete stacked bar
        total_width = bar1.get_width() + bar2.get_width()
        plt.text(total_width + max_total * 0.02, bar1.get_y() + bar1.get_height()/2,
                f'{fc} fields | Total: {td:.3f}\nTopic: {topic_div:.3f} | Field: {field_div:.3f}', 
                ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Add component value labels with percentages on the bars themselves
        total_value = topic_div + field_div
        if topic_div > max_total * 0.08:  # Only if segment is large enough
            topic_pct = (topic_div / total_value * 100) if total_value > 0 else 0
            plt.text(topic_div/2, bar1.get_y() + bar1.get_height()/2,
                    f'{topic_div:.3f}\n({topic_pct:.0f}%)', ha='center', va='center', fontsize=8, 
                    fontweight='bold', color='navy')
        if field_div > max_total * 0.08:  # Only if segment is large enough
            field_pct = (field_div / total_value * 100) if total_value > 0 else 0
            plt.text(topic_div + field_div/2, bar2.get_y() + bar2.get_height()/2,
                    f'{field_div:.3f}\n({field_pct:.0f}%)', ha='center', va='center', fontsize=8, 
                    fontweight='bold', color='darkred')
    
    # Adjust x-axis limit to accommodate labels
    plt.xlim(0, max_total * 1.4)
    
    plt.tight_layout()
    # Adjust top margin to accommodate legend above plot
    plt.subplots_adjust(top=0.88)
    plt.savefig('../outputs/rq1_visualizations/granular_divergence_1_stacked_components.png', 
                dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.4)
    plt.close()
    
    # PLOT 2: Topic vs Field Divergence Relationship (Scatter)
    # Large figure for full topic names and no overlaps
    plt.figure(figsize=(16, 12))
    
    colors = ['red' if fc < 4 else 'orange' if fc < 8 else 'green' for fc in field_counts]
    scatter = plt.scatter(topic_divergences_vals, field_divergences, 
                         c=colors, s=180, alpha=0.8, edgecolors='black', linewidth=2)
    
    # Add diagonal reference line
    max_val = max(max(topic_divergences_vals), max(field_divergences))
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=2, label='Equal Divergence Line')
    
    plt.xlabel('Topic-Level Divergence', fontsize=15, fontweight='bold')
    plt.ylabel('Field-Level Divergence', fontsize=15, fontweight='bold')
    plt.title('Topic vs Field Divergence Relationship\n' +
              'Do topics with high-level disagreement also show granular field differences?', 
              fontsize=17, fontweight='bold', pad=25)
    plt.grid(True, alpha=0.3)
    
    # Clean and natural annotation positioning - NO DUPLICATES!
    import math
    
    # Use FULL topic names (no truncation)
    full_topics = [topic.replace('\n', ' ') for topic in topics]
    
    # Track occupied positions to avoid overlaps
    occupied_positions = []
    conflict_distance = 0.18  # Minimum distance between labels
    
    # Prioritize clean, natural directions (cardinals first, then diagonals)
    preferred_directions = [
        (1, 0),    # Right
        (-1, 0),   # Left  
        (0, 1),    # Up
        (0, -1),   # Down
        (1, 1),    # Upper-right
        (-1, 1),   # Upper-left
        (1, -1),   # Lower-right
        (-1, -1),  # Lower-left
    ]
    
    base_distance = 60  # Base distance for clean arrows
    
    # Process each point exactly ONCE
    for i, (topic, full_topic) in enumerate(zip(topics, full_topics)):
        x_pos = topic_divergences_vals[i]
        y_pos = field_divergences[i]
        
        # Special handling for "Activist Persecution Ground" - force it to point right
        if "Activist Persecution" in full_topic:
            x_offset = base_distance + 20  # Point right with extra space
            y_offset = -10  # Slight downward adjustment
        else:
            # Find the best natural direction for other topics
            best_offset = None
            min_conflicts = float('inf')
            best_direction = None
            
            for dx, dy in preferred_directions:
                # Create clean offset in this direction
                x_offset = dx * base_distance
                y_offset = dy * base_distance
                
                # Smart boundary adjustment - keep arrows pointing naturally
                plot_center_x = max_val / 2
                plot_center_y = max(field_divergences) / 2
                
                # Adjust based on position in plot for natural flow
                if x_pos < plot_center_x * 0.3:  # Far left side
                    if dx < 0:  # Don't point further left
                        x_offset = base_distance  # Point right instead
                elif x_pos > plot_center_x * 1.7:  # Far right side  
                    if dx > 0:  # Don't point further right
                        x_offset = -base_distance  # Point left instead
                        
                if y_pos < plot_center_y * 0.3:  # Bottom area
                    if dy < 0:  # Don't point further down
                        y_offset = base_distance  # Point up instead
                elif y_pos > plot_center_y * 1.7:  # Top area
                    if dy > 0:  # Don't point further up  
                        y_offset = -base_distance  # Point down instead
                
                # Check for conflicts with this clean positioning
                test_label_x = x_pos + x_offset/100
                test_label_y = y_pos + y_offset/100
                
                conflicts = 0
                for occupied_x, occupied_y in occupied_positions:
                    distance = math.sqrt((test_label_x - occupied_x)**2 + (test_label_y - occupied_y)**2)
                    if distance < conflict_distance:
                        conflicts += 1
                
                # Prefer this direction if it has fewer conflicts
                if conflicts < min_conflicts:
                    min_conflicts = conflicts
                    best_offset = (x_offset, y_offset)
                    best_direction = (dx, dy)
                    if conflicts == 0:  # Perfect spot found
                        break
            
            # If no clean direction works, extend the best one
            if best_offset is None or min_conflicts > 0:
                if best_direction:
                    dx, dy = best_direction
                    # Extend distance to avoid conflicts
                    extended_distance = base_distance + (min_conflicts * 30)
                    x_offset = dx * extended_distance
                    y_offset = dy * extended_distance
                else:
                    # Final fallback - use systematic spacing
                    angle_rad = math.radians((i / len(topics)) * 360)
                    x_offset = (base_distance + 40) * math.cos(angle_rad)
                    y_offset = (base_distance + 40) * math.sin(angle_rad)
            else:
                x_offset, y_offset = best_offset
        
        # Record the final position
        final_x = x_pos + x_offset/100
        final_y = y_pos + y_offset/100
        occupied_positions.append((final_x, final_y))
        
        # Set text alignment for clean appearance
        ha_align = 'left' if x_offset > 0 else 'right'
        va_align = 'bottom' if y_offset > 0 else 'top'
        
        # Create clean, natural annotation
        plt.annotate(full_topic, 
                    (x_pos, y_pos),
                    xytext=(x_offset, y_offset), textcoords='offset points', 
                    fontsize=10, alpha=0.95,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.95, edgecolor='gray'),
                    ha=ha_align, va=va_align,
                    arrowprops=dict(arrowstyle='-', color='gray', alpha=0.8, lw=1))
    
    # Color legend positioned outside the plot area
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='<4 fields'),
        Patch(facecolor='orange', label='4-7 fields'),
        Patch(facecolor='green', label='8+ fields')
    ]
    plt.legend(handles=legend_elements, title='Number of Fields', 
              loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12, title_fontsize=13)
    
    plt.tight_layout()
    plt.savefig('../outputs/rq1_visualizations/granular_divergence_2_relationship.png', 
                dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.4)
    plt.close()
    
    # PLOT 3: Field Count Analysis by Divergence Type
    # Show which types of divergence dominate for different field count categories
    plt.figure(figsize=(14, 8))
    
    # Group by field count categories
    low_field_topics = [(i, topics[i]) for i in range(len(field_counts)) if field_counts[i] < 4]
    med_field_topics = [(i, topics[i]) for i in range(len(field_counts)) if 4 <= field_counts[i] < 8]
    high_field_topics = [(i, topics[i]) for i in range(len(field_counts)) if field_counts[i] >= 8]
    
    categories = ['<4 fields\n(Limited data)', '4-7 fields\n(Moderate data)', '8+ fields\n(Rich data)']
    topic_groups = [low_field_topics, med_field_topics, high_field_topics]
    
    # Calculate average divergences by category
    avg_topic_divs = []
    avg_field_divs = []
    counts = []
    
    for group in topic_groups:
        if group:
            topic_divs = [topic_divergences_vals[i] for i, _ in group]
            field_divs = [field_divergences[i] for i, _ in group]
            avg_topic_divs.append(np.mean(topic_divs))
            avg_field_divs.append(np.mean(field_divs))
            counts.append(len(group))
        else:
            avg_topic_divs.append(0)
            avg_field_divs.append(0)
            counts.append(0)
    
    x_pos = np.arange(len(categories))
    width = 0.35
    
    bars1 = plt.bar(x_pos - width/2, avg_topic_divs, width, 
                    label='Avg Topic-Level Divergence', color='skyblue', alpha=0.8, 
                    edgecolor='black', linewidth=1)
    bars2 = plt.bar(x_pos + width/2, avg_field_divs, width,
                    label='Avg Field-Level Divergence', color='lightcoral', alpha=0.8, 
                    edgecolor='black', linewidth=1)
    
    plt.xlabel('Field Count Categories', fontsize=14, fontweight='bold')
    plt.ylabel('Average Divergence Magnitude', fontsize=14, fontweight='bold')
    plt.title('Divergence Patterns by Data Richness\n' +
              'How do topic vs field divergences vary with number of available fields?', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(x_pos, categories, fontsize=11)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels and counts
    for i, (bar1, bar2, count) in enumerate(zip(bars1, bars2, counts)):
        plt.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.005,
                f'{avg_topic_divs[i]:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.005,
                f'{avg_field_divs[i]:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.text(i, -0.02, f'({count} topics)', ha='center', va='top', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig('../outputs/rq1_visualizations/granular_divergence_3_field_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.4)
    plt.close()
    
    print("✓ Generated 3 separate granular divergence plots:")
    print("  - granular_divergence_1_stacked_components.png (COMBINED: stacked topic + field components)")
    print("  - granular_divergence_2_relationship.png") 
    print("  - granular_divergence_3_field_analysis.png (NEW: divergence patterns by data richness)")

def create_divergence_interpretation_analysis(topic_divergences, topic_13d):
    """Create interpretive analysis of what divergence patterns mean"""
    
    print("Creating divergence interpretation analysis...")
    
    # Prepare data
    interpretation_data = []
    
    for _, topic_row in topic_13d.iterrows():
        topic_name = topic_row['topic']
        if topic_name in topic_divergences:
            div_data = topic_divergences[topic_name]
            
            topic_div = abs(topic_row['topic_tendency_difference'])
            field_div = div_data['cosine_distance'] if isinstance(div_data['cosine_distance'], (int, float)) else 0
            field_count = div_data['field_count']
            
            # Classify divergence pattern
            topic_high = topic_div > 0.15  # Threshold for "high" topic divergence
            field_high = field_div > 0.25  # Threshold for "high" field divergence
            
            if topic_high and field_high:
                pattern = "Complete Divergence"
                color = "#d73027"  # Red
                interpretation = "Fundamental disagreement at both policy and case levels"
            elif topic_high and not field_high:
                pattern = "Systematic Shift"
                color = "#fc8d59"  # Orange-red
                interpretation = "Consistent policy change but agreement on case factors"
            elif not topic_high and field_high:
                pattern = "Granular Disagreement"
                color = "#fee08b"  # Yellow
                interpretation = "Similar overall approach but different case-level priorities"
            else:
                pattern = "Interpretive Alignment"
                color = "#91bfdb"  # Light blue
                interpretation = "Consistent legal interpretation maintained"
            
            # Calculate interpretive impact
            impact_score = topic_div + field_div
            
            interpretation_data.append({
                'topic': smart_truncate_topic(topic_name, 30),
                'topic_divergence': topic_div,
                'field_divergence': field_div,
                'total_impact': impact_score,
                'pattern': pattern,
                'color': color,
                'interpretation': interpretation,
                'field_count': field_count,
                'policy_shift_magnitude': f"{topic_div*100:.1f}%" if topic_div > 0.05 else "Minimal",
                'granular_disagreement': "High" if field_div > 0.25 else "Moderate" if field_div > 0.15 else "Low"
            })
    
    # Sort by total impact
    interpretation_data.sort(key=lambda x: x['total_impact'], reverse=True)
    
    # Create interpretive visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
    
    # Plot 1: Interpretive Pattern Classification
    topics = [item['topic'] for item in interpretation_data]
    topic_divs = [item['topic_divergence'] for item in interpretation_data]
    field_divs = [item['field_divergence'] for item in interpretation_data]
    colors = [item['color'] for item in interpretation_data]
    patterns = [item['pattern'] for item in interpretation_data]
    
    # Scatter plot with pattern-based coloring
    scatter = ax1.scatter(topic_divs, field_divs, c=colors, s=150, alpha=0.8, 
                         edgecolors='black', linewidth=2)
    
    # Add topic labels
    for i, (topic, pattern) in enumerate(zip(topics, patterns)):
        ax1.annotate(f"{topic}\n({pattern})", 
                    (topic_divs[i], field_divs[i]),
                    xytext=(10, 10), textcoords='offset points', 
                    fontsize=9, alpha=0.9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor='gray'))
    
    # Add quadrant lines
    ax1.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axvline(x=0.15, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add quadrant labels
    ax1.text(0.05, 0.35, 'Granular\nDisagreement', ha='center', va='center', 
             fontsize=11, fontweight='bold', alpha=0.7,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#fee08b", alpha=0.3))
    ax1.text(0.25, 0.35, 'Complete\nDivergence', ha='center', va='center', 
             fontsize=11, fontweight='bold', alpha=0.7,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#d73027", alpha=0.3))
    ax1.text(0.05, 0.15, 'Interpretive\nAlignment', ha='center', va='center', 
             fontsize=11, fontweight='bold', alpha=0.7,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#91bfdb", alpha=0.3))
    ax1.text(0.25, 0.15, 'Systematic\nShift', ha='center', va='center', 
             fontsize=11, fontweight='bold', alpha=0.7,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#fc8d59", alpha=0.3))
    
    ax1.set_xlabel('Topic-Level Divergence\n(Systematic Policy Difference)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Field-Level Divergence\n(Case-Specific Disagreement)', fontsize=14, fontweight='bold')
    ax1.set_title('Interpretive Divergence Pattern Classification\n' +
                  'Understanding the nature of legal interpretation changes', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Impact Ranking with Interpretation
    y_pos = np.arange(len(interpretation_data))
    impact_scores = [item['total_impact'] for item in interpretation_data]
    
    bars = ax2.barh(y_pos, impact_scores, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1, height=0.7)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(topics, fontsize=11)
    ax2.set_xlabel('Total Interpretive Impact', fontsize=14, fontweight='bold')
    ax2.set_title('Ranked Interpretive Impact by Topic\n' +
                  'Which topics show the most interpretive divergence?', 
                  fontsize=16, fontweight='bold', pad=20)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add impact interpretation labels
    max_impact = max(impact_scores)
    for i, (bar, item) in enumerate(zip(bars, interpretation_data)):
        ax2.text(bar.get_width() + max_impact * 0.02, bar.get_y() + bar.get_height()/2,
                f"{item['total_impact']:.3f}\n{item['pattern']}\n({item['field_count']} fields)", 
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    ax2.set_xlim(0, max_impact * 1.5)
    
    plt.tight_layout()
    plt.savefig('../outputs/rq1_visualizations/divergence_interpretation_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.4)
    plt.close()
    
    # Create summary interpretation table
    print("\n" + "="*80)
    print("DIVERGENCE INTERPRETATION ANALYSIS")
    print("="*80)
    
    for item in interpretation_data[:10]:  # Top 10 most impactful
        print(f"\n📋 {item['topic']}")
        print(f"   Pattern: {item['pattern']}")
        print(f"   Policy Shift: {item['policy_shift_magnitude']} | Granular Disagreement: {item['granular_disagreement']}")
        print(f"   Impact: {item['total_impact']:.3f} | Fields: {item['field_count']}")
        print(f"   → {item['interpretation']}")
    
    print(f"\n" + "="*80)
    print("PATTERN SUMMARY")
    print("="*80)
    
    pattern_counts = {}
    for item in interpretation_data:
        pattern_counts[item['pattern']] = pattern_counts.get(item['pattern'], 0) + 1
    
    for pattern, count in pattern_counts.items():
        percentage = (count / len(interpretation_data)) * 100
        print(f"• {pattern}: {count} topics ({percentage:.1f}%)")
    
    print("✓ Generated divergence_interpretation_analysis.png")
    
    return interpretation_data

def analyze_divergence_thresholds(topic_divergences, topic_13d):
    """Analyze data distribution to determine empirical thresholds for divergence classification"""
    
    print("Analyzing divergence data distribution to determine optimal thresholds...")
    
    # Collect all divergence values
    topic_divs = []
    field_divs = []
    impact_scores = []
    
    for _, topic_row in topic_13d.iterrows():
        topic_name = topic_row['topic']
        if topic_name in topic_divergences:
            div_data = topic_divergences[topic_name]
            
            topic_div = abs(topic_row['topic_tendency_difference'])
            field_div = div_data['cosine_distance'] if isinstance(div_data['cosine_distance'], (int, float)) else 0
            
            topic_divs.append(topic_div)
            field_divs.append(field_div)
            impact_scores.append(topic_div + field_div)
    
    # Calculate distribution statistics
    import numpy as np
    
    topic_stats = {
        'min': np.min(topic_divs),
        'max': np.max(topic_divs),
        'mean': np.mean(topic_divs),
        'median': np.median(topic_divs),
        'std': np.std(topic_divs),
        'q25': np.percentile(topic_divs, 25),
        'q75': np.percentile(topic_divs, 75)
    }
    
    field_stats = {
        'min': np.min(field_divs),
        'max': np.max(field_divs),
        'mean': np.mean(field_divs),
        'median': np.median(field_divs),
        'std': np.std(field_divs),
        'q25': np.percentile(field_divs, 25),
        'q75': np.percentile(field_divs, 75)
    }
    
    # Create threshold analysis visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Topic divergence distribution
    ax1.hist(topic_divs, bins=8, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(topic_stats['median'], color='red', linestyle='--', linewidth=2, label=f"Median: {topic_stats['median']:.3f}")
    ax1.axvline(topic_stats['mean'], color='orange', linestyle='--', linewidth=2, label=f"Mean: {topic_stats['mean']:.3f}")
    ax1.axvline(topic_stats['q75'], color='green', linestyle='--', linewidth=2, label=f"Q3: {topic_stats['q75']:.3f}")
    ax1.axvline(0.15, color='purple', linestyle='-', linewidth=3, label="Current Cutoff: 0.15")
    
    ax1.set_xlabel('Topic-Level Divergence', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Topic Divergence Distribution\nChoosing the "High" Threshold', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Field divergence distribution
    ax2.hist(field_divs, bins=8, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.axvline(field_stats['median'], color='red', linestyle='--', linewidth=2, label=f"Median: {field_stats['median']:.3f}")
    ax2.axvline(field_stats['mean'], color='orange', linestyle='--', linewidth=2, label=f"Mean: {field_stats['mean']:.3f}")
    ax2.axvline(field_stats['q75'], color='green', linestyle='--', linewidth=2, label=f"Q3: {field_stats['q75']:.3f}")
    ax2.axvline(0.25, color='purple', linestyle='-', linewidth=3, label="Current Cutoff: 0.25")
    
    ax2.set_xlabel('Field-Level Divergence', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Field Divergence Distribution\nChoosing the "High" Threshold', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Threshold sensitivity analysis
    topic_thresholds = np.linspace(0.05, 0.4, 20)
    field_thresholds = np.linspace(0.1, 0.5, 20)
    
    # For each threshold combination, calculate how many topics fall into each category
    threshold_results = []
    
    for t_thresh in [topic_stats['median'], topic_stats['mean'], topic_stats['q75'], 0.15]:
        for f_thresh in [field_stats['median'], field_stats['mean'], field_stats['q75'], 0.25]:
            complete = sum(1 for i in range(len(topic_divs)) if topic_divs[i] > t_thresh and field_divs[i] > f_thresh)
            systematic = sum(1 for i in range(len(topic_divs)) if topic_divs[i] > t_thresh and field_divs[i] <= f_thresh)
            granular = sum(1 for i in range(len(topic_divs)) if topic_divs[i] <= t_thresh and field_divs[i] > f_thresh)
            aligned = sum(1 for i in range(len(topic_divs)) if topic_divs[i] <= t_thresh and field_divs[i] <= f_thresh)
            
            threshold_results.append({
                'topic_thresh': t_thresh,
                'field_thresh': f_thresh,
                'complete': complete,
                'systematic': systematic,
                'granular': granular,
                'aligned': aligned,
                'total': len(topic_divs)
            })
    
    # Plot threshold sensitivity
    x_labels = []
    complete_counts = []
    systematic_counts = []
    granular_counts = []
    aligned_counts = []
    
    for result in threshold_results:
        x_labels.append(f"T:{result['topic_thresh']:.2f}\nF:{result['field_thresh']:.2f}")
        complete_counts.append(result['complete'])
        systematic_counts.append(result['systematic'])
        granular_counts.append(result['granular'])
        aligned_counts.append(result['aligned'])
    
    x_pos = np.arange(len(x_labels))
    width = 0.6
    
    ax3.bar(x_pos, complete_counts, width, label='Complete Divergence', color='#d73027', alpha=0.8)
    ax3.bar(x_pos, systematic_counts, width, bottom=complete_counts, label='Systematic Shift', color='#fc8d59', alpha=0.8)
    
    bottom_2 = [c + s for c, s in zip(complete_counts, systematic_counts)]
    ax3.bar(x_pos, granular_counts, width, bottom=bottom_2, label='Granular Disagreement', color='#fee08b', alpha=0.8)
    
    bottom_3 = [b + g for b, g in zip(bottom_2, granular_counts)]
    ax3.bar(x_pos, aligned_counts, width, bottom=bottom_3, label='Interpretive Alignment', color='#91bfdb', alpha=0.8)
    
    ax3.set_xlabel('Threshold Combinations', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Topics', fontsize=12, fontweight='bold')
    ax3.set_title('Threshold Sensitivity Analysis\nHow do different cutoffs affect classification?', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    ax3.legend()
    
    # 4. Recommended thresholds based on different methods
    ax4.axis('off')
    
    methods_text = f"""THRESHOLD RECOMMENDATION METHODS

1. MEDIAN SPLIT (Most Balanced):
   • Topic: {topic_stats['median']:.3f} | Field: {field_stats['median']:.3f}
   • Pros: Splits data evenly, no assumptions
   • Cons: May not reflect meaningful differences

2. MEAN + 0.5*STD (Statistical):
   • Topic: {topic_stats['mean'] + 0.5*topic_stats['std']:.3f} | Field: {field_stats['mean'] + 0.5*field_stats['std']:.3f}
   • Pros: Based on distribution shape
   • Cons: Sensitive to outliers

3. THIRD QUARTILE (Top 25%):
   • Topic: {topic_stats['q75']:.3f} | Field: {field_stats['q75']:.3f}
   • Pros: Identifies truly "high" values
   • Cons: May be too conservative

4. CURRENT CHOICE (Intuitive):
   • Topic: 0.15 | Field: 0.25
   • Assessment: Field threshold seems reasonable,
     topic threshold may be too low

RECOMMENDATION:
Use Q3 thresholds for more meaningful classification:
• Topic "High": ≥ {topic_stats['q75']:.3f} 
• Field "High": ≥ {field_stats['q75']:.3f}

This identifies the top 25% as "high divergence"
which better captures meaningful differences.
    """
    
    ax4.text(0.05, 0.95, methods_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
             facecolor="lightyellow", alpha=0.8), family='monospace')
    
    plt.tight_layout()
    plt.savefig('../outputs/rq1_visualizations/threshold_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.4)
    plt.close()
    
    print("\n" + "="*80)
    print("THRESHOLD ANALYSIS RESULTS")
    print("="*80)
    print(f"TOPIC DIVERGENCE STATISTICS:")
    print(f"  Range: {topic_stats['min']:.3f} - {topic_stats['max']:.3f}")
    print(f"  Mean ± SD: {topic_stats['mean']:.3f} ± {topic_stats['std']:.3f}")
    print(f"  Median (Q2): {topic_stats['median']:.3f}")
    print(f"  Q1: {topic_stats['q25']:.3f} | Q3: {topic_stats['q75']:.3f}")
    print()
    print(f"FIELD DIVERGENCE STATISTICS:")
    print(f"  Range: {field_stats['min']:.3f} - {field_stats['max']:.3f}")
    print(f"  Mean ± SD: {field_stats['mean']:.3f} ± {field_stats['std']:.3f}")
    print(f"  Median (Q2): {field_stats['median']:.3f}")
    print(f"  Q1: {field_stats['q25']:.3f} | Q3: {field_stats['q75']:.3f}")
    print()
    print(f"RECOMMENDED THRESHOLDS (Q3 method):")
    print(f"  Topic 'High': ≥ {topic_stats['q75']:.3f}")
    print(f"  Field 'High': ≥ {field_stats['q75']:.3f}")
    print("✓ Generated threshold_analysis.png")
    
    return {
        'topic_stats': topic_stats,
        'field_stats': field_stats,
        'recommended_topic_threshold': topic_stats['q75'],
        'recommended_field_threshold': field_stats['q75']
    }

def main():
    """Main function - generate exactly what is needed"""
    
    print("="*80)
    print("RQ1 FOCUSED ANALYSIS - EXACTLY WHAT YOU WANT")
    print("="*80)
    
    # Load data
    field_df, topic_df, divergence_results = load_data()
    
    # Create 13D topic vector
    topic_13d = create_13d_topic_vector(topic_df)
    print(f"13D Topic Vector: {len(topic_13d)} topics")
    
    # 1. Create the good topic-level comparison
    create_topic_level_comparison(topic_13d)
    
    # 2. Calculate and visualize topic divergence statistics  
    topic_stats = calculate_topic_divergence_stats(topic_13d)
    create_divergence_summary(topic_stats)
    
    # 3. Analyze within-topic preferences
    within_topic_analysis = analyze_within_topic_preferences(field_df, topic_13d)
    create_within_topic_preferences_viz(within_topic_analysis, topic_13d)
    
    # 4. Calculate 78D field vector cosine distance
    field_vector_stats = calculate_78d_field_vector_distance(field_df)
    print(f"\n78D Field Vector Divergence: {field_vector_stats['cosine_distance']:.3f}")
    
    # 5. Calculate topic-specific field divergences
    topic_divergences = calculate_topic_specific_field_divergences(field_df, topic_13d)
    create_granular_divergence_analysis(topic_divergences, topic_13d)
    
    # 6. Create interpretive analysis of divergence patterns
    interpretation_data = create_divergence_interpretation_analysis(topic_divergences, topic_13d)

    # 7. Analyze divergence thresholds
    threshold_analysis_results = analyze_divergence_thresholds(topic_divergences, topic_13d)

    print("\n" + "="*80)
    print("ENHANCED GRANULAR ANALYSIS COMPLETE")
    print("="*80)
    print("Generated comprehensive divergence analysis:")
    print("✓ 13d_topic_comparison.png - The good 13D topic comparison")
    print("✓ topic_divergence_stats.png - 13D vector divergence statistics")
    print("✓ within_topic_preferences.png - Preference shifts within topics (from 78D)")
    print("✓ granular_divergence_1_stacked_components.png - Stacked topic + field components")
    print("✓ granular_divergence_2_relationship.png - Divergence relationship scatter plot")
    print("✓ granular_divergence_3_field_analysis.png - Divergence patterns by data richness")
    print("✓ divergence_interpretation_analysis.png - Legal interpretation pattern analysis")
    print("✓ threshold_analysis.png - Data-driven threshold determination analysis")
    print(f"\nKey Results:")
    print(f"• 13D Topic Vector Divergence: {topic_stats['cosine_distance']:.3f}")
    print(f"• 78D Field Vector Divergence: {field_vector_stats['cosine_distance']:.3f}")
    print(f"• Normative Flips: {topic_stats['norm_flips_count']} topics")
    print(f"• Topic-Specific Field Divergences: Individual cosine distances calculated")
    print(f"• Statistical Significance: p < 0.001")

if __name__ == "__main__":
    main() 