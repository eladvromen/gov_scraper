import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import re

def load_processed_inference_results(file_path: str) -> List:
    """Load processed inference results from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data["processed_results"]  # Extract the processed_results array

def convert_processed_list_to_dict(data: List) -> Dict:
    """Convert list of processed cases to dictionary with topic and fields as key."""
    result = {}
    for case in data:
        # Create a unique key from topic and fields in metadata
        metadata = case['metadata']
        key = f"{metadata['topic']}_{metadata['fields']['country']}_{metadata['fields']['age']}"
        result[key] = {
            'topic': metadata['topic'],
            'fields': metadata['fields'],
            'vignette_text': metadata['vignette_text'],
            'decision': case['decision'],  # Already extracted and cleaned
            'reasoning': case['reasoning'],
            'original_response': case['original_response']
        }
    return result

def compare_decisions(pre_brexit_data: List, post_brexit_data: List) -> Tuple[Dict, pd.DataFrame]:
    """
    Compare decisions between pre and post Brexit models for matching cases.
    Returns:
    - Summary statistics
    - DataFrame with case-by-case comparison
    """
    # Convert lists to dictionaries for easier lookup
    pre_brexit_dict = convert_processed_list_to_dict(pre_brexit_data)
    post_brexit_dict = convert_processed_list_to_dict(post_brexit_data)
    
    comparison_stats = {
        'total_cases': 0,
        'matching_decisions': 0,
        'mismatching_decisions': 0,
        'decision_pairs': defaultdict(int),  # (pre_decision, post_decision) -> count
        'only_in_pre': 0,
        'only_in_post': 0
    }
    
    # Create comparison DataFrame
    comparison_data = []
    
    # Find common cases
    all_cases = set(pre_brexit_dict.keys()) | set(post_brexit_dict.keys())
    
    for case_id in all_cases:
        pre_case = pre_brexit_dict.get(case_id)
        post_case = post_brexit_dict.get(case_id)
        
        if pre_case and post_case:
            comparison_stats['total_cases'] += 1
            pre_decision = pre_case['decision']
            post_decision = post_case['decision']
            
            comparison_data.append({
                'case_id': case_id,
                'topic': pre_case['topic'],
                'country': pre_case['fields']['country'],
                'pre_brexit_decision': pre_decision,
                'post_brexit_decision': post_decision,
                'match': pre_decision == post_decision,
                'vignette_text': pre_case['vignette_text']
            })
            
            if pre_decision == post_decision:
                comparison_stats['matching_decisions'] += 1
            else:
                comparison_stats['mismatching_decisions'] += 1
            
            comparison_stats['decision_pairs'][(pre_decision, post_decision)] += 1
        
        elif pre_case:
            comparison_stats['only_in_pre'] += 1
        else:
            comparison_stats['only_in_post'] += 1
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_stats, comparison_df

def plot_decision_matrix(comparison_df: pd.DataFrame, output_dir: str):
    """Create a confusion matrix-style plot of decisions between models."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create cross-tabulation of decisions
    decision_matrix = pd.crosstab(
        comparison_df['pre_brexit_decision'],
        comparison_df['post_brexit_decision']
    )
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(decision_matrix, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Decision Comparison Matrix: Pre-Brexit vs Post-Brexit')
    plt.xlabel('Post-Brexit Decisions')
    plt.ylabel('Pre-Brexit Decisions')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/decision_matrix.png')
    plt.close()

def plot_decision_distributions_by_topic(pre_brexit_data: List, post_brexit_data: List, output_dir: str):
    """Create decision distribution plots grouped by topics for each model."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def create_distribution_data(data: List, model_name: str) -> pd.DataFrame:
        """Create DataFrame with decisions and topics for plotting."""
        plot_data = []
        for case in data:
            plot_data.append({
                'topic': case['metadata']['topic'],
                'decision': case['decision'],
                'model': model_name
            })
        return pd.DataFrame(plot_data)
    
    # Create DataFrames for both models
    pre_brexit_df = create_distribution_data(pre_brexit_data, 'Pre-Brexit')
    post_brexit_df = create_distribution_data(post_brexit_data, 'Post-Brexit')
    
    # Combine data for comparison plots
    combined_df = pd.concat([pre_brexit_df, post_brexit_df], ignore_index=True)
    
    # Plot 1: Side-by-side comparison of decision distributions by topic
    plt.figure(figsize=(16, 10))
    
    # Get unique topics and decisions for consistent ordering
    topics = sorted(combined_df['topic'].unique())
    decisions = sorted(combined_df['decision'].unique())
    
    # Create subplots for each topic
    n_topics = len(topics)
    n_cols = 3
    n_rows = (n_topics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, topic in enumerate(topics):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        topic_data = combined_df[combined_df['topic'] == topic]
        
        # Create cross-tabulation for this topic
        topic_crosstab = pd.crosstab(topic_data['model'], topic_data['decision'])
        
        # Plot grouped bar chart
        topic_crosstab.plot(kind='bar', ax=ax, rot=45)
        ax.set_title(f'Decisions for {topic}')
        ax.set_xlabel('Model')
        ax.set_ylabel('Count')
        ax.legend(title='Decision', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Hide empty subplots
    for i in range(n_topics, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/decision_distributions_by_topic.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Stacked bar chart showing proportions for each model
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Pre-Brexit model
    pre_brexit_crosstab = pd.crosstab(pre_brexit_df['topic'], pre_brexit_df['decision'])
    pre_brexit_proportions = pre_brexit_crosstab.div(pre_brexit_crosstab.sum(axis=1), axis=0)
    pre_brexit_proportions.plot(kind='bar', stacked=True, ax=ax1, colormap='Set3')
    ax1.set_title('Pre-Brexit Model: Decision Proportions by Topic')
    ax1.set_xlabel('Topic')
    ax1.set_ylabel('Proportion')
    ax1.legend(title='Decision', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    
    # Post-Brexit model
    post_brexit_crosstab = pd.crosstab(post_brexit_df['topic'], post_brexit_df['decision'])
    post_brexit_proportions = post_brexit_crosstab.div(post_brexit_crosstab.sum(axis=1), axis=0)
    post_brexit_proportions.plot(kind='bar', stacked=True, ax=ax2, colormap='Set3')
    ax2.set_title('Post-Brexit Model: Decision Proportions by Topic')
    ax2.set_xlabel('Topic')
    ax2.set_ylabel('Proportion')
    ax2.legend(title='Decision', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/decision_proportions_by_topic.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Overall decision distribution comparison
    plt.figure(figsize=(12, 6))
    
    # Count decisions for each model
    decision_counts = combined_df.groupby(['model', 'decision']).size().unstack(fill_value=0)
    decision_counts.plot(kind='bar', ax=plt.gca())
    plt.title('Overall Decision Distribution Comparison')
    plt.xlabel('Model')
    plt.ylabel('Count')
    plt.legend(title='Decision', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/overall_decision_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\nDecision Distribution Analysis:")
    print("\nPre-Brexit Model Decision Counts by Topic:")
    print(pre_brexit_crosstab)
    print("\nPost-Brexit Model Decision Counts by Topic:")
    print(post_brexit_crosstab)
    
    print("\nPre-Brexit Model Decision Proportions by Topic:")
    print(pre_brexit_proportions.round(3))
    print("\nPost-Brexit Model Decision Proportions by Topic:")
    print(post_brexit_proportions.round(3))

def analyze_mismatches(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze patterns in mismatching decisions."""
    return comparison_df[~comparison_df['match']].sort_values('topic')

def main():
    # File paths - updated to use processed files
    pre_brexit_file = "../inference/results/processed/processed_subset_inference_llama3_8b_pre_brexit_2013_2016_instruct_20250623_120225_20250623_122325.json"
    post_brexit_file = "../inference/results/processed/processed_subset_inference_llama3_8b_post_brexit_2019_2025_instruct_20250623_123821_20250623_124925.json"
    output_dir = "plots"
    
    # Load data
    pre_brexit_data = load_processed_inference_results(pre_brexit_file)
    post_brexit_data = load_processed_inference_results(post_brexit_file)
    
    # Compare decisions
    comparison_stats, comparison_df = compare_decisions(pre_brexit_data, post_brexit_data)
    
    # Generate visualizations
    plot_decision_matrix(comparison_df, output_dir)
    plot_decision_distributions_by_topic(pre_brexit_data, post_brexit_data, output_dir)
    
    # Print summary statistics
    print("\nComparison Summary:")
    print(f"Total matching cases: {comparison_stats['total_cases']}")
    print(f"Matching decisions: {comparison_stats['matching_decisions']} "
          f"({comparison_stats['matching_decisions']/comparison_stats['total_cases']*100:.2f}%)")
    print(f"Mismatching decisions: {comparison_stats['mismatching_decisions']} "
          f"({comparison_stats['mismatching_decisions']/comparison_stats['total_cases']*100:.2f}%)")
    print(f"\nCases only in pre-Brexit: {comparison_stats['only_in_pre']}")
    print(f"Cases only in post-Brexit: {comparison_stats['only_in_post']}")
    
    print("\nDecision Pair Analysis:")
    for (pre_decision, post_decision), count in sorted(
        comparison_stats['decision_pairs'].items(), 
        key=lambda x: x[1], 
        reverse=True
    ):
        if pre_decision != post_decision:
            print(f"Pre-Brexit: {pre_decision} → Post-Brexit: {post_decision}: {count} cases")
    
    print("\nMismatched Cases Analysis:")
    mismatches_df = analyze_mismatches(comparison_df)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    print("\nMismatches by Topic:")
    topic_mismatches = mismatches_df.groupby('topic').size()
    print(topic_mismatches)
    
    print("\nDetailed Mismatches:")
    for _, row in mismatches_df.iterrows():
        print(f"\nTopic: {row['topic']}")
        print(f"Country: {row['country']}")
        print(f"Pre-Brexit Decision: {row['pre_brexit_decision']}")
        print(f"Post-Brexit Decision: {row['post_brexit_decision']}")
        print("Vignette:")
        print(row['vignette_text'])
        print("-" * 80)

if __name__ == "__main__":
    main() 