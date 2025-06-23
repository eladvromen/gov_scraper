import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import re

def load_processed_inference_results(filepath):
    """Load processed inference results from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract the processed results array
    processed_results = data.get('processed_results', [])
    
    # Create a dictionary mapping sample_id to the full result
    results_dict = {}
    missing_sample_id_count = 0
    
    for result in processed_results:
        sample_id = result.get('sample_id')
        if sample_id is None:
            missing_sample_id_count += 1
            print(f"Warning: Result missing sample_id: {result.get('metadata', {}).get('topic', 'Unknown')}")
            continue
        
        # Store the complete result
        results_dict[sample_id] = result
    
    if missing_sample_id_count > 0:
        print(f"Warning: {missing_sample_id_count} results missing sample_id")
    
    print(f"Loaded {len(results_dict)} results with sample_id from {filepath}")
    return results_dict

def compare_models(pre_brexit_results, post_brexit_results):
    """Compare model decisions using sample_id for accurate matching"""
    
    # Find common sample_ids between the two datasets
    pre_brexit_ids = set(pre_brexit_results.keys())
    post_brexit_ids = set(post_brexit_results.keys())
    common_ids = pre_brexit_ids.intersection(post_brexit_ids)
    
    print(f"Pre-Brexit dataset: {len(pre_brexit_ids)} samples")
    print(f"Post-Brexit dataset: {len(post_brexit_ids)} samples")
    print(f"Common sample_ids: {len(common_ids)} samples")
    
    if len(common_ids) == 0:
        print("ERROR: No matching sample_ids found between datasets!")
        print("Sample pre-Brexit IDs:", list(pre_brexit_ids)[:5])
        print("Sample post-Brexit IDs:", list(post_brexit_ids)[:5])
        return None
    
    # Compare decisions for matching samples
    comparison_results = []
    
    for sample_id in common_ids:
        pre_result = pre_brexit_results[sample_id]
        post_result = post_brexit_results[sample_id]
        
        pre_decision = pre_result.get('decision')
        post_decision = post_result.get('decision')
        
        # Extract metadata for analysis
        metadata = pre_result.get('metadata', {})
        
        comparison = {
            'sample_id': sample_id,
            'topic': metadata.get('topic'),
            'meta_topic': metadata.get('meta_topic'),
            'fields': metadata.get('fields', {}),
            'vignette_text': metadata.get('vignette_text'),
            'pre_brexit_decision': pre_decision,
            'post_brexit_decision': post_decision,
            'decisions_match': pre_decision == post_decision,
            'pre_brexit_reasoning': pre_result.get('reasoning'),
            'post_brexit_reasoning': post_result.get('reasoning'),
            'pre_brexit_response': pre_result.get('original_response'),
            'post_brexit_response': post_result.get('original_response')
        }
        
        comparison_results.append(comparison)
    
    # Calculate agreement statistics
    matching_decisions = sum(1 for c in comparison_results if c['decisions_match'])
    total_comparisons = len(comparison_results)
    agreement_rate = matching_decisions / total_comparisons if total_comparisons > 0 else 0
    
    print(f"\n=== MODEL COMPARISON RESULTS ===")
    print(f"Total matching samples: {total_comparisons}")
    print(f"Decisions match: {matching_decisions} ({agreement_rate:.1%})")
    print(f"Decisions differ: {total_comparisons - matching_decisions} ({1-agreement_rate:.1%})")
    
    return comparison_results



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
    
    # Load processed data using sample_id-based approach
    print("Loading processed inference results...")
    pre_brexit_results = load_processed_inference_results(pre_brexit_file)
    post_brexit_results = load_processed_inference_results(post_brexit_file)
    
    # Compare models using sample_id for accurate matching
    comparison_results = compare_models(pre_brexit_results, post_brexit_results)
    
    if comparison_results is None:
        print("❌ Cannot proceed without matching samples. Check sample_id generation.")
        return
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert comparison results to DataFrame for analysis
    comparison_df = pd.DataFrame(comparison_results)
    
    # Generate plots
    print(f"\nGenerating plots in {output_dir}...")
    
    if not comparison_df.empty:
        # Decision matrix plot
        decision_matrix = pd.crosstab(
            comparison_df['pre_brexit_decision'],
            comparison_df['post_brexit_decision']
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(decision_matrix, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Decision Comparison Matrix: Pre-Brexit vs Post-Brexit')
        plt.xlabel('Post-Brexit Decisions')
        plt.ylabel('Pre-Brexit Decisions')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/decision_matrix.png')
        plt.close()
        print("✓ Decision matrix plot saved")
        
        # Overall decision distribution
        plt.figure(figsize=(12, 6))
        
        # Count decisions for each model
        pre_decisions = comparison_df['pre_brexit_decision'].value_counts()
        post_decisions = comparison_df['post_brexit_decision'].value_counts()
        
        # Create comparison plot
        decision_comparison = pd.DataFrame({
            'Pre-Brexit': pre_decisions,
            'Post-Brexit': post_decisions
        }).fillna(0)
        
        decision_comparison.plot(kind='bar', ax=plt.gca())
        plt.title('Overall Decision Distribution Comparison')
        plt.xlabel('Decision')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/overall_decision_distribution.png')
        plt.close()
        print("✓ Overall decision distribution plot saved")
    
    # Generate distribution plots by topic
    plot_decision_distributions_by_topic(
        [result for result in pre_brexit_results.values()], 
        [result for result in post_brexit_results.values()], 
        output_dir
    )
    print("✓ Decision distribution plots saved")
    
    # Analyze mismatches
    mismatches = [r for r in comparison_results if not r['decisions_match']]
    
    print(f"\n=== MISMATCH ANALYSIS ===")
    print(f"Cases with different decisions: {len(mismatches)}")
    
    if len(mismatches) > 0:
        print("\nMismatched cases by topic:")
        mismatch_topics = {}
        for mismatch in mismatches:
            topic = mismatch['topic']
            mismatch_topics[topic] = mismatch_topics.get(topic, 0) + 1
        
        for topic, count in sorted(mismatch_topics.items()):
            print(f"  {topic}: {count}")
        
        # Show a few examples
        print("\nExample mismatched cases:")
        for i, mismatch in enumerate(mismatches[:3]):
            print(f"\nCase {i+1}: {mismatch['topic']}")
            print(f"  Sample ID: {mismatch['sample_id']}")
            print(f"  Pre-Brexit: {mismatch['pre_brexit_decision']}")
            print(f"  Post-Brexit: {mismatch['post_brexit_decision']}")
            print(f"  Fields: {mismatch['fields']}")
            print(f"  Vignette: {mismatch['vignette_text'][:100]}...")
    
    print(f"\n✅ Analysis complete! Check {output_dir} for visualizations.")
    print(f"📊 Total samples analyzed: {len(comparison_results)}")
    
    # Now that we have sample_id, the matching should be completely reliable!

if __name__ == "__main__":
    main() 