#!/usr/bin/env python3
"""
Normative Divergence Analysis
Calculate cosine similarity between model vectors to measure behavioral divergence
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_analysis_results() -> pd.DataFrame:
    """Load the detailed analysis results"""
    results_file = "../../../outputs/grant_rate_analysis/grant_rate_analysis_by_vignette_fields.csv"
    df = pd.read_csv(results_file)
    logger.info(f"Loaded {len(df)} field comparisons")
    return df

def create_model_vectors(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Create vectors for each model with normalized scores"""
    
    # Create unique identifiers for each comparison
    df['comparison_id'] = df['topic'] + '|' + df['field_name'] + '|' + df['field_value']
    
    # Get vectors for each model
    pre_brexit_vector = df['pre_brexit_normalized'].values
    post_brexit_vector = df['post_brexit_normalized'].values
    comparison_labels = df['comparison_id'].tolist()
    
    logger.info(f"Created vectors with {len(comparison_labels)} dimensions")
    return pre_brexit_vector, post_brexit_vector, comparison_labels

def calculate_normative_divergence(pre_vector: np.ndarray, post_vector: np.ndarray) -> Dict:
    """Calculate various divergence metrics between model vectors"""
    
    # Reshape for sklearn
    pre_reshaped = pre_vector.reshape(1, -1)
    post_reshaped = post_vector.reshape(1, -1)
    
    # Cosine similarity
    cos_sim = cosine_similarity(pre_reshaped, post_reshaped)[0][0]
    cos_distance = 1 - cos_sim
    
    # Pearson correlation
    try:
        pearson_r, pearson_p = pearsonr(pre_vector, post_vector)
        if np.isnan(pearson_r):
            pearson_r, pearson_p = 0.0, 1.0
    except:
        pearson_r, pearson_p = 0.0, 1.0
    
    # Spearman correlation (rank-based)
    try:
        spearman_r, spearman_p = spearmanr(pre_vector, post_vector)
        if np.isnan(spearman_r):
            spearman_r, spearman_p = 0.0, 1.0
    except:
        spearman_r, spearman_p = 0.0, 1.0
    
    # Euclidean distance
    euclidean_dist = np.linalg.norm(pre_vector - post_vector)
    
    # Manhattan distance
    manhattan_dist = np.sum(np.abs(pre_vector - post_vector))
    
    # Mean absolute difference
    mean_abs_diff = np.mean(np.abs(pre_vector - post_vector))
    
    # Standard deviation of differences
    diff_std = np.std(pre_vector - post_vector)
    
    divergence_metrics = {
        'cosine_similarity': float(cos_sim),
        'cosine_distance': float(cos_distance),
        'pearson_correlation': float(pearson_r),
        'pearson_p_value': float(pearson_p),
        'spearman_correlation': float(spearman_r),
        'spearman_p_value': float(spearman_p),
        'euclidean_distance': float(euclidean_dist),
        'manhattan_distance': float(manhattan_dist),
        'mean_absolute_difference': float(mean_abs_diff),
        'difference_std': float(diff_std),
        'vector_length': len(pre_vector)
    }
    
    logger.info(f"Cosine similarity: {cos_sim:.4f}")
    logger.info(f"Cosine distance: {cos_distance:.4f}")
    logger.info(f"Pearson correlation: {pearson_r:.4f} (p={pearson_p:.4f})")
    
    return divergence_metrics

def analyze_by_subsets(df: pd.DataFrame) -> Dict:
    """Analyze normative divergence by different subsets"""
    
    subset_results = {}
    
    # By field type
    for field_type in ['ordinal', 'horizontal']:
        subset_df = df[df['field_type'] == field_type]
        if len(subset_df) > 1:
            pre_vec, post_vec, labels = create_model_vectors(subset_df)
            divergence = calculate_normative_divergence(pre_vec, post_vec)
            subset_results[f'{field_type}_fields'] = divergence
    
    # By statistical significance
    for significance in [True, False]:
        subset_df = df[df['statistical_significance'] == significance]
        if len(subset_df) > 1:
            pre_vec, post_vec, labels = create_model_vectors(subset_df)
            divergence = calculate_normative_divergence(pre_vec, post_vec)
            subset_results[f'significant_{significance}'] = divergence
    
    # By topic (for topics with multiple comparisons)
    topic_counts = df['topic'].value_counts()
    multi_comparison_topics = topic_counts[topic_counts > 1].index
    
    for topic in multi_comparison_topics:
        subset_df = df[df['topic'] == topic]
        if len(subset_df) > 1:
            pre_vec, post_vec, labels = create_model_vectors(subset_df)
            divergence = calculate_normative_divergence(pre_vec, post_vec)
            subset_results[f'topic_{topic}'] = divergence
    
    return subset_results

def test_statistical_robustness(df: pd.DataFrame, n_bootstrap: int = 1000) -> Dict:
    """Test statistical robustness using bootstrap resampling"""
    
    pre_vector, post_vector, labels = create_model_vectors(df)
    
    # Bootstrap sampling
    bootstrap_cos_sim = []
    bootstrap_pearson = []
    
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(len(pre_vector), size=len(pre_vector), replace=True)
        pre_sample = pre_vector[indices]
        post_sample = post_vector[indices]
        
        # Calculate metrics
        cos_sim = cosine_similarity(pre_sample.reshape(1, -1), post_sample.reshape(1, -1))[0][0]
        pearson_r, _ = pearsonr(pre_sample, post_sample)
        
        bootstrap_cos_sim.append(cos_sim)
        bootstrap_pearson.append(pearson_r)
    
    # Calculate confidence intervals
    cos_sim_ci = np.percentile(bootstrap_cos_sim, [2.5, 97.5])
    pearson_ci = np.percentile(bootstrap_pearson, [2.5, 97.5])
    
    robustness_results = {
        'bootstrap_iterations': n_bootstrap,
        'cosine_similarity_mean': float(np.mean(bootstrap_cos_sim)),
        'cosine_similarity_std': float(np.std(bootstrap_cos_sim)),
        'cosine_similarity_ci': [float(cos_sim_ci[0]), float(cos_sim_ci[1])],
        'pearson_correlation_mean': float(np.mean(bootstrap_pearson)),
        'pearson_correlation_std': float(np.std(bootstrap_pearson)),
        'pearson_correlation_ci': [float(pearson_ci[0]), float(pearson_ci[1])]
    }
    
    logger.info(f"Bootstrap results:")
    logger.info(f"  Cosine similarity: {np.mean(bootstrap_cos_sim):.4f} ± {np.std(bootstrap_cos_sim):.4f}")
    logger.info(f"  95% CI: [{cos_sim_ci[0]:.4f}, {cos_sim_ci[1]:.4f}]")
    
    return robustness_results

def weight_by_sample_size(df: pd.DataFrame) -> Dict:
    """Calculate weighted divergence metrics using sample sizes"""
    
    # Use minimum sample size as weight
    df['weight'] = np.minimum(df['pre_brexit_sample_size'], df['post_brexit_sample_size'])
    
    # Weighted average of absolute differences
    weighted_mean_diff = np.average(
        np.abs(df['pre_brexit_normalized'] - df['post_brexit_normalized']),
        weights=df['weight']
    )
    
    # Create weighted vectors (repeat values based on sample size)
    weighted_pre = []
    weighted_post = []
    
    for _, row in df.iterrows():
        weight = int(row['weight'] / 10)  # Scale down weights
        weighted_pre.extend([row['pre_brexit_normalized']] * weight)
        weighted_post.extend([row['post_brexit_normalized']] * weight)
    
    weighted_pre = np.array(weighted_pre)
    weighted_post = np.array(weighted_post)
    
    # Calculate metrics on weighted vectors
    cos_sim = cosine_similarity(weighted_pre.reshape(1, -1), weighted_post.reshape(1, -1))[0][0]
    pearson_r, pearson_p = pearsonr(weighted_pre, weighted_post)
    
    weighted_results = {
        'weighted_mean_difference': float(weighted_mean_diff),
        'weighted_cosine_similarity': float(cos_sim),
        'weighted_pearson_correlation': float(pearson_r),
        'weighted_pearson_p_value': float(pearson_p),
        'weighted_vector_length': len(weighted_pre)
    }
    
    logger.info(f"Weighted analysis:")
    logger.info(f"  Cosine similarity: {cos_sim:.4f}")
    logger.info(f"  Mean weighted difference: {weighted_mean_diff:.4f}")
    
    return weighted_results

def explore_topic_divergence(df: pd.DataFrame, subset_results: Dict) -> Dict:
    """Explore which topics show highest/lowest divergence"""
    
    # Extract topic-specific divergence scores
    topic_divergences = []
    
    for key, metrics in subset_results.items():
        if key.startswith('topic_'):
            topic_name = key.replace('topic_', '')
            topic_divergences.append({
                'topic': topic_name,
                'cosine_similarity': metrics['cosine_similarity'],
                'cosine_distance': metrics['cosine_distance'],
                'pearson_correlation': metrics['pearson_correlation'],
                'vector_length': metrics['vector_length']
            })
    
    # Sort by divergence (cosine distance)
    topic_divergences.sort(key=lambda x: x['cosine_distance'], reverse=True)
    
    # Calculate field-level divergence for each topic
    topic_field_analysis = {}
    
    for topic in df['topic'].unique():
        topic_data = df[df['topic'] == topic]
        if len(topic_data) > 1:
            # Calculate mean absolute difference for this topic
            mean_abs_diff = topic_data['cross_model_difference'].abs().mean()
            max_abs_diff = topic_data['cross_model_difference'].abs().max()
            
            # Find most divergent field in this topic
            most_divergent_field = topic_data.loc[topic_data['cross_model_difference'].abs().idxmax()]
            
            topic_field_analysis[topic] = {
                'mean_absolute_difference': float(mean_abs_diff),
                'max_absolute_difference': float(max_abs_diff),
                'most_divergent_field': most_divergent_field['field_value'],
                'most_divergent_difference': float(most_divergent_field['cross_model_difference']),
                'num_comparisons': len(topic_data),
                'significant_comparisons': len(topic_data[topic_data['statistical_significance']])
            }
    
    return {
        'topic_divergence_ranking': topic_divergences,
        'topic_field_analysis': topic_field_analysis
    }

def create_visualization(df: pd.DataFrame, output_dir: Path, topic_exploration: Dict):
    """Create comprehensive visualizations of normative divergence"""
    
    # 1. Overall scatter plot with enhanced styling
    plt.figure(figsize=(14, 10))
    
    # Color by field type and size by sample size
    colors = {'ordinal': 'blue', 'horizontal': 'red'}
    for field_type in ['ordinal', 'horizontal']:
        subset = df[df['field_type'] == field_type]
        plt.scatter(subset['pre_brexit_normalized'], subset['post_brexit_normalized'], 
                   c=colors[field_type], label=field_type, alpha=0.7,
                   s=subset['pre_brexit_sample_size']/10)  # Size by sample size
    
    # Add diagonal line (perfect correlation)
    min_val = min(df['pre_brexit_normalized'].min(), df['post_brexit_normalized'].min())
    max_val = max(df['pre_brexit_normalized'].max(), df['post_brexit_normalized'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect correlation')
    
    # Highlight most divergent points
    most_divergent = df.loc[df['cross_model_difference'].abs().idxmax()]
    plt.scatter(most_divergent['pre_brexit_normalized'], most_divergent['post_brexit_normalized'], 
               c='black', s=200, marker='X', label='Most divergent')
    
    plt.xlabel('Pre-Brexit Normalized Score')
    plt.ylabel('Post-Brexit Normalized Score')
    plt.title('Model Normative Alignment\n(Bubble size = Sample size)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'normative_alignment_scatter.png', dpi=300)
    plt.close()
    
    # 2. Topic divergence ranking
    topic_rankings = topic_exploration['topic_divergence_ranking']
    if topic_rankings:
        plt.figure(figsize=(12, 8))
        topics = [t['topic'][:30] + '...' if len(t['topic']) > 30 else t['topic'] for t in topic_rankings]
        distances = [t['cosine_distance'] for t in topic_rankings]
        
        colors = ['red' if d > 0.5 else 'orange' if d > 0.3 else 'green' for d in distances]
        
        plt.barh(topics, distances, color=colors, alpha=0.7)
        plt.xlabel('Cosine Distance (Normative Divergence)')
        plt.title('Topic-Level Normative Divergence\n(Higher = More Divergent)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'topic_divergence_ranking.png', dpi=300)
        plt.close()
    
    # 3. Field-level divergence heatmap
    plt.figure(figsize=(16, 12))
    
    # Create pivot table for heatmap
    pivot_data = df.pivot_table(
        values='cross_model_difference', 
        index='topic', 
        columns='field_value', 
        aggfunc='mean'
    )
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Cross-Model Difference'})
    plt.title('Field-Level Normative Divergence Heatmap\n(Red = Pre-Brexit favors, Blue = Post-Brexit favors)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'field_divergence_heatmap.png', dpi=300)
    plt.close()
    
    # 4. Topic-wise detailed analysis
    topics_with_multiple = df['topic'].value_counts()
    topics_with_multiple = topics_with_multiple[topics_with_multiple > 1].index
    
    n_topics = len(topics_with_multiple)
    n_cols = 3
    n_rows = (n_topics + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 5*n_rows))
    
    for i, topic in enumerate(topics_with_multiple):
        topic_data = df[df['topic'] == topic]
        
        plt.subplot(n_rows, n_cols, i+1)
        
        # Color by statistical significance
        colors = ['red' if sig else 'blue' for sig in topic_data['statistical_significance']]
        
        plt.scatter(topic_data['pre_brexit_normalized'], topic_data['post_brexit_normalized'], 
                   c=colors, alpha=0.7)
        plt.axline((0, 0), slope=1, color='black', linestyle='--', alpha=0.5)
        
        plt.xlabel('Pre-Brexit')
        plt.ylabel('Post-Brexit')
        plt.title(f'{topic}\n(Red=Significant, Blue=Not significant)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'topic_wise_detailed_analysis.png', dpi=300)
    plt.close()
    
    # 5. Divergence distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['cross_model_difference'], bins=20, alpha=0.7, color='blue')
    plt.axvline(0, color='red', linestyle='--', label='No difference')
    plt.xlabel('Cross-Model Difference')
    plt.ylabel('Frequency')
    plt.title('Distribution of Cross-Model Differences')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    significant = df[df['statistical_significance']]
    non_significant = df[~df['statistical_significance']]
    
    plt.hist(significant['cross_model_difference'], bins=15, alpha=0.7, 
             color='red', label='Significant')
    plt.hist(non_significant['cross_model_difference'], bins=15, alpha=0.7, 
             color='blue', label='Non-significant')
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Cross-Model Difference')
    plt.ylabel('Frequency')
    plt.title('Difference Distribution by Significance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'divergence_distribution.png', dpi=300)
    plt.close()

def main():
    """Main analysis function"""
    logger.info("Starting Normative Divergence Analysis")
    
    # Load results
    df = load_analysis_results()
    
    # Create output directory
    output_dir = Path("../../../outputs/normative_divergence")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Overall divergence analysis
    logger.info("Calculating overall normative divergence...")
    pre_vector, post_vector, labels = create_model_vectors(df)
    overall_divergence = calculate_normative_divergence(pre_vector, post_vector)
    
    # Subset analysis
    logger.info("Analyzing divergence by subsets...")
    subset_results = analyze_by_subsets(df)
    
    # Statistical robustness testing
    logger.info("Testing statistical robustness...")
    robustness_results = test_statistical_robustness(df)
    
    # Weighted analysis
    logger.info("Calculating weighted divergence metrics...")
    weighted_results = weight_by_sample_size(df)
    
    # Topic exploration
    logger.info("Exploring topic-specific divergence patterns...")
    topic_exploration = explore_topic_divergence(df, subset_results)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    create_visualization(df, output_dir, topic_exploration)
    
    # Compile results
    results = {
        'overall_divergence': overall_divergence,
        'subset_analysis': subset_results,
        'robustness_testing': robustness_results,
        'weighted_analysis': weighted_results,
        'topic_exploration': topic_exploration,
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Save results
    with open(output_dir / 'normative_divergence_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary report with topic exploration
    most_divergent_topic = topic_exploration['topic_divergence_ranking'][0] if topic_exploration['topic_divergence_ranking'] else None
    least_divergent_topic = topic_exploration['topic_divergence_ranking'][-1] if topic_exploration['topic_divergence_ranking'] else None
    
    most_divergent_name = most_divergent_topic['topic'] if most_divergent_topic else 'N/A'
    most_divergent_distance = f"{most_divergent_topic['cosine_distance']:.4f}" if most_divergent_topic else 'N/A'
    least_divergent_name = least_divergent_topic['topic'] if least_divergent_topic else 'N/A'
    least_divergent_distance = f"{least_divergent_topic['cosine_distance']:.4f}" if least_divergent_topic else 'N/A'
    
    alignment_level = "High" if overall_divergence['cosine_similarity'] > 0.8 else "Medium" if overall_divergence['cosine_similarity'] > 0.6 else "Low"
    
    summary_report = f"""
# Normative Divergence Analysis Summary

## Overall Divergence
- **Cosine Similarity**: {overall_divergence['cosine_similarity']:.4f}
- **Cosine Distance**: {overall_divergence['cosine_distance']:.4f}
- **Pearson Correlation**: {overall_divergence['pearson_correlation']:.4f}
- **Mean Absolute Difference**: {overall_divergence['mean_absolute_difference']:.4f}

## Statistical Robustness (Bootstrap)
- **Cosine Similarity CI**: [{robustness_results['cosine_similarity_ci'][0]:.4f}, {robustness_results['cosine_similarity_ci'][1]:.4f}]
- **Standard Deviation**: {robustness_results['cosine_similarity_std']:.4f}

## Weighted Analysis
- **Weighted Cosine Similarity**: {weighted_results['weighted_cosine_similarity']:.4f}
- **Weighted Mean Difference**: {weighted_results['weighted_mean_difference']:.4f}

## Topic-Specific Insights
- **Most Divergent Topic**: {most_divergent_name} (Distance: {most_divergent_distance})
- **Least Divergent Topic**: {least_divergent_name} (Distance: {least_divergent_distance})
- **Topics Analyzed**: {len(topic_exploration['topic_divergence_ranking'])}

## Interpretation
- **Normative Divergence Score**: {overall_divergence['cosine_distance']:.4f}
- **Behavioral Alignment**: {alignment_level}
"""
    
    with open(output_dir / 'normative_divergence_summary.md', 'w') as f:
        f.write(summary_report)
    
    logger.info("Analysis complete!")
    logger.info(f"Results saved to {output_dir}")
    
    # Print key findings
    print("\n" + "="*50)
    print("NORMATIVE DIVERGENCE ANALYSIS RESULTS")
    print("="*50)
    print(f"Cosine Similarity: {overall_divergence['cosine_similarity']:.4f}")
    print(f"Cosine Distance: {overall_divergence['cosine_distance']:.4f}")
    print(f"Pearson Correlation: {overall_divergence['pearson_correlation']:.4f}")
    print(f"Statistical Significance: {'Yes' if overall_divergence['pearson_p_value'] < 0.05 else 'No'}")
    print(f"Bootstrap 95% CI: [{robustness_results['cosine_similarity_ci'][0]:.4f}, {robustness_results['cosine_similarity_ci'][1]:.4f}]")

if __name__ == "__main__":
    main() 