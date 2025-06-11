#!/usr/bin/env python3
"""
Comprehensive Quality Assessment and Monitoring for LLaMA Preprocessing
======================================================================

Provides detailed quality metrics, validation, and monitoring for the
preprocessing pipeline to ensure training-ready data quality.
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PreprocessingQualityMonitor:
    """Comprehensive quality assessment for LLaMA preprocessing pipeline."""
    
    def __init__(self, 
                 processed_data_dir: str = "preprocessing/outputs/llama_training_ready",
                 output_dir: str = "preprocessing/outputs/quality_assessment"):
        """Initialize the quality monitor."""
        self.processed_data_dir = Path(processed_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Quality thresholds
        self.quality_thresholds = {
            'min_chunk_tokens': 512,
            'max_chunk_tokens': 2048,
            'target_chunk_tokens': 1024,
            'min_chunk_chars': 1000,
            'max_artifact_ratio': 0.05,  # 5% artifacts max
            'min_sentence_avg_length': 10,  # words per sentence
            'max_sentence_avg_length': 50,
            'min_legal_term_density': 0.01,  # 1% legal terms min
            'min_paragraph_coherence': 0.7,  # coherence score
        }
        
        # Legal domain vocabulary for assessment
        self.legal_terms = {
            'tribunal', 'appellant', 'respondent', 'determination', 'asylum',
            'immigration', 'refugee', 'persecution', 'convention', 'appeal',
            'adjudicator', 'hearing', 'evidence', 'credibility', 'protection',
            'removal', 'deportation', 'human rights', 'article', 'convention',
            'exclusion', 'internal flight', 'relocation', 'country guidance',
            'paragraph', 'submission', 'representative', 'secretary of state'
        }
        
    def load_processed_datasets(self) -> Dict[str, List[Dict]]:
        """Load all processed datasets for quality assessment."""
        logger.info("Loading processed datasets for quality assessment...")
        
        datasets = {}
        for dataset_dir in self.processed_data_dir.iterdir():
            if dataset_dir.is_dir():
                jsonl_file = dataset_dir / f"{dataset_dir.name}_chunks.jsonl"
                if jsonl_file.exists():
                    chunks = []
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                chunks.append(json.loads(line.strip()))
                            except json.JSONDecodeError:
                                continue
                    datasets[dataset_dir.name] = chunks
                    logger.info(f"Loaded {len(chunks):,} chunks from {dataset_dir.name}")
        
        return datasets
    
    def assess_text_quality(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Assess text quality metrics for a dataset."""
        logger.info(f"Assessing text quality for {len(chunks):,} chunks...")
        
        quality_metrics = {
            'token_distribution': {},
            'character_distribution': {},
            'sentence_metrics': {},
            'paragraph_metrics': {},
            'legal_domain_metrics': {},
            'artifact_metrics': {},
            'coherence_metrics': {}
        }
        
        token_counts = []
        char_counts = []
        sentence_counts = []
        paragraph_counts = []
        legal_term_counts = []
        artifact_counts = []
        
        for chunk in tqdm(chunks, desc="Analyzing text quality"):
            text = chunk.get('text', '')
            if not text:
                continue
                
            # Token and character counts
            token_count = chunk.get('token_count', 0)
            char_count = len(text)
            token_counts.append(token_count)
            char_counts.append(char_count)
            
            # Sentence analysis
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            sentence_counts.append(len(sentences))
            
            # Paragraph analysis
            paragraphs = text.split('\n\n')
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            paragraph_counts.append(len(paragraphs))
            
            # Legal domain analysis
            text_lower = text.lower()
            legal_terms_found = sum(1 for term in self.legal_terms if term in text_lower)
            legal_term_counts.append(legal_terms_found / len(self.legal_terms))
            
            # Artifact detection
            artifacts = self._detect_artifacts(text)
            artifact_counts.append(len(artifacts))
        
        # Calculate distributions
        quality_metrics['token_distribution'] = {
            'mean': np.mean(token_counts),
            'median': np.median(token_counts),
            'std': np.std(token_counts),
            'min': np.min(token_counts),
            'max': np.max(token_counts),
            'p25': np.percentile(token_counts, 25),
            'p75': np.percentile(token_counts, 75),
            'within_target': sum(1 for t in token_counts if 512 <= t <= 2048) / len(token_counts)
        }
        
        quality_metrics['character_distribution'] = {
            'mean': np.mean(char_counts),
            'median': np.median(char_counts),
            'std': np.std(char_counts),
            'min': np.min(char_counts),
            'max': np.max(char_counts)
        }
        
        quality_metrics['sentence_metrics'] = {
            'avg_sentences_per_chunk': np.mean(sentence_counts),
            'avg_chars_per_sentence': np.mean(char_counts) / np.mean(sentence_counts) if np.mean(sentence_counts) > 0 else 0
        }
        
        quality_metrics['paragraph_metrics'] = {
            'avg_paragraphs_per_chunk': np.mean(paragraph_counts),
            'avg_chars_per_paragraph': np.mean(char_counts) / np.mean(paragraph_counts) if np.mean(paragraph_counts) > 0 else 0
        }
        
        quality_metrics['legal_domain_metrics'] = {
            'avg_legal_term_density': np.mean(legal_term_counts),
            'chunks_with_legal_terms': sum(1 for count in legal_term_counts if count > 0) / len(legal_term_counts)
        }
        
        quality_metrics['artifact_metrics'] = {
            'avg_artifacts_per_chunk': np.mean(artifact_counts),
            'chunks_with_artifacts': sum(1 for count in artifact_counts if count > 0) / len(artifact_counts)
        }
        
        return quality_metrics
    
    def _detect_artifacts(self, text: str) -> List[str]:
        """Detect potential preprocessing artifacts in text."""
        artifacts = []
        
        artifact_patterns = [
            (r'\n\s*\d+\s*\n', 'isolated_numbers'),
            (r'Page \d+', 'page_numbers'),
            (r'\[?\d{4}\]?\s*UKUT\s*\d+', 'case_citations_in_text'),
            (r'Date of hearing:', 'hearing_date_artifact'),
            (r'Before:', 'judge_listing_artifact'),
            (r'Crown Copyright', 'copyright_notice'),
            (r'\n\s*[A-Z]\.\s*[A-Z]\.\s*[A-Z]\.', 'judge_initials'),
            (r'(\w)\1{4,}', 'repeated_characters'),  # 5+ repeated chars
            (r'\s{5,}', 'excessive_whitespace'),
        ]
        
        for pattern, artifact_type in artifact_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                artifacts.extend([artifact_type] * len(matches))
        
        return artifacts
    
    def assess_semantic_coherence(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Assess semantic coherence and legal reasoning flow."""
        logger.info("Assessing semantic coherence...")
        
        coherence_metrics = {
            'paragraph_transitions': {},
            'legal_argument_flow': {},
            'citation_coherence': {},
            'temporal_consistency': {}
        }
        
        transition_scores = []
        citation_patterns = []
        
        for chunk in tqdm(chunks[:500], desc="Analyzing coherence"):  # Sample for performance
            text = chunk.get('text', '')
            if not text:
                continue
            
            # Paragraph transition analysis
            paragraphs = text.split('\n\n')
            if len(paragraphs) > 1:
                transitions = self._analyze_paragraph_transitions(paragraphs)
                transition_scores.extend(transitions)
            
            # Legal citation analysis
            citations = re.findall(r'\[\d{4}\]\s+\w+\s+\d+', text)
            citation_patterns.extend(citations)
        
        coherence_metrics['paragraph_transitions'] = {
            'avg_transition_score': np.mean(transition_scores) if transition_scores else 0,
            'coherent_transitions': sum(1 for score in transition_scores if score > 0.5) / len(transition_scores) if transition_scores else 0
        }
        
        coherence_metrics['citation_coherence'] = {
            'unique_citations': len(set(citation_patterns)),
            'total_citations': len(citation_patterns),
            'citation_density': len(citation_patterns) / len(chunks) if chunks else 0
        }
        
        return coherence_metrics
    
    def _analyze_paragraph_transitions(self, paragraphs: List[str]) -> List[float]:
        """Analyze transition quality between paragraphs."""
        transition_scores = []
        
        transition_words = {
            'however', 'furthermore', 'moreover', 'therefore', 'consequently',
            'nevertheless', 'accordingly', 'subsequently', 'additionally',
            'in conclusion', 'in summary', 'as a result', 'on the other hand'
        }
        
        for i in range(len(paragraphs) - 1):
            current_para = paragraphs[i].lower()
            next_para = paragraphs[i + 1].lower()
            
            # Simple coherence scoring based on transition words and topic continuity
            score = 0.0
            
            # Check for transition words
            for word in transition_words:
                if word in next_para[:100]:  # First 100 chars of next paragraph
                    score += 0.3
                    break
            
            # Check for topic continuity (shared important words)
            current_words = set(re.findall(r'\b\w{4,}\b', current_para))
            next_words = set(re.findall(r'\b\w{4,}\b', next_para[:200]))  # First 200 chars
            
            if current_words and next_words:
                overlap = len(current_words.intersection(next_words))
                score += min(overlap / len(current_words), 0.7)
            
            transition_scores.append(min(score, 1.0))
        
        return transition_scores
    
    def assess_temporal_consistency(self, datasets: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Assess consistency across temporal datasets."""
        logger.info("Assessing temporal consistency...")
        
        temporal_metrics = {}
        
        for dataset_name, chunks in datasets.items():
            if not chunks:
                continue
                
            # Calculate dataset-specific metrics
            token_dist = [chunk.get('token_count', 0) for chunk in chunks]
            char_dist = [len(chunk.get('text', '')) for chunk in chunks]
            
            temporal_metrics[dataset_name] = {
                'chunk_count': len(chunks),
                'avg_token_count': np.mean(token_dist),
                'token_std': np.std(token_dist),
                'avg_char_count': np.mean(char_dist),
                'char_std': np.std(char_dist),
                'unique_cases': len(set(chunk.get('case_id', '') for chunk in chunks))
            }
        
        # Calculate consistency across datasets
        if len(temporal_metrics) > 1:
            token_means = [metrics['avg_token_count'] for metrics in temporal_metrics.values()]
            char_means = [metrics['avg_char_count'] for metrics in temporal_metrics.values()]
            
            temporal_metrics['cross_dataset_consistency'] = {
                'token_count_cv': np.std(token_means) / np.mean(token_means) if np.mean(token_means) > 0 else 0,
                'char_count_cv': np.std(char_means) / np.mean(char_means) if np.mean(char_means) > 0 else 0,
                'balanced_datasets': max(len(chunks) for chunks in datasets.values()) / min(len(chunks) for chunks in datasets.values() if chunks)
            }
        
        return temporal_metrics
    
    def generate_quality_report(self, datasets: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Generate comprehensive quality assessment report."""
        logger.info("Generating comprehensive quality report...")
        
        quality_report = {
            'assessment_timestamp': datetime.now().isoformat(),
            'total_datasets': len(datasets),
            'total_chunks': sum(len(chunks) for chunks in datasets.values()),
            'dataset_quality': {},
            'temporal_consistency': {},
            'overall_quality_score': 0.0,
            'recommendations': []
        }
        
        # Assess each dataset
        all_chunks = []
        for dataset_name, chunks in datasets.items():
            if chunks:
                dataset_quality = self.assess_text_quality(chunks)
                semantic_quality = self.assess_semantic_coherence(chunks)
                
                # Combine metrics
                dataset_quality.update(semantic_quality)
                quality_report['dataset_quality'][dataset_name] = dataset_quality
                all_chunks.extend(chunks)
        
        # Assess temporal consistency
        quality_report['temporal_consistency'] = self.assess_temporal_consistency(datasets)
        
        # Calculate overall quality score
        quality_score = self._calculate_overall_quality_score(quality_report)
        quality_report['overall_quality_score'] = quality_score
        
        # Generate recommendations
        recommendations = self._generate_recommendations(quality_report)
        quality_report['recommendations'] = recommendations
        
        return quality_report
    
    def _calculate_overall_quality_score(self, quality_report: Dict[str, Any]) -> float:
        """Calculate overall quality score based on multiple metrics."""
        scores = []
        
        for dataset_name, metrics in quality_report['dataset_quality'].items():
            dataset_score = 0.0
            
            # Token distribution score (40% weight)
            token_dist = metrics.get('token_distribution', {})
            within_target = token_dist.get('within_target', 0)
            dataset_score += within_target * 0.4
            
            # Legal domain score (25% weight)
            legal_metrics = metrics.get('legal_domain_metrics', {})
            legal_density = legal_metrics.get('avg_legal_term_density', 0)
            legal_coverage = legal_metrics.get('chunks_with_legal_terms', 0)
            dataset_score += (legal_density * 10 + legal_coverage) * 0.125  # Normalize and weight
            
            # Artifact score (20% weight) - lower is better
            artifact_metrics = metrics.get('artifact_metrics', {})
            artifact_ratio = artifact_metrics.get('avg_artifacts_per_chunk', 0)
            artifact_score = max(0, 1 - artifact_ratio / 5)  # Normalize
            dataset_score += artifact_score * 0.2
            
            # Coherence score (15% weight)
            coherence_metrics = metrics.get('paragraph_transitions', {})
            coherence_score = coherence_metrics.get('coherent_transitions', 0)
            dataset_score += coherence_score * 0.15
            
            scores.append(min(dataset_score, 1.0))
        
        return np.mean(scores) if scores else 0.0
    
    def _generate_recommendations(self, quality_report: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on quality assessment."""
        recommendations = []
        
        overall_score = quality_report.get('overall_quality_score', 0)
        
        if overall_score < 0.7:
            recommendations.append("🚨 Overall quality score is below 70%. Consider additional cleaning steps.")
        
        # Check token distribution
        for dataset_name, metrics in quality_report['dataset_quality'].items():
            token_dist = metrics.get('token_distribution', {})
            within_target = token_dist.get('within_target', 0)
            
            if within_target < 0.8:
                recommendations.append(f"⚠️  {dataset_name}: Only {within_target:.1%} chunks within target token range. Review chunking strategy.")
            
            # Check legal domain coverage
            legal_metrics = metrics.get('legal_domain_metrics', {})
            legal_coverage = legal_metrics.get('chunks_with_legal_terms', 0)
            
            if legal_coverage < 0.8:
                recommendations.append(f"⚠️  {dataset_name}: Only {legal_coverage:.1%} chunks contain legal terms. Verify domain specificity.")
            
            # Check artifacts
            artifact_metrics = metrics.get('artifact_metrics', {})
            chunks_with_artifacts = artifact_metrics.get('chunks_with_artifacts', 0)
            
            if chunks_with_artifacts > 0.1:
                recommendations.append(f"⚠️  {dataset_name}: {chunks_with_artifacts:.1%} chunks contain artifacts. Improve cleaning.")
        
        # Check temporal consistency
        temporal_metrics = quality_report.get('temporal_consistency', {})
        cross_dataset = temporal_metrics.get('cross_dataset_consistency', {})
        
        if cross_dataset.get('token_count_cv', 0) > 0.3:
            recommendations.append("⚠️  High variation in token counts across temporal datasets. Consider consistent chunking.")
        
        if cross_dataset.get('balanced_datasets', 1) > 2:
            recommendations.append("⚠️  Significant imbalance between temporal datasets. Consider sampling strategies.")
        
        if not recommendations:
            recommendations.append("✅ Quality assessment passed all checks. Data appears ready for training.")
        
        return recommendations
    
    def create_quality_visualizations(self, quality_report: Dict[str, Any]):
        """Create visualizations for quality assessment."""
        logger.info("Creating quality visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LLaMA Preprocessing Quality Assessment', fontsize=16)
        
        # 1. Token distribution across datasets
        ax1 = axes[0, 0]
        datasets = []
        token_means = []
        token_stds = []
        
        for dataset_name, metrics in quality_report['dataset_quality'].items():
            token_dist = metrics.get('token_distribution', {})
            datasets.append(dataset_name.replace('_', '\n'))
            token_means.append(token_dist.get('mean', 0))
            token_stds.append(token_dist.get('std', 0))
        
        x_pos = np.arange(len(datasets))
        ax1.bar(x_pos, token_means, yerr=token_stds, capsize=5, alpha=0.7)
        ax1.axhline(y=1024, color='r', linestyle='--', label='Target (1024)')
        ax1.axhline(y=512, color='orange', linestyle='--', label='Min (512)')
        ax1.axhline(y=2048, color='orange', linestyle='--', label='Max (2048)')
        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Average Token Count')
        ax1.set_title('Token Distribution by Dataset')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(datasets)
        ax1.legend()
        
        # 2. Quality scores by dataset
        ax2 = axes[0, 1]
        quality_scores = []
        for dataset_name, metrics in quality_report['dataset_quality'].items():
            # Calculate simplified quality score for visualization
            token_score = metrics.get('token_distribution', {}).get('within_target', 0)
            legal_score = metrics.get('legal_domain_metrics', {}).get('chunks_with_legal_terms', 0)
            artifact_score = 1 - metrics.get('artifact_metrics', {}).get('chunks_with_artifacts', 0)
            
            overall = (token_score + legal_score + artifact_score) / 3
            quality_scores.append(overall)
        
        colors = ['green' if score > 0.8 else 'orange' if score > 0.6 else 'red' for score in quality_scores]
        ax2.bar(x_pos, quality_scores, color=colors, alpha=0.7)
        ax2.axhline(y=0.8, color='green', linestyle='--', label='Good (0.8)')
        ax2.axhline(y=0.6, color='orange', linestyle='--', label='Fair (0.6)')
        ax2.set_xlabel('Dataset')
        ax2.set_ylabel('Quality Score')
        ax2.set_title('Quality Score by Dataset')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(datasets)
        ax2.set_ylim(0, 1)
        ax2.legend()
        
        # 3. Chunk count distribution
        ax3 = axes[1, 0]
        chunk_counts = [len(datasets) for datasets in quality_report['dataset_quality'].keys()]
        temporal_data = quality_report.get('temporal_consistency', {})
        
        dataset_names = list(temporal_data.keys())
        chunk_counts = [temporal_data[name].get('chunk_count', 0) for name in dataset_names if name != 'cross_dataset_consistency']
        
        if chunk_counts:
            ax3.pie(chunk_counts, labels=[name.replace('_', '\n') for name in dataset_names if name != 'cross_dataset_consistency'], 
                   autopct='%1.1f%%', startangle=90)
            ax3.set_title('Chunk Distribution Across Datasets')
        
        # 4. Overall quality summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Summary text
        overall_score = quality_report.get('overall_quality_score', 0)
        total_chunks = quality_report.get('total_chunks', 0)
        total_datasets = quality_report.get('total_datasets', 0)
        
        summary_text = f"""
        QUALITY SUMMARY
        
        Overall Score: {overall_score:.2f}/1.00
        
        Total Datasets: {total_datasets}
        Total Chunks: {total_chunks:,}
        
        Status: {'✅ PASS' if overall_score > 0.7 else '⚠️ REVIEW' if overall_score > 0.5 else '❌ FAIL'}
        
        Key Metrics:
        • Token Range Compliance
        • Legal Domain Coverage
        • Artifact Detection
        • Semantic Coherence
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.output_dir / 'quality_assessment_dashboard.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Quality visualization saved to {viz_path}")
    
    def run_quality_assessment(self) -> Dict[str, Any]:
        """Run complete quality assessment pipeline."""
        logger.info("🔍 Starting comprehensive quality assessment...")
        
        # Load processed datasets
        datasets = self.load_processed_datasets()
        
        if not datasets:
            logger.error("No processed datasets found!")
            return {}
        
        # Generate quality report
        quality_report = self.generate_quality_report(datasets)
        
        # Create visualizations
        try:
            self.create_quality_visualizations(quality_report)
        except Exception as e:
            logger.warning(f"Could not create visualizations: {e}")
        
        # Save detailed report
        report_path = self.output_dir / 'quality_assessment_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        self._print_quality_summary(quality_report)
        
        logger.info(f"Quality assessment complete. Report saved to {report_path}")
        
        return quality_report
    
    def _print_quality_summary(self, quality_report: Dict[str, Any]):
        """Print a formatted quality summary."""
        print("\n" + "="*80)
        print("🔍 PREPROCESSING QUALITY ASSESSMENT SUMMARY")
        print("="*80)
        
        overall_score = quality_report.get('overall_quality_score', 0)
        total_chunks = quality_report.get('total_chunks', 0)
        
        print(f"📊 Overall Quality Score: {overall_score:.2f}/1.00")
        print(f"📄 Total Chunks Processed: {total_chunks:,}")
        
        status_emoji = "✅" if overall_score > 0.7 else "⚠️" if overall_score > 0.5 else "❌"
        status_text = "PASS" if overall_score > 0.7 else "NEEDS REVIEW" if overall_score > 0.5 else "FAIL"
        print(f"🎯 Status: {status_emoji} {status_text}")
        
        print(f"\n📋 RECOMMENDATIONS:")
        recommendations = quality_report.get('recommendations', [])
        for rec in recommendations:
            print(f"   {rec}")
        
        print(f"\n📁 Detailed report: {self.output_dir}/quality_assessment_report.json")
        print("="*80)

def main():
    """Main function for standalone quality assessment."""
    import argparse
    
    parser = argparse.ArgumentParser(description='LLaMA Preprocessing Quality Assessment')
    parser.add_argument('--processed_data_dir', '-d', type=str,
                       default='preprocessing/outputs/llama_training_ready',
                       help='Directory containing processed datasets')
    parser.add_argument('--output_dir', '-o', type=str,
                       default='preprocessing/outputs/quality_assessment',
                       help='Output directory for quality assessment')
    
    args = parser.parse_args()
    
    monitor = PreprocessingQualityMonitor(args.processed_data_dir, args.output_dir)
    quality_report = monitor.run_quality_assessment()
    
    return quality_report

if __name__ == "__main__":
    main() 