#!/usr/bin/env python3
"""
Text Tokenization Evaluator for Legal BERT
==========================================

This script evaluates whether extracted text is ready for tokenization by analyzing:
1. Character-level quality (encoding, special characters, artifacts)
2. Token-level quality (vocabulary coverage, subword distribution)
3. Content structure (sentence boundaries, legal terminology)
4. BERT-specific readiness (length distribution, special token handling)
"""

import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Import tokenizer for evaluation
try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    print("Warning: transformers not available. Install with: pip install transformers")
    TOKENIZER_AVAILABLE = False

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextTokenizationEvaluator:
    """Comprehensive evaluation of text quality for BERT tokenization."""
    
    def __init__(self, data_path: str, output_dir: str = "preprocessing/tokenization_analysis"):
        """
        Initialize the evaluator.
        
        Args:
            data_path: Path to processed legal cases (CSV or Parquet)
            output_dir: Directory to save analysis results
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        if self.data_path.suffix == '.parquet':
            self.df = pd.read_parquet(data_path)
        else:
            self.df = pd.read_csv(data_path)
        
        logger.info(f"Loaded {len(self.df):,} cases for tokenization evaluation")
        
        # Initialize tokenizer if available
        self.tokenizer = None
        if TOKENIZER_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
                logger.info("✓ Legal BERT tokenizer loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load Legal BERT tokenizer: {e}")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                    logger.info("✓ Standard BERT tokenizer loaded as fallback")
                except Exception as e2:
                    logger.error(f"Could not load any BERT tokenizer: {e2}")
        
        # Analysis results storage
        self.evaluation_results = {}
        
    def analyze_character_quality(self, sample_size: int = 1000) -> Dict[str, Any]:
        """Analyze character-level text quality."""
        logger.info(f"Analyzing character quality on sample of {sample_size} texts...")
        
        # Get sample of texts
        df_with_text = self.df[self.df['decision_text_cleaned'].notna() & 
                               (self.df['decision_text_cleaned'] != '')]
        
        if len(df_with_text) == 0:
            return {'error': 'No texts found for analysis'}
        
        sample_df = df_with_text.sample(n=min(sample_size, len(df_with_text)), random_state=42)
        
        char_analysis = {
            'total_texts': len(sample_df),
            'encoding_issues': 0,
            'binary_artifacts': 0,
            'excessive_whitespace': 0,
            'non_printable_chars': 0,
            'unicode_issues': 0,
            'avg_chars_per_text': 0,
            'char_distribution': defaultdict(int),
            'problematic_texts': []
        }
        
        total_chars = 0
        
        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Character analysis"):
            text = str(row['decision_text_cleaned'])
            total_chars += len(text)
            issues = []
            
            # Check for encoding issues
            try:
                text.encode('utf-8').decode('utf-8')
            except UnicodeError:
                char_analysis['encoding_issues'] += 1
                issues.append('encoding_issues')
            
            # Check for binary artifacts
            if re.search(r'[\x00-\x08\x0e-\x1f\x7f-\x9f]', text):
                char_analysis['binary_artifacts'] += 1
                issues.append('binary_artifacts')
            
            # Check for excessive whitespace
            if re.search(r'\s{10,}', text) or text.count('\n') / max(len(text), 1) > 0.1:
                char_analysis['excessive_whitespace'] += 1
                issues.append('excessive_whitespace')
            
            # Check for non-printable characters (excluding normal whitespace)
            non_printable = sum(1 for c in text if not c.isprintable() and c not in '\n\t\r')
            if non_printable > len(text) * 0.01:  # More than 1% non-printable
                char_analysis['non_printable_chars'] += 1
                issues.append('non_printable_chars')
            
            # Check for unicode issues (unusual character combinations)
            if re.search(r'[\ufffd\ufeff]', text):  # Replacement character, BOM
                char_analysis['unicode_issues'] += 1
                issues.append('unicode_issues')
            
            # Character distribution for common chars
            for char in text:
                if char.isalpha() or char.isdigit() or char in '.,!?;:()[]{}"\'-\n ':
                    char_analysis['char_distribution'][char] += 1
            
            # Track problematic texts
            if issues:
                char_analysis['problematic_texts'].append({
                    'reference_number': row.get('reference_number', 'Unknown'),
                    'issues': issues,
                    'text_length': len(text),
                    'text_preview': text[:200] + '...' if len(text) > 200 else text
                })
        
        char_analysis['avg_chars_per_text'] = total_chars / len(sample_df) if len(sample_df) > 0 else 0
        
        # Convert counts to percentages
        for key in ['encoding_issues', 'binary_artifacts', 'excessive_whitespace', 
                   'non_printable_chars', 'unicode_issues']:
            char_analysis[f"{key}_percentage"] = (char_analysis[key] / len(sample_df)) * 100
        
        return char_analysis
    
    def analyze_tokenization_quality(self, sample_size: int = 500) -> Dict[str, Any]:
        """Analyze tokenization-specific quality metrics."""
        if not self.tokenizer:
            logger.warning("No tokenizer available - skipping tokenization analysis")
            return {'error': 'No tokenizer available'}
        
        logger.info(f"Analyzing tokenization quality on sample of {sample_size} texts...")
        
        # Get sample of texts
        df_with_text = self.df[self.df['decision_text_cleaned'].notna() & 
                               (self.df['decision_text_cleaned'] != '')]
        
        if len(df_with_text) == 0:
            return {'error': 'No texts found for analysis'}
        
        sample_df = df_with_text.sample(n=min(sample_size, len(df_with_text)), random_state=42)
        
        tokenization_analysis = {
            'total_texts': len(sample_df),
            'token_lengths': [],
            'subword_ratios': [],
            'unk_token_counts': [],
            'vocab_coverage': [],
            'avg_tokens_per_sentence': [],
            'problematic_tokenizations': []
        }
        
        vocab = set(self.tokenizer.get_vocab().keys())
        
        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Tokenization analysis"):
            text = str(row['decision_text_cleaned'])
            
            try:
                # Tokenize the text
                tokens = self.tokenizer.tokenize(text)
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                
                # Basic metrics
                tokenization_analysis['token_lengths'].append(len(tokens))
                
                # Subword analysis
                subword_count = sum(1 for token in tokens if token.startswith('##'))
                subword_ratio = subword_count / max(len(tokens), 1)
                tokenization_analysis['subword_ratios'].append(subword_ratio)
                
                # UNK token analysis
                unk_count = sum(1 for token in tokens if token == self.tokenizer.unk_token)
                tokenization_analysis['unk_token_counts'].append(unk_count)
                
                # Vocabulary coverage
                in_vocab = sum(1 for token in tokens if token in vocab)
                vocab_coverage = in_vocab / max(len(tokens), 1)
                tokenization_analysis['vocab_coverage'].append(vocab_coverage)
                
                # Sentence tokenization
                sentences = re.split(r'[.!?]+', text)
                if len(sentences) > 1:
                    avg_tokens_per_sent = len(tokens) / len(sentences)
                    tokenization_analysis['avg_tokens_per_sentence'].append(avg_tokens_per_sent)
                
                # Identify problematic cases
                issues = []
                if len(tokens) > 512:
                    issues.append('exceeds_max_length')
                if subword_ratio > 0.5:
                    issues.append('high_subword_ratio')
                if unk_count > len(tokens) * 0.05:
                    issues.append('high_unk_rate')
                if vocab_coverage < 0.8:
                    issues.append('low_vocab_coverage')
                
                if issues:
                    tokenization_analysis['problematic_tokenizations'].append({
                        'reference_number': row.get('reference_number', 'Unknown'),
                        'issues': issues,
                        'token_count': len(tokens),
                        'subword_ratio': subword_ratio,
                        'unk_count': unk_count,
                        'vocab_coverage': vocab_coverage,
                        'text_preview': text[:200] + '...' if len(text) > 200 else text
                    })
                
            except Exception as e:
                logger.warning(f"Tokenization failed for text {idx}: {e}")
                continue
        
        # Calculate summary statistics
        if tokenization_analysis['token_lengths']:
            tokenization_analysis['stats'] = {
                'avg_token_length': np.mean(tokenization_analysis['token_lengths']),
                'median_token_length': np.median(tokenization_analysis['token_lengths']),
                'max_token_length': np.max(tokenization_analysis['token_lengths']),
                'texts_exceeding_512': sum(1 for length in tokenization_analysis['token_lengths'] if length > 512),
                'avg_subword_ratio': np.mean(tokenization_analysis['subword_ratios']),
                'avg_unk_count': np.mean(tokenization_analysis['unk_token_counts']),
                'avg_vocab_coverage': np.mean(tokenization_analysis['vocab_coverage']),
            }
            
            # Percentage calculations
            total_texts = len(tokenization_analysis['token_lengths'])
            tokenization_analysis['stats']['exceeding_512_percentage'] = (
                tokenization_analysis['stats']['texts_exceeding_512'] / total_texts
            ) * 100 if total_texts > 0 else 0
        
        return tokenization_analysis
    
    def analyze_content_structure(self, sample_size: int = 500) -> Dict[str, Any]:
        """Analyze content structure for BERT processing."""
        logger.info(f"Analyzing content structure on sample of {sample_size} texts...")
        
        # Get sample of texts
        df_with_text = self.df[self.df['decision_text_cleaned'].notna() & 
                               (self.df['decision_text_cleaned'] != '')]
        
        if len(df_with_text) == 0:
            return {'error': 'No texts found for analysis'}
        
        sample_df = df_with_text.sample(n=min(sample_size, len(df_with_text)), random_state=42)
        
        structure_analysis = {
            'total_texts': len(sample_df),
            'sentence_counts': [],
            'paragraph_counts': [],
            'avg_sentence_lengths': [],
            'legal_term_densities': [],
            'punctuation_densities': [],
            'capitalization_ratios': [],
            'structure_issues': []
        }
        
        # Legal terminology for content validation
        legal_terms = [
            'appellant', 'applicant', 'tribunal', 'immigration', 'asylum', 'appeal', 
            'decision', 'determination', 'secretary of state', 'home office', 
            'paragraph', 'evidence', 'hearing', 'judge', 'court', 'law', 'act',
            'regulation', 'rule', 'application', 'refusal', 'removal', 'deportation'
        ]
        
        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Structure analysis"):
            text = str(row['decision_text_cleaned'])
            issues = []
            
            # Sentence analysis
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            structure_analysis['sentence_counts'].append(len(sentences))
            
            if sentences:
                avg_sent_length = np.mean([len(s.split()) for s in sentences])
                structure_analysis['avg_sentence_lengths'].append(avg_sent_length)
                
                # Check for very long or very short sentences
                if avg_sent_length > 50:
                    issues.append('very_long_sentences')
                elif avg_sent_length < 5:
                    issues.append('very_short_sentences')
            
            # Paragraph analysis
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            structure_analysis['paragraph_counts'].append(len(paragraphs))
            
            # Legal term density
            words = text.lower().split()
            legal_term_count = sum(1 for word in words if any(term in word for term in legal_terms))
            legal_density = legal_term_count / max(len(words), 1)
            structure_analysis['legal_term_densities'].append(legal_density)
            
            if legal_density < 0.01:  # Less than 1% legal terms
                issues.append('low_legal_term_density')
            
            # Punctuation analysis
            punct_count = sum(1 for c in text if c in '.,!?;:()[]{}"\'-')
            punct_density = punct_count / max(len(text), 1)
            structure_analysis['punctuation_densities'].append(punct_density)
            
            # Capitalization analysis
            cap_count = sum(1 for c in text if c.isupper())
            cap_ratio = cap_count / max(len([c for c in text if c.isalpha()]), 1)
            structure_analysis['capitalization_ratios'].append(cap_ratio)
            
            if cap_ratio > 0.3:  # More than 30% uppercase
                issues.append('excessive_capitalization')
            elif cap_ratio < 0.02:  # Less than 2% uppercase
                issues.append('insufficient_capitalization')
            
            # Check for structure issues
            if len(sentences) < 3:
                issues.append('too_few_sentences')
            elif len(sentences) > 1000:
                issues.append('too_many_sentences')
            
            if issues:
                structure_analysis['structure_issues'].append({
                    'reference_number': row.get('reference_number', 'Unknown'),
                    'issues': issues,
                    'sentence_count': len(sentences),
                    'legal_term_density': legal_density,
                    'capitalization_ratio': cap_ratio,
                    'text_preview': text[:200] + '...' if len(text) > 200 else text
                })
        
        # Calculate summary statistics
        structure_analysis['stats'] = {
            'avg_sentences': np.mean(structure_analysis['sentence_counts']) if structure_analysis['sentence_counts'] else 0,
            'avg_paragraphs': np.mean(structure_analysis['paragraph_counts']) if structure_analysis['paragraph_counts'] else 0,
            'avg_sentence_length': np.mean(structure_analysis['avg_sentence_lengths']) if structure_analysis['avg_sentence_lengths'] else 0,
            'avg_legal_term_density': np.mean(structure_analysis['legal_term_densities']) if structure_analysis['legal_term_densities'] else 0,
            'avg_punctuation_density': np.mean(structure_analysis['punctuation_densities']) if structure_analysis['punctuation_densities'] else 0,
            'avg_capitalization_ratio': np.mean(structure_analysis['capitalization_ratios']) if structure_analysis['capitalization_ratios'] else 0,
        }
        
        return structure_analysis
    
    def analyze_bert_readiness(self, sample_size: int = 300) -> Dict[str, Any]:
        """Analyze BERT-specific readiness metrics."""
        if not self.tokenizer:
            logger.warning("No tokenizer available - skipping BERT readiness analysis")
            return {'error': 'No tokenizer available'}
        
        logger.info(f"Analyzing BERT readiness on sample of {sample_size} texts...")
        
        # Get sample of both full text and last sections
        df_with_text = self.df[self.df['decision_text_cleaned'].notna() & 
                               (self.df['decision_text_cleaned'] != '')]
        
        if len(df_with_text) == 0:
            return {'error': 'No texts found for analysis'}
        
        sample_df = df_with_text.sample(n=min(sample_size, len(df_with_text)), random_state=42)
        
        bert_analysis = {
            'total_texts': len(sample_df),
            'full_text_analysis': {},
            'last_section_analysis': {},
            'truncation_impact': {},
            'special_token_handling': {}
        }
        
        # Analyze both full text and last sections
        for text_type in ['decision_text_cleaned', 'decision_text_last_section']:
            if text_type not in sample_df.columns:
                continue
                
            analysis_key = 'full_text_analysis' if 'cleaned' in text_type else 'last_section_analysis'
            
            token_lengths = []
            attention_patterns = []
            segment_distributions = []
            
            for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), 
                               desc=f"BERT analysis ({text_type})"):
                text = str(row[text_type]) if pd.notna(row[text_type]) else ""
                
                if not text:
                    continue
                
                try:
                    # Tokenize with BERT special tokens
                    encoding = self.tokenizer(
                        text,
                        padding=False,
                        truncation=True,
                        max_length=512,
                        return_attention_mask=True,
                        return_tensors=None
                    )
                    
                    tokens = encoding['input_ids']
                    attention_mask = encoding['attention_mask']
                    
                    token_lengths.append(len(tokens))
                    attention_patterns.append(sum(attention_mask) / len(attention_mask))
                    
                    # Analyze token distribution
                    segments = []
                    current_segment = []
                    for token_id in tokens:
                        token = self.tokenizer.convert_ids_to_tokens(token_id)
                        if token in ['[CLS]', '[SEP]', '[PAD]']:
                            if current_segment:
                                segments.append(len(current_segment))
                                current_segment = []
                        else:
                            current_segment.append(token)
                    
                    if current_segment:
                        segments.append(len(current_segment))
                    
                    segment_distributions.append(segments)
                    
                except Exception as e:
                    logger.warning(f"BERT analysis failed for text {idx}: {e}")
                    continue
            
            bert_analysis[analysis_key] = {
                'avg_token_length': np.mean(token_lengths) if token_lengths else 0,
                'max_token_length': np.max(token_lengths) if token_lengths else 0,
                'truncated_texts': sum(1 for length in token_lengths if length >= 512),
                'truncation_rate': (sum(1 for length in token_lengths if length >= 512) / 
                                  len(token_lengths)) * 100 if token_lengths else 0,
                'avg_attention_coverage': np.mean(attention_patterns) if attention_patterns else 0,
                'segment_count_avg': np.mean([len(segs) for segs in segment_distributions]) if segment_distributions else 0
            }
        
        # Compare full text vs last section
        if bert_analysis['full_text_analysis'] and bert_analysis['last_section_analysis']:
            bert_analysis['truncation_impact'] = {
                'length_reduction': (
                    bert_analysis['full_text_analysis']['avg_token_length'] - 
                    bert_analysis['last_section_analysis']['avg_token_length']
                ),
                'truncation_benefit': (
                    bert_analysis['full_text_analysis']['truncation_rate'] - 
                    bert_analysis['last_section_analysis']['truncation_rate']
                ),
                'attention_improvement': (
                    bert_analysis['last_section_analysis']['avg_attention_coverage'] - 
                    bert_analysis['full_text_analysis']['avg_attention_coverage']
                )
            }
        
        return bert_analysis
    
    def generate_preprocessing_recommendations(self) -> List[str]:
        """Generate specific recommendations based on analysis results."""
        recommendations = []
        
        # Character quality recommendations
        char_analysis = self.evaluation_results.get('character_quality', {})
        if char_analysis.get('encoding_issues_percentage', 0) > 1:
            recommendations.append(
                "🔤 HIGH PRIORITY: Fix encoding issues in text extraction - "
                f"{char_analysis['encoding_issues_percentage']:.1f}% of texts have encoding problems"
            )
        
        if char_analysis.get('binary_artifacts_percentage', 0) > 2:
            recommendations.append(
                "🔧 MEDIUM PRIORITY: Remove binary artifacts from extraction process - "
                f"{char_analysis['binary_artifacts_percentage']:.1f}% of texts contain binary data"
            )
        
        if char_analysis.get('excessive_whitespace_percentage', 0) > 5:
            recommendations.append(
                "📝 LOW PRIORITY: Normalize whitespace in preprocessing - "
                f"{char_analysis['excessive_whitespace_percentage']:.1f}% of texts have excessive whitespace"
            )
        
        # Tokenization recommendations
        token_analysis = self.evaluation_results.get('tokenization_quality', {})
        if token_analysis and token_analysis.get('stats'):
            stats = token_analysis['stats']
            
            if stats.get('exceeding_512_percentage', 0) > 20:
                recommendations.append(
                    "✂️ HIGH PRIORITY: Implement better text truncation strategy - "
                    f"{stats['exceeding_512_percentage']:.1f}% of texts exceed 512 tokens"
                )
            
            if stats.get('avg_subword_ratio', 0) > 0.4:
                recommendations.append(
                    "🔤 MEDIUM PRIORITY: Consider vocabulary-specific preprocessing - "
                    f"High subword ratio ({stats['avg_subword_ratio']:.2f}) indicates domain-specific terms"
                )
            
            if stats.get('avg_unk_count', 0) > 5:
                recommendations.append(
                    "❓ MEDIUM PRIORITY: Address unknown tokens - "
                    f"Average {stats['avg_unk_count']:.1f} UNK tokens per text"
                )
        
        # Structure recommendations
        structure_analysis = self.evaluation_results.get('content_structure', {})
        if structure_analysis and structure_analysis.get('stats'):
            stats = structure_analysis['stats']
            
            if stats.get('avg_legal_term_density', 0) < 0.02:
                recommendations.append(
                    "⚖️ HIGH PRIORITY: Verify legal content extraction - "
                    f"Low legal term density ({stats['avg_legal_term_density']:.3f}) suggests extraction issues"
                )
            
            if stats.get('avg_sentence_length', 0) > 40:
                recommendations.append(
                    "📏 LOW PRIORITY: Consider sentence segmentation - "
                    f"Very long average sentence length ({stats['avg_sentence_length']:.1f} words)"
                )
        
        # BERT-specific recommendations
        bert_analysis = self.evaluation_results.get('bert_readiness', {})
        if bert_analysis and bert_analysis.get('truncation_impact'):
            impact = bert_analysis['truncation_impact']
            
            if impact.get('truncation_benefit', 0) > 15:
                recommendations.append(
                    "✅ OPTIMIZATION: Last section extraction working well - "
                    f"{impact['truncation_benefit']:.1f}% reduction in texts requiring truncation"
                )
            
            if impact.get('attention_improvement', 0) > 0.1:
                recommendations.append(
                    "🎯 OPTIMIZATION: Attention mechanism benefits from section extraction - "
                    f"{impact['attention_improvement']:.3f} improvement in attention coverage"
                )
        
        # General recommendations if no specific issues found
        if not recommendations:
            recommendations.append("✅ GOOD NEWS: Text quality appears suitable for BERT tokenization!")
            recommendations.append("💡 SUGGESTION: Consider running final validation on larger sample")
            recommendations.append("🚀 NEXT STEP: Proceed with model training and fine-tuning")
        
        return recommendations
    
    def run_comprehensive_evaluation(self, 
                                   char_sample_size: int = 1000,
                                   token_sample_size: int = 500,
                                   structure_sample_size: int = 500,
                                   bert_sample_size: int = 300) -> Dict[str, Any]:
        """Run complete text quality evaluation."""
        logger.info("Starting comprehensive text tokenization evaluation...")
        
        # Run all analyses
        self.evaluation_results['character_quality'] = self.analyze_character_quality(char_sample_size)
        self.evaluation_results['tokenization_quality'] = self.analyze_tokenization_quality(token_sample_size)
        self.evaluation_results['content_structure'] = self.analyze_content_structure(structure_sample_size)
        self.evaluation_results['bert_readiness'] = self.analyze_bert_readiness(bert_sample_size)
        
        # Generate recommendations
        self.evaluation_results['recommendations'] = self.generate_preprocessing_recommendations()
        
        # Save results
        results_path = self.output_dir / 'tokenization_evaluation_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Deep convert the results
            json_results = json.loads(json.dumps(self.evaluation_results, default=convert_numpy))
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {results_path}")
        
        return self.evaluation_results
    
    def print_evaluation_summary(self):
        """Print a comprehensive summary of evaluation results."""
        if not self.evaluation_results:
            logger.error("No evaluation results available. Run evaluation first.")
            return
        
        print("\n" + "="*80)
        print("🔍 TEXT TOKENIZATION READINESS EVALUATION")
        print("="*80)
        
        # Character Quality Summary
        char_analysis = self.evaluation_results.get('character_quality', {})
        if char_analysis and 'total_texts' in char_analysis:
            print(f"\n📝 CHARACTER QUALITY (sample: {char_analysis['total_texts']:,} texts)")
            print(f"   ✅ Clean texts: {100 - char_analysis.get('encoding_issues_percentage', 0) - char_analysis.get('binary_artifacts_percentage', 0):.1f}%")
            print(f"   🔤 Encoding issues: {char_analysis.get('encoding_issues_percentage', 0):.1f}%")
            print(f"   ⚡ Binary artifacts: {char_analysis.get('binary_artifacts_percentage', 0):.1f}%")
            print(f"   📏 Avg chars per text: {char_analysis.get('avg_chars_per_text', 0):,.0f}")
        
        # Tokenization Quality Summary
        token_analysis = self.evaluation_results.get('tokenization_quality', {})
        if token_analysis and token_analysis.get('stats'):
            stats = token_analysis['stats']
            print(f"\n🎯 TOKENIZATION QUALITY (sample: {token_analysis['total_texts']:,} texts)")
            print(f"   📊 Avg tokens: {stats.get('avg_token_length', 0):.0f}")
            print(f"   ✂️ Exceeding 512 tokens: {stats.get('exceeding_512_percentage', 0):.1f}%")
            print(f"   🔤 Avg subword ratio: {stats.get('avg_subword_ratio', 0):.2f}")
            print(f"   ❓ Avg UNK tokens: {stats.get('avg_unk_count', 0):.1f}")
            print(f"   📚 Vocab coverage: {stats.get('avg_vocab_coverage', 0):.3f}")
        
        # Content Structure Summary  
        structure_analysis = self.evaluation_results.get('content_structure', {})
        if structure_analysis and structure_analysis.get('stats'):
            stats = structure_analysis['stats']
            print(f"\n📋 CONTENT STRUCTURE (sample: {structure_analysis['total_texts']:,} texts)")
            print(f"   📄 Avg sentences: {stats.get('avg_sentences', 0):.0f}")
            print(f"   📏 Avg sentence length: {stats.get('avg_sentence_length', 0):.1f} words")
            print(f"   ⚖️ Legal term density: {stats.get('avg_legal_term_density', 0):.3f}")
            print(f"   🔤 Capitalization ratio: {stats.get('avg_capitalization_ratio', 0):.3f}")
        
        # BERT Readiness Summary
        bert_analysis = self.evaluation_results.get('bert_readiness', {})
        if bert_analysis:
            if bert_analysis.get('last_section_analysis'):
                last_stats = bert_analysis['last_section_analysis']
                print(f"\n🤖 BERT READINESS (last sections)")
                print(f"   📊 Avg tokens: {last_stats.get('avg_token_length', 0):.0f}")
                print(f"   ✂️ Truncation rate: {last_stats.get('truncation_rate', 0):.1f}%")
                print(f"   🎯 Attention coverage: {last_stats.get('avg_attention_coverage', 0):.3f}")
            
            if bert_analysis.get('truncation_impact'):
                impact = bert_analysis['truncation_impact']
                print(f"   📈 Truncation benefit: {impact.get('truncation_benefit', 0):.1f}% reduction")
        
        # Recommendations
        recommendations = self.evaluation_results.get('recommendations', [])
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "="*80)
        
        # Overall assessment
        if self._assess_overall_quality():
            print("🎉 OVERALL ASSESSMENT: Text quality is suitable for BERT tokenization!")
            print("🚀 READY FOR: Model training and fine-tuning")
        else:
            print("⚠️  OVERALL ASSESSMENT: Text quality needs improvement before tokenization")
            print("🔧 ACTION NEEDED: Address high-priority recommendations above")
        
        print("="*80)
    
    def _assess_overall_quality(self) -> bool:
        """Assess overall text quality based on key metrics."""
        # Check critical issues
        char_analysis = self.evaluation_results.get('character_quality', {})
        encoding_issues = char_analysis.get('encoding_issues_percentage', 0)
        binary_artifacts = char_analysis.get('binary_artifacts_percentage', 0)
        
        token_analysis = self.evaluation_results.get('tokenization_quality', {})
        if token_analysis and token_analysis.get('stats'):
            exceeding_512 = token_analysis['stats'].get('exceeding_512_percentage', 0)
            unk_count = token_analysis['stats'].get('avg_unk_count', 0)
        else:
            exceeding_512 = 0
            unk_count = 0
        
        structure_analysis = self.evaluation_results.get('content_structure', {})
        if structure_analysis and structure_analysis.get('stats'):
            legal_density = structure_analysis['stats'].get('avg_legal_term_density', 1)
        else:
            legal_density = 1
        
        # Define quality thresholds
        quality_checks = [
            encoding_issues < 2,      # Less than 2% encoding issues
            binary_artifacts < 3,     # Less than 3% binary artifacts  
            exceeding_512 < 30,       # Less than 30% texts exceed 512 tokens
            unk_count < 10,           # Less than 10 UNK tokens on average
            legal_density > 0.01      # At least 1% legal terms
        ]
        
        return sum(quality_checks) >= 4  # At least 4 out of 5 checks pass

def main():
    """Main function to run text tokenization evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate text quality for BERT tokenization')
    parser.add_argument('--data_path', '-d', type=str, required=True,
                       help='Path to processed legal cases (CSV or Parquet)')
    parser.add_argument('--output_dir', '-o', type=str, 
                       default='preprocessing/tokenization_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--char_sample', type=int, default=1000,
                       help='Sample size for character analysis')
    parser.add_argument('--token_sample', type=int, default=500,
                       help='Sample size for tokenization analysis')
    parser.add_argument('--structure_sample', type=int, default=500,
                       help='Sample size for structure analysis')
    parser.add_argument('--bert_sample', type=int, default=300,
                       help='Sample size for BERT analysis')
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not Path(args.data_path).exists():
        print(f"❌ Error: Data file not found: {args.data_path}")
        return
    
    # Run evaluation
    print("🔍 Starting text tokenization evaluation...")
    print(f"📁 Data: {args.data_path}")
    print(f"📊 Output: {args.output_dir}")
    
    evaluator = TextTokenizationEvaluator(args.data_path, args.output_dir)
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(
        char_sample_size=args.char_sample,
        token_sample_size=args.token_sample,
        structure_sample_size=args.structure_sample,
        bert_sample_size=args.bert_sample
    )
    
    # Print summary
    evaluator.print_evaluation_summary()
    
    print(f"\n📄 Detailed results saved to: {args.output_dir}/tokenization_evaluation_results.json")
    
    return results

if __name__ == "__main__":
    main() 