#!/usr/bin/env python3
"""
LLaMA 3 Training Text Evaluator
==============================

Evaluates legal case text quality specifically for unsupervised training of LLaMA 3 13B base model.
Focuses on:
1. Full text quality assessment (not truncated sections)
2. Training data volume and efficiency
3. LLaMA 3 tokenization patterns
4. Legal domain text quality for language modeling
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

class LlamaTrainingTextEvaluator:
    """Evaluate text quality for LLaMA 3 unsupervised training."""
    
    def __init__(self, data_path: str, output_dir: str = "preprocessing/llama_training_analysis"):
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
        
        logger.info(f"Loaded {len(self.df):,} cases for LLaMA training evaluation")
        
        # Initialize LLaMA tokenizer
        self.tokenizer = None
        if TOKENIZER_AVAILABLE:
            try:
                # Try LLaMA 3 tokenizer first
                self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
                logger.info("✓ LLaMA 3 tokenizer loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load LLaMA 3 tokenizer: {e}")
                try:
                    # Fallback to similar tokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
                    logger.info("✓ Fallback tokenizer loaded")
                except Exception as e2:
                    logger.error(f"Could not load any suitable tokenizer: {e2}")
        
        # Analysis results storage
        self.evaluation_results = {}
        
    def analyze_training_volume(self) -> Dict[str, Any]:
        """Analyze the total volume of training data available."""
        logger.info("Analyzing training data volume...")
        
        # Get texts with content
        df_with_text = self.df[self.df['decision_text_cleaned'].notna() & 
                               (self.df['decision_text_cleaned'] != '')]
        
        if len(df_with_text) == 0:
            return {'error': 'No texts found for analysis'}
        
        volume_analysis = {
            'total_cases': len(self.df),
            'cases_with_text': len(df_with_text),
            'coverage_percentage': (len(df_with_text) / len(self.df)) * 100,
            'total_characters': df_with_text['decision_text_length'].sum(),
            'total_words': df_with_text['decision_text_word_count'].sum(),
            'avg_chars_per_case': df_with_text['decision_text_length'].mean(),
            'avg_words_per_case': df_with_text['decision_text_word_count'].mean(),
            'text_length_distribution': {
                'min': df_with_text['decision_text_length'].min(),
                'q25': df_with_text['decision_text_length'].quantile(0.25),
                'median': df_with_text['decision_text_length'].median(),
                'q75': df_with_text['decision_text_length'].quantile(0.75),
                'max': df_with_text['decision_text_length'].max(),
                'std': df_with_text['decision_text_length'].std()
            }
        }
        
        # Estimate training tokens if tokenizer available
        if self.tokenizer:
            # Sample to estimate token count
            sample_size = min(1000, len(df_with_text))
            sample_df = df_with_text.sample(n=sample_size, random_state=42)
            
            total_tokens_sample = 0
            for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Estimating tokens"):
                text = str(row['decision_text_cleaned'])
                try:
                    tokens = self.tokenizer.tokenize(text)
                    total_tokens_sample += len(tokens)
                except:
                    # Fallback estimation: ~4 chars per token
                    total_tokens_sample += len(text) // 4
            
            avg_tokens_per_case = total_tokens_sample / len(sample_df)
            estimated_total_tokens = avg_tokens_per_case * len(df_with_text)
            
            volume_analysis.update({
                'estimated_total_tokens': int(estimated_total_tokens),
                'avg_tokens_per_case': avg_tokens_per_case,
                'tokens_per_billion': estimated_total_tokens / 1_000_000_000,
                'training_efficiency': {
                    'tokens_per_mb': estimated_total_tokens / (volume_analysis['total_characters'] / 1_000_000),
                    'chars_per_token': volume_analysis['total_characters'] / estimated_total_tokens
                }
            })
        
        return volume_analysis
    
    def analyze_text_quality_for_training(self, sample_size: int = 1000) -> Dict[str, Any]:
        """Analyze text quality specifically for language model training."""
        logger.info(f"Analyzing text quality for training on sample of {sample_size} texts...")
        
        # Get sample of texts
        df_with_text = self.df[self.df['decision_text_cleaned'].notna() & 
                               (self.df['decision_text_cleaned'] != '')]
        
        if len(df_with_text) == 0:
            return {'error': 'No texts found for analysis'}
        
        sample_df = df_with_text.sample(n=min(sample_size, len(df_with_text)), random_state=42)
        
        quality_analysis = {
            'sample_size': len(sample_df),
            'text_contamination': {
                'binary_artifacts': 0,
                'extraction_errors': 0,
                'encoding_issues': 0,
                'corrupted_texts': 0
            },
            'content_quality': {
                'legal_content_density': [],
                'coherent_texts': 0,
                'fragmented_texts': 0,
                'repetitive_texts': 0
            },
            'language_quality': {
                'proper_english': 0,
                'mixed_language': 0,
                'poor_ocr': 0
            },
            'problematic_cases': []
        }
        
        # Legal terminology for content validation
        legal_terms = [
            'tribunal', 'immigration', 'asylum', 'appeal', 'decision', 'determination',
            'secretary of state', 'home office', 'appellant', 'applicant', 'evidence',
            'hearing', 'judge', 'court', 'law', 'act', 'regulation', 'rule',
            'application', 'refusal', 'removal', 'deportation', 'visa', 'entry clearance'
        ]
        
        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Quality analysis"):
            text = str(row['decision_text_cleaned'])
            issues = []
            
            # Check for contamination
            # Binary artifacts
            if re.search(r'[\x00-\x08\x0e-\x1f\x7f-\x9f]', text):
                quality_analysis['text_contamination']['binary_artifacts'] += 1
                issues.append('binary_artifacts')
            
            # Extraction errors (common patterns)
            extraction_patterns = [
                r'PK\x03\x04', r'Content_Types\.xml', r'_rels/\.rels',
                r'word/document\.xml', r'[A-Za-z0-9+/]{100,}={0,2}',  # base64-like
                r'(?:%[0-9A-Fa-f]{2}){3,}'  # URL encoded (3+ hex pairs)
            ]
            if any(re.search(pattern, text) for pattern in extraction_patterns):
                quality_analysis['text_contamination']['extraction_errors'] += 1
                issues.append('extraction_errors')
            
            # Encoding issues
            try:
                text.encode('utf-8').decode('utf-8')
                if '�' in text or '\ufffd' in text:  # Replacement characters
                    quality_analysis['text_contamination']['encoding_issues'] += 1
                    issues.append('encoding_issues')
            except UnicodeError:
                quality_analysis['text_contamination']['encoding_issues'] += 1
                issues.append('encoding_issues')
            
            # Content quality assessment
            words = text.lower().split()
            
            # Legal content density
            legal_term_count = sum(1 for word in words if any(term in word for term in legal_terms))
            legal_density = legal_term_count / max(len(words), 1)
            quality_analysis['content_quality']['legal_content_density'].append(legal_density)
            
            # Coherence check (basic)
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) >= 5 and legal_density > 0.02:
                quality_analysis['content_quality']['coherent_texts'] += 1
            elif len(sentences) < 3 or legal_density < 0.005:
                quality_analysis['content_quality']['fragmented_texts'] += 1
                issues.append('fragmented_content')
            
            # Repetition check (simple)
            if len(set(words)) < len(words) * 0.3:  # Less than 30% unique words
                quality_analysis['content_quality']['repetitive_texts'] += 1
                issues.append('repetitive_content')
            
            # Language quality
            english_chars = sum(1 for c in text if c.isascii())
            if len(text) > 0:
                english_ratio = english_chars / len(text)
                if english_ratio > 0.95:
                    quality_analysis['language_quality']['proper_english'] += 1
                elif english_ratio > 0.8:
                    quality_analysis['language_quality']['mixed_language'] += 1
                    issues.append('mixed_language')
                else:
                    quality_analysis['language_quality']['poor_ocr'] += 1
                    issues.append('poor_ocr')
            
            # OCR quality indicators
            ocr_error_patterns = [
                r'[Il1]{3,}',  # Multiple I/l/1 confusion
                r'rn{2,}',     # m -> rn confusion
                r'\b[a-z]\s+[a-z]\s+[a-z]\b',  # Spaced single chars
                r'[^\w\s]{5,}'  # Long special char sequences
            ]
            ocr_errors = sum(1 for pattern in ocr_error_patterns if re.search(pattern, text))
            if ocr_errors > 3:
                quality_analysis['language_quality']['poor_ocr'] += 1
                issues.append('ocr_errors')
            
            # Track problematic cases
            if issues:
                quality_analysis['problematic_cases'].append({
                    'reference_number': row.get('reference_number', 'Unknown'),
                    'issues': issues,
                    'text_length': len(text),
                    'legal_density': legal_density,
                    'english_ratio': english_ratio if 'english_ratio' in locals() else 1.0,
                    'text_preview': text[:300] + '...' if len(text) > 300 else text
                })
        
        # Calculate percentages and summaries
        total_sample = len(sample_df)
        
        # Fix: Create list of keys before iterating to avoid "dictionary changed size during iteration"
        contamination_keys = list(quality_analysis['text_contamination'].keys())
        for category in contamination_keys:
            if not category.endswith('_percentage'):  # Only process non-percentage keys
                quality_analysis['text_contamination'][f'{category}_percentage'] = (
                    quality_analysis['text_contamination'][category] / total_sample
                ) * 100
        
        content_keys = list(quality_analysis['content_quality'].keys())
        for category in content_keys:
            if isinstance(quality_analysis['content_quality'][category], int):
                quality_analysis['content_quality'][f'{category}_percentage'] = (
                    quality_analysis['content_quality'][category] / total_sample
                ) * 100
        
        language_keys = list(quality_analysis['language_quality'].keys())
        for category in language_keys:
            if not category.endswith('_percentage'):  # Only process non-percentage keys
                quality_analysis['language_quality'][f'{category}_percentage'] = (
                    quality_analysis['language_quality'][category] / total_sample
                ) * 100
        
        # Legal content density statistics
        if quality_analysis['content_quality']['legal_content_density']:
            densities = quality_analysis['content_quality']['legal_content_density']
            quality_analysis['content_quality']['legal_density_stats'] = {
                'mean': np.mean(densities),
                'median': np.median(densities),
                'std': np.std(densities),
                'min': np.min(densities),
                'max': np.max(densities)
            }
        
        return quality_analysis
    
    def analyze_llama_tokenization(self, sample_size: int = 500) -> Dict[str, Any]:
        """Analyze tokenization patterns specific to LLaMA 3."""
        if not self.tokenizer:
            logger.warning("No tokenizer available - skipping tokenization analysis")
            return {'error': 'No tokenizer available'}
        
        logger.info(f"Analyzing LLaMA tokenization on sample of {sample_size} texts...")
        
        # Get sample of texts
        df_with_text = self.df[self.df['decision_text_cleaned'].notna() & 
                               (self.df['decision_text_cleaned'] != '')]
        
        if len(df_with_text) == 0:
            return {'error': 'No texts found for analysis'}
        
        sample_df = df_with_text.sample(n=min(sample_size, len(df_with_text)), random_state=42)
        
        tokenization_analysis = {
            'sample_size': len(sample_df),
            'token_statistics': {
                'lengths': [],
                'avg_tokens_per_char': [],
                'compression_ratios': []
            },
            'context_utilization': {
                'under_1k_tokens': 0,
                'under_4k_tokens': 0,
                'under_8k_tokens': 0,
                'exceeding_8k_tokens': 0,
                'optimal_length_cases': 0
            },
            'vocabulary_analysis': {
                'unique_tokens': set(),
                'legal_specific_tokens': [],
                'oov_rate': []
            },
            'training_efficiency': {
                'usable_cases': 0,
                'total_training_tokens': 0,
                'avg_efficiency': 0
            }
        }
        
        vocab = set(self.tokenizer.get_vocab().keys()) if hasattr(self.tokenizer, 'get_vocab') else set()
        
        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="LLaMA tokenization analysis"):
            text = str(row['decision_text_cleaned'])
            
            try:
                # Tokenize the text
                tokens = self.tokenizer.tokenize(text)
                token_count = len(tokens)
                
                # Basic statistics
                tokenization_analysis['token_statistics']['lengths'].append(token_count)
                
                char_count = len(text)
                if char_count > 0:
                    tokens_per_char = token_count / char_count
                    tokenization_analysis['token_statistics']['avg_tokens_per_char'].append(tokens_per_char)
                    
                    # Compression ratio (higher = more efficient)
                    compression_ratio = char_count / token_count if token_count > 0 else 0
                    tokenization_analysis['token_statistics']['compression_ratios'].append(compression_ratio)
                
                # Context utilization
                if token_count < 1000:
                    tokenization_analysis['context_utilization']['under_1k_tokens'] += 1
                elif token_count < 4000:
                    tokenization_analysis['context_utilization']['under_4k_tokens'] += 1
                elif token_count < 8192:
                    tokenization_analysis['context_utilization']['under_8k_tokens'] += 1
                    tokenization_analysis['context_utilization']['optimal_length_cases'] += 1
                else:
                    tokenization_analysis['context_utilization']['exceeding_8k_tokens'] += 1
                
                # Vocabulary analysis
                if vocab:
                    unique_tokens_in_text = set(tokens)
                    tokenization_analysis['vocabulary_analysis']['unique_tokens'].update(unique_tokens_in_text)
                    
                    # OOV rate
                    oov_count = sum(1 for token in tokens if token not in vocab)
                    oov_rate = oov_count / max(token_count, 1)
                    tokenization_analysis['vocabulary_analysis']['oov_rate'].append(oov_rate)
                
                # Training efficiency
                if token_count <= 8192:  # Usable for training
                    tokenization_analysis['training_efficiency']['usable_cases'] += 1
                    tokenization_analysis['training_efficiency']['total_training_tokens'] += token_count
                
            except Exception as e:
                logger.warning(f"Tokenization failed for text {idx}: {e}")
                continue
        
        # Calculate summary statistics
        if tokenization_analysis['token_statistics']['lengths']:
            lengths = tokenization_analysis['token_statistics']['lengths']
            tokenization_analysis['token_statistics']['summary'] = {
                'mean': np.mean(lengths),
                'median': np.median(lengths),
                'std': np.std(lengths),
                'min': np.min(lengths),
                'max': np.max(lengths),
                'q75': np.percentile(lengths, 75),
                'q95': np.percentile(lengths, 95)
            }
        
        # Context utilization percentages
        total_sample = len(sample_df)
        
        # Fix: Create list of keys before iterating to avoid "dictionary changed size during iteration" 
        context_keys = list(tokenization_analysis['context_utilization'].keys())
        for key in context_keys:
            if isinstance(tokenization_analysis['context_utilization'][key], int):
                percentage_key = f"{key}_percentage"
                tokenization_analysis['context_utilization'][percentage_key] = (
                    tokenization_analysis['context_utilization'][key] / total_sample
                ) * 100
        
        # Training efficiency metrics
        usable_cases = tokenization_analysis['training_efficiency']['usable_cases']
        if usable_cases > 0:
            tokenization_analysis['training_efficiency']['avg_tokens_per_usable_case'] = (
                tokenization_analysis['training_efficiency']['total_training_tokens'] / usable_cases
            )
            tokenization_analysis['training_efficiency']['usability_rate'] = (
                usable_cases / total_sample
            ) * 100
        
        # Vocabulary statistics
        if tokenization_analysis['vocabulary_analysis']['oov_rate']:
            oov_rates = tokenization_analysis['vocabulary_analysis']['oov_rate']
            tokenization_analysis['vocabulary_analysis']['oov_stats'] = {
                'mean': np.mean(oov_rates),
                'median': np.median(oov_rates),
                'max': np.max(oov_rates)
            }
        
        tokenization_analysis['vocabulary_analysis']['unique_tokens_count'] = len(
            tokenization_analysis['vocabulary_analysis']['unique_tokens']
        )
        
        return tokenization_analysis
    
    def generate_training_recommendations(self) -> List[str]:
        """Generate specific recommendations for LLaMA training."""
        recommendations = []
        
        # Volume analysis recommendations
        volume_analysis = self.evaluation_results.get('training_volume', {})
        if volume_analysis:
            total_tokens = volume_analysis.get('estimated_total_tokens', 0)
            if total_tokens < 10_000_000:  # Less than 10M tokens
                recommendations.append(
                    f"📊 CONCERN: Limited training data - only {total_tokens/1_000_000:.1f}M tokens. "
                    "Consider expanding dataset or supplementing with additional legal texts."
                )
            elif total_tokens > 1_000_000_000:  # More than 1B tokens
                recommendations.append(
                    f"🎉 EXCELLENT: Large training corpus - {total_tokens/1_000_000_000:.1f}B tokens. "
                    "Sufficient for effective continued pretraining."
                )
            
            coverage = volume_analysis.get('coverage_percentage', 0)
            if coverage < 95:
                recommendations.append(
                    f"⚠️ TEXT COVERAGE: Only {coverage:.1f}% of cases have extracted text. "
                    "Consider re-running extraction on failed cases."
                )
        
        # Quality analysis recommendations
        quality_analysis = self.evaluation_results.get('text_quality', {})
        if quality_analysis:
            contamination = quality_analysis.get('text_contamination', {})
            
            if contamination.get('binary_artifacts_percentage', 0) > 5:
                recommendations.append(
                    f"🔧 HIGH PRIORITY: Clean binary artifacts - "
                    f"{contamination['binary_artifacts_percentage']:.1f}% of texts contaminated. "
                    "This will degrade training quality."
                )
            
            if contamination.get('extraction_errors_percentage', 0) > 10:
                recommendations.append(
                    f"🔧 HIGH PRIORITY: Fix extraction errors - "
                    f"{contamination['extraction_errors_percentage']:.1f}% of texts have extraction artifacts."
                )
            
            content_quality = quality_analysis.get('content_quality', {})
            if content_quality.get('fragmented_texts_percentage', 0) > 15:
                recommendations.append(
                    f"📝 MEDIUM PRIORITY: Many fragmented texts - "
                    f"{content_quality['fragmented_texts_percentage']:.1f}% may reduce training efficiency."
                )
            
            legal_density = content_quality.get('legal_density_stats', {})
            if legal_density.get('mean', 0) < 0.03:
                recommendations.append(
                    f"⚖️ CONCERN: Low legal content density ({legal_density.get('mean', 0):.3f}). "
                    "Verify that extraction captured legal decisions properly."
                )
        
        # Tokenization recommendations
        token_analysis = self.evaluation_results.get('llama_tokenization', {})
        if token_analysis:
            context_util = token_analysis.get('context_utilization', {})
            
            exceeding_8k = context_util.get('exceeding_8k_tokens_percentage', 0)
            if exceeding_8k > 20:
                recommendations.append(
                    f"✂️ CONSIDER: {exceeding_8k:.1f}% of texts exceed 8K tokens. "
                    "May need truncation strategy for very long documents."
                )
            
            optimal_length = context_util.get('optimal_length_cases_percentage', 0)
            if optimal_length > 60:
                recommendations.append(
                    f"✅ EXCELLENT: {optimal_length:.1f}% of texts are optimal length (1K-8K tokens). "
                    "Good fit for LLaMA context window."
                )
            
            efficiency = token_analysis.get('training_efficiency', {})
            usability = efficiency.get('usability_rate', 0)
            if usability < 90:
                recommendations.append(
                    f"⚠️ EFFICIENCY: Only {usability:.1f}% of texts usable for training. "
                    "Consider text cleaning to improve usability."
                )
        
        # General recommendations
        if not recommendations:
            recommendations.extend([
                "✅ GOOD NEWS: Text quality appears suitable for LLaMA training!",
                "🚀 READY FOR: Continued pretraining on legal domain",
                "💡 SUGGESTION: Start with small batch training to validate setup"
            ])
        
        return recommendations
    
    def run_comprehensive_evaluation(self,
                                   quality_sample_size: int = 1000,
                                   token_sample_size: int = 500) -> Dict[str, Any]:
        """Run complete evaluation for LLaMA training."""
        logger.info("Starting comprehensive LLaMA training text evaluation...")
        
        # Run all analyses
        self.evaluation_results['training_volume'] = self.analyze_training_volume()
        self.evaluation_results['text_quality'] = self.analyze_text_quality_for_training(quality_sample_size)
        self.evaluation_results['llama_tokenization'] = self.analyze_llama_tokenization(token_sample_size)
        
        # Generate recommendations
        self.evaluation_results['recommendations'] = self.generate_training_recommendations()
        
        # Save results
        results_path = self.output_dir / 'llama_training_evaluation_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, set):
                    return list(obj)
                return obj
            
            # Deep convert the results
            json_results = json.loads(json.dumps(self.evaluation_results, default=convert_numpy))
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {results_path}")
        
        return self.evaluation_results
    
    def print_evaluation_summary(self):
        """Print a comprehensive summary for LLaMA training."""
        if not self.evaluation_results:
            logger.error("No evaluation results available. Run evaluation first.")
            return
        
        print("\n" + "="*80)
        print("🦙 LLAMA 3 TRAINING TEXT EVALUATION")
        print("="*80)
        
        # Training Volume Summary
        volume = self.evaluation_results.get('training_volume', {})
        if volume:
            print(f"\n📊 TRAINING DATA VOLUME")
            print(f"   📄 Total cases: {volume.get('total_cases', 0):,}")
            print(f"   ✅ With text: {volume.get('cases_with_text', 0):,} ({volume.get('coverage_percentage', 0):.1f}%)")
            print(f"   📝 Total characters: {volume.get('total_characters', 0):,}")
            print(f"   🔤 Estimated tokens: {volume.get('estimated_total_tokens', 0):,}")
            if volume.get('tokens_per_billion'):
                print(f"   🎯 Training scale: {volume['tokens_per_billion']:.2f}B tokens")
        
        # Text Quality Summary
        quality = self.evaluation_results.get('text_quality', {})
        if quality:
            contamination = quality.get('text_contamination', {})
            content = quality.get('content_quality', {})
            language = quality.get('language_quality', {})
            
            print(f"\n🔍 TEXT QUALITY (sample: {quality.get('sample_size', 0):,} texts)")
            print(f"   🧹 Clean texts: {100 - contamination.get('binary_artifacts_percentage', 0) - contamination.get('extraction_errors_percentage', 0):.1f}%")
            print(f"   ⚡ Binary artifacts: {contamination.get('binary_artifacts_percentage', 0):.1f}%")
            print(f"   🔧 Extraction errors: {contamination.get('extraction_errors_percentage', 0):.1f}%")
            print(f"   📚 Coherent content: {content.get('coherent_texts_percentage', 0):.1f}%")
            print(f"   🌍 Proper English: {language.get('proper_english_percentage', 0):.1f}%")
            
            legal_stats = content.get('legal_density_stats', {})
            if legal_stats:
                print(f"   ⚖️ Legal content density: {legal_stats.get('mean', 0):.3f} (avg)")
        
        # Tokenization Summary
        tokenization = self.evaluation_results.get('llama_tokenization', {})
        if tokenization:
            stats = tokenization.get('token_statistics', {}).get('summary', {})
            context = tokenization.get('context_utilization', {})
            efficiency = tokenization.get('training_efficiency', {})
            
            print(f"\n🦙 LLAMA TOKENIZATION (sample: {tokenization.get('sample_size', 0):,} texts)")
            print(f"   📊 Avg tokens per text: {stats.get('mean', 0):.0f}")
            print(f"   📏 Median length: {stats.get('median', 0):.0f} tokens")
            print(f"   🎯 Optimal length (1K-8K): {context.get('optimal_length_cases_percentage', 0):.1f}%")
            print(f"   ✂️ Exceeding 8K tokens: {context.get('exceeding_8k_tokens_percentage', 0):.1f}%")
            print(f"   ✅ Usable for training: {efficiency.get('usability_rate', 0):.1f}%")
            
            if efficiency.get('total_training_tokens'):
                print(f"   🔥 Training tokens available: {efficiency['total_training_tokens']:,}")
        
        # Recommendations
        recommendations = self.evaluation_results.get('recommendations', [])
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "="*80)
        
        # Overall training readiness assessment
        if self._assess_training_readiness():
            print("🎉 TRAINING READINESS: Excellent! Ready for LLaMA continued pretraining")
            print("🚀 NEXT STEPS: Set up training pipeline and start with small batch validation")
        else:
            print("⚠️  TRAINING READINESS: Issues detected - address recommendations first")
            print("🔧 ACTION NEEDED: Clean data quality issues before training")
        
        print("="*80)
    
    def _assess_training_readiness(self) -> bool:
        """Assess overall readiness for LLaMA training."""
        # Check critical metrics
        volume = self.evaluation_results.get('training_volume', {})
        quality = self.evaluation_results.get('text_quality', {})
        tokenization = self.evaluation_results.get('llama_tokenization', {})
        
        # Volume checks
        tokens = volume.get('estimated_total_tokens', 0)
        coverage = volume.get('coverage_percentage', 0)
        
        # Quality checks
        contamination = quality.get('text_contamination', {})
        binary_artifacts = contamination.get('binary_artifacts_percentage', 0)
        extraction_errors = contamination.get('extraction_errors_percentage', 0)
        
        content = quality.get('content_quality', {})
        coherent_texts = content.get('coherent_texts_percentage', 0)
        
        # Tokenization checks
        efficiency = tokenization.get('training_efficiency', {})
        usability = efficiency.get('usability_rate', 0)
        
        # Define readiness criteria
        readiness_checks = [
            tokens > 1_000_000,        # At least 1M tokens
            coverage > 90,             # At least 90% text coverage
            binary_artifacts < 15,     # Less than 15% binary artifacts
            extraction_errors < 20,    # Less than 20% extraction errors
            coherent_texts > 70,       # At least 70% coherent texts
            usability > 80             # At least 80% usable for training
        ]
        
        return sum(readiness_checks) >= 5  # At least 5 out of 6 checks pass

def main():
    """Main function to run LLaMA training text evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate text quality for LLaMA 3 training')
    parser.add_argument('--data_path', '-d', type=str, required=True,
                       help='Path to processed legal cases (CSV or Parquet)')
    parser.add_argument('--output_dir', '-o', type=str, 
                       default='preprocessing/llama_training_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--quality_sample', type=int, default=1000,
                       help='Sample size for quality analysis')
    parser.add_argument('--token_sample', type=int, default=500,
                       help='Sample size for tokenization analysis')
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not Path(args.data_path).exists():
        print(f"❌ Error: Data file not found: {args.data_path}")
        return
    
    # Run evaluation
    print("🦙 Starting LLaMA 3 training text evaluation...")
    print(f"📁 Data: {args.data_path}")
    print(f"📊 Output: {args.output_dir}")
    
    evaluator = LlamaTrainingTextEvaluator(args.data_path, args.output_dir)
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(
        quality_sample_size=args.quality_sample,
        token_sample_size=args.token_sample
    )
    
    # Print summary
    evaluator.print_evaluation_summary()
    
    print(f"\n📄 Detailed results saved to: {args.output_dir}/llama_training_evaluation_results.json")
    
    return results

if __name__ == "__main__":
    main() 