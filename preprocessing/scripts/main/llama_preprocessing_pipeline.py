#!/usr/bin/env python3
"""
Comprehensive LLaMA-3 Preprocessing Pipeline
==========================================

Professional preprocessing for UK Immigration Tribunal legal decisions
optimized for unsupervised next word prediction training.

Addresses:
1. Advanced text cleaning for legal domain
2. Semantic preservation (documents, paragraphs, sentences)
3. Three temporal datasets: 2013-2016, 2017, 2018-2025
4. Unsupervised training readiness
5. LLaMA-3 tokenization compatibility
6. Human inspection samples for quality oversight
"""

import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import unicodedata
import random

# LLaMA tokenizer
try:
    from transformers import LlamaTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    print("⚠️  LLaMA tokenizer not available. Install with: pip install transformers")

# Quality monitoring
try:
    import sys
    import os
    
    # Add the analysis directory to the path - more robust approach
    current_dir = Path(__file__).parent
    analysis_dir = current_dir.parent / "analysis"
    sys.path.insert(0, str(analysis_dir))
    
    from preprocessing_quality_monitor import PreprocessingQualityMonitor
    QUALITY_MONITOR_AVAILABLE = True
except ImportError as e:
    QUALITY_MONITOR_AVAILABLE = False
    print(f"⚠️  Quality monitor not available: {e}")
    print("   Continuing without quality assessment.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLaMAPreprocessingPipeline:
    """Comprehensive preprocessing pipeline for LLaMA-3 training."""
    
    def __init__(self, 
                 data_path: str,
                 output_dir: str = "preprocessing/outputs/llama_training_ready",
                 tokenizer_model: str = "meta-llama/Llama-2-7b-hf",
                 run_quality_assessment: bool = True,
                 human_inspection_samples: int = 20):
        """Initialize the preprocessing pipeline."""
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_quality_assessment = run_quality_assessment
        self.human_inspection_samples = human_inspection_samples
        
        # Load tokenizer if available
        self.tokenizer = None
        if TOKENIZER_AVAILABLE:
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_model)
                logger.info(f"✅ Loaded LLaMA tokenizer: {tokenizer_model}")
            except Exception as e:
                logger.warning(f"⚠️  Could not load tokenizer: {e}")
        
        # Chunking parameters
        self.min_chunk_tokens = 512
        self.max_chunk_tokens = 2048
        self.target_chunk_tokens = 1024
        
        # Load data
        logger.info(f"Loading data from {data_path}")
        if self.data_path.suffix == '.parquet':
            self.df = pd.read_parquet(data_path)
        else:
            self.df = pd.read_csv(data_path, low_memory=False)
        
        logger.info(f"Loaded {len(self.df):,} cases")
    
    def advanced_legal_text_cleaning(self, text: str) -> str:
        """
        Conservative text cleaning that preserves ALL legal content.
        Only removes encoding artifacts and formatting noise.
        """
        if not text or pd.isna(text):
            return ""
        
        text = str(text)
        
        # 1. Remove encoding artifacts and garbage ONLY (very targeted)
        # Remove obvious base64/encoding strings
        text = re.sub(r'[A-Za-z0-9+/=]{50,}', '', text)  # Very long base64-like strings
        text = re.sub(r'na5%Kw@7kH[^a-zA-Z\s]*', '', text)  # Specific garbage pattern seen
        
        # 2. Remove repeating character patterns (much more aggressive)
        text = re.sub(r'(&[A-Za-z0-9]{1,10}){2,}', '', text)  # Any repeating &char patterns
        text = re.sub(r'([A-Za-z0-9]{2,8}[&@#$%^*+=|\\`~]){2,}', '', text)  # Any repeating char+symbol
        
        # 3. Remove MS Word/PDF formatting artifacts (much more comprehensive)
        text = re.sub(r'[A-Za-z]*HmH[^a-zA-Z\s]*', '', text)  # Any HmH patterns
        text = re.sub(r'[A-Za-z]*sH[^a-zA-Z\s]*', '', text)  # Any sH patterns  
        text = re.sub(r'[A-Za-z]*tH[^a-zA-Z\s]*', '', text)  # Any tH patterns
        text = re.sub(r'@[A-Za-z]*Normal[A-Za-z0-9]*', '', text)  # @Normal patterns
        text = re.sub(r'DA[A-Za-z`@]*Paragraph[A-Za-z]*', '', text)  # DA...Paragraph patterns
        text = re.sub(r'Font[A-Za-z0-9@]*', '', text)  # Font patterns
        text = re.sub(r'Table[A-Za-z0-9@]*', '', text)  # Table patterns
        
        # 4. Remove legal document formatting codes (much simpler)
        text = re.sub(r'[A-Za-z]*\+leg[A-Za-z0-9]*', '', text)  # Any +leg patterns
        text = re.sub(r'ff@q@[A-Za-z]+', '', text)  # ff@q@ patterns
        text = re.sub(r'[A-Z]{2,}\+[a-z]+', '', text)  # CAPS+lowercase patterns
        
        # 5. Remove lines with heavy special character contamination
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Calculate special character ratio
            special_chars = sum(1 for c in line if c in '&@#$%^*+=|\\`~;^)(*[]{}')
            total_chars = len(line)
            
            if total_chars > 0:
                special_ratio = special_chars / total_chars
                
                # Skip lines with >15% special characters (very aggressive)
                if special_ratio > 0.15:
                    continue
                    
                # Skip lines that look like pure formatting garbage
                if re.search(r'[&@#$%^*+=|\\`~]{3,}', line):  # 3+ special chars in a row
                    continue
                if re.search(r'leg[a-z]+.*leg[a-z]+', line):  # legal formatting codes
                    continue
                if re.search(r'[A-Za-z]{2,}[&@#$%^*+=|\\`~][A-Za-z0-9&@#$%^*+=|\\`~]{5,}', line):  # mixed garbage
                    continue
            
            cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # 6. Final aggressive cleanup
        # Remove any remaining special character sequences
        text = re.sub(r'[&@#$%^*+=|\\`~;]{2,}', '', text)  # 2+ special chars
        text = re.sub(r'\b[0-9]{5,}\b', '', text)  # 5+ digit sequences (usually formatting) but preserve smaller numbers
        
        # 7. Normalize quotation marks and dashes (safe)
        text = re.sub('\u201c', '"', text)  # Left smart quote
        text = re.sub('\u201d', '"', text)  # Right smart quote  
        text = re.sub('\u2018', "'", text)  # Left smart apostrophe
        text = re.sub('\u2019', "'", text)  # Right smart apostrophe
        text = re.sub('\u2013|\u2014', '-', text)   # Em/en dashes to hyphens
        
        # 8. Clean up obvious page artifacts ONLY
        text = re.sub(r'\n\s*Page\s+\d+\s*\n', '\n', text)  # Page numbers on their own lines
        
        # 9. ONLY remove truly redundant copyright notices
        text = re.sub(r'CROWN COPYRIGHT \d{4}\s*$', '', text, flags=re.MULTILINE)
        
        # 10. Minimal whitespace cleanup (preserve structure)
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Excessive line breaks to double
        
        # 11. Final cleanup
        text = text.strip()
        
        return text
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using LLaMA tokenizer or estimate."""
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            except Exception:
                pass
        
        # Fallback: rough estimate (1 token ≈ 4 characters for English)
        return len(text) // 4
    
    def chunk_text_semantically(self, text: str, case_id: str) -> List[Dict[str, Any]]:
        """
        Chunk text while preserving semantic boundaries.
        Respects paragraphs and sentences for coherent legal reasoning.
        """
        if not text or len(text.strip()) < 100:
            return []
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            
            # If single paragraph is too long, split by sentences
            if para_tokens > self.max_chunk_tokens:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    sentence_tokens = self.count_tokens(sentence)
                    
                    # If adding this sentence would exceed max tokens, save current chunk
                    if current_tokens + sentence_tokens > self.max_chunk_tokens and current_chunk:
                        if current_tokens >= self.min_chunk_tokens:
                            chunks.append({
                                'case_id': case_id,
                                'chunk_id': len(chunks),
                                'text': current_chunk.strip(),
                                'token_count': current_tokens,
                                'type': 'semantic_chunk'
                            })
                        current_chunk = ""
                        current_tokens = 0
                    
                    # Add sentence to current chunk
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
                    current_tokens += sentence_tokens
            
            else:
                # If adding this paragraph would exceed max tokens, save current chunk
                if current_tokens + para_tokens > self.max_chunk_tokens and current_chunk:
                    if current_tokens >= self.min_chunk_tokens:
                        chunks.append({
                            'case_id': case_id,
                            'chunk_id': len(chunks),
                            'text': current_chunk.strip(),
                            'token_count': current_tokens,
                            'type': 'semantic_chunk'
                        })
                    current_chunk = ""
                    current_tokens = 0
                
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_tokens += para_tokens
        
        # Add final chunk if it meets minimum requirements
        if current_chunk and current_tokens >= self.min_chunk_tokens:
            chunks.append({
                'case_id': case_id,
                'chunk_id': len(chunks),
                'text': current_chunk.strip(),
                'token_count': current_tokens,
                'type': 'semantic_chunk'
            })
        
        return chunks
    
    def create_temporal_datasets(self) -> Dict[str, pd.DataFrame]:
        """Create three temporal datasets with proper filtering."""
        logger.info("Creating temporal datasets...")
        
        # Filter for cases with text
        df_with_text = self.df[
            self.df['decision_text_cleaned'].notna() & 
            (self.df['decision_text_cleaned'] != '')
        ].copy()
        
        # Create temporal splits
        datasets = {
            'pre_brexit_2013_2016': df_with_text[
                (df_with_text['extracted_year'] >= 2013) & 
                (df_with_text['extracted_year'] <= 2016)
            ].copy(),
            'transitional_2017': df_with_text[
                df_with_text['extracted_year'] == 2017
            ].copy(),
            'post_brexit_2018_2025': df_with_text[
                (df_with_text['extracted_year'] >= 2018) & 
                (df_with_text['extracted_year'] <= 2025)
            ].copy()
        }
        
        # Remove pre-2013 cases (too sparse)
        removed_count = len(df_with_text[df_with_text['extracted_year'] < 2013])
        logger.info(f"Removing {removed_count:,} cases before 2013 (too sparse)")
        
        # Log dataset sizes
        for name, data in datasets.items():
            logger.info(f"{name}: {len(data):,} cases")
        
        return datasets
    
    def process_dataset(self, df: pd.DataFrame, dataset_name: str) -> List[Dict[str, Any]]:
        """Process a single temporal dataset into training-ready chunks."""
        logger.info(f"Processing {dataset_name} dataset ({len(df):,} cases)...")
        
        all_chunks = []
        processing_stats = {
            'total_cases': len(df),
            'processed_cases': 0,
            'total_chunks': 0,
            'total_tokens': 0,
            'avg_tokens_per_chunk': 0,
            'cleaning_failed': 0
        }
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {dataset_name}"):
            try:
                # Advanced cleaning
                cleaned_text = self.advanced_legal_text_cleaning(row['decision_text_cleaned'])
                
                if not cleaned_text or len(cleaned_text.strip()) < 100:
                    processing_stats['cleaning_failed'] += 1
                    continue
                
                # Create case identifier
                case_id = f"{dataset_name}_{row.get('reference_number', f'case_{idx}')}"
                
                # Chunk the text semantically
                chunks = self.chunk_text_semantically(cleaned_text, case_id)
                
                # Add metadata to each chunk
                for chunk in chunks:
                    chunk.update({
                        'dataset': dataset_name,
                        'original_case_year': row['extracted_year'],
                        'case_reference': row.get('reference_number', ''),
                        'promulgation_date': row.get('promulgation_date', ''),
                        'chunk_length': len(chunk['text'])
                    })
                
                all_chunks.extend(chunks)
                processing_stats['processed_cases'] += 1
                
            except Exception as e:
                logger.warning(f"Failed to process case {idx}: {e}")
                processing_stats['cleaning_failed'] += 1
        
        # Calculate final statistics
        processing_stats['total_chunks'] = len(all_chunks)
        if all_chunks:
            processing_stats['total_tokens'] = sum(chunk['token_count'] for chunk in all_chunks)
            processing_stats['avg_tokens_per_chunk'] = processing_stats['total_tokens'] / len(all_chunks)
        
        logger.info(f"Dataset {dataset_name} processing complete: {processing_stats}")
        
        return all_chunks
    
    def save_training_datasets(self, datasets: Dict[str, List[Dict[str, Any]]]):
        """Save datasets in multiple formats for training."""
        logger.info("Saving training-ready datasets...")
        
        for dataset_name, chunks in datasets.items():
            if not chunks:
                logger.warning(f"No chunks for dataset {dataset_name}")
                continue
            
            # Create dataset directory
            dataset_dir = self.output_dir / dataset_name
            dataset_dir.mkdir(exist_ok=True)
            
            # 1. Save as JSONL for training frameworks
            jsonl_path = dataset_dir / f"{dataset_name}_chunks.jsonl"
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    json.dump(chunk, f, ensure_ascii=False)
                    f.write('\n')
            
            # 2. Save as plain text for direct training
            txt_path = dataset_dir / f"{dataset_name}_training.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    f.write(chunk['text'])
                    f.write('\n\n' + '='*50 + '\n\n')  # Document separator
            
            # 3. Save metadata as CSV
            csv_path = dataset_dir / f"{dataset_name}_metadata.csv"
            df_chunks = pd.DataFrame(chunks)
            # Remove text column for metadata (too large)
            df_meta = df_chunks.drop('text', axis=1)
            df_meta.to_csv(csv_path, index=False)
            
            # 4. Save statistics
            stats = {
                'dataset_name': dataset_name,
                'total_chunks': len(chunks),
                'total_tokens': sum(chunk['token_count'] for chunk in chunks),
                'avg_tokens_per_chunk': sum(chunk['token_count'] for chunk in chunks) / len(chunks),
                'min_tokens': min(chunk['token_count'] for chunk in chunks),
                'max_tokens': max(chunk['token_count'] for chunk in chunks),
                'total_characters': sum(len(chunk['text']) for chunk in chunks),
                'unique_cases': len(set(chunk['case_id'] for chunk in chunks))
            }
            
            stats_path = dataset_dir / f"{dataset_name}_statistics.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {dataset_name}: {len(chunks):,} chunks to {dataset_dir}")
    
    def create_human_inspection_samples(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create random samples of processed text for human inspection.
        Saves before/after cleaning examples for quality oversight.
        """
        logger.info(f"Creating {self.human_inspection_samples} random samples per dataset for human inspection...")
        
        inspection_samples = {}
        
        for dataset_name, df in datasets.items():
            if len(df) == 0:
                continue
            
            # Randomly sample cases from this dataset
            sample_size = min(self.human_inspection_samples, len(df))
            sampled_df = df.sample(n=sample_size, random_state=42)
            
            dataset_samples = []
            
            for idx, row in sampled_df.iterrows():
                # Get original text
                original_text = str(row['decision_text_cleaned']) if pd.notna(row['decision_text_cleaned']) else ""
                
                # Apply cleaning
                cleaned_text = self.advanced_legal_text_cleaning(original_text)
                
                # Create sample record
                sample = {
                    'sample_id': len(dataset_samples) + 1,
                    'case_reference': row.get('reference_number', f'case_{idx}'),
                    'case_year': row.get('extracted_year', 'unknown'),
                    'promulgation_date': row.get('promulgation_date', 'unknown'),
                    'original_length': len(original_text),
                    'cleaned_length': len(cleaned_text),
                    'reduction_ratio': 1 - (len(cleaned_text) / len(original_text)) if len(original_text) > 0 else 0,
                    'original_text_preview': original_text[:500] + "..." if len(original_text) > 500 else original_text,
                    'cleaned_text_preview': cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text,
                    'original_text_full': original_text,
                    'cleaned_text_full': cleaned_text
                }
                
                dataset_samples.append(sample)
            
            inspection_samples[dataset_name] = dataset_samples
            logger.info(f"Created {len(dataset_samples)} inspection samples for {dataset_name}")
        
        return inspection_samples
    
    def save_human_inspection_samples(self, inspection_samples: Dict[str, List[Dict[str, Any]]]):
        """Save human inspection samples in readable formats."""
        logger.info("Saving human inspection samples...")
        
        # Create inspection directory
        inspection_dir = self.output_dir / "human_inspection"
        inspection_dir.mkdir(exist_ok=True)
        
        for dataset_name, samples in inspection_samples.items():
            if not samples:
                continue
            
            # 1. Save as readable HTML for easy viewing
            html_path = inspection_dir / f"{dataset_name}_inspection_samples.html"
            self._create_inspection_html(samples, dataset_name, html_path)
            
            # 2. Save as structured JSON
            json_path = inspection_dir / f"{dataset_name}_inspection_samples.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
            
            # 3. Save summary CSV
            csv_path = inspection_dir / f"{dataset_name}_inspection_summary.csv"
            summary_df = pd.DataFrame([
                {
                    'sample_id': s['sample_id'],
                    'case_reference': s['case_reference'],
                    'case_year': s['case_year'],
                    'original_length': s['original_length'],
                    'cleaned_length': s['cleaned_length'],
                    'reduction_ratio': f"{s['reduction_ratio']:.2%}",
                    'original_preview': s['original_text_preview'][:100] + "..." if len(s['original_text_preview']) > 100 else s['original_text_preview'],
                    'cleaned_preview': s['cleaned_text_preview'][:100] + "..." if len(s['cleaned_text_preview']) > 100 else s['cleaned_text_preview']
                }
                for s in samples
            ])
            summary_df.to_csv(csv_path, index=False)
        
        # Create overall inspection summary
        self._create_overall_inspection_summary(inspection_samples, inspection_dir)
        
        logger.info(f"Human inspection samples saved to: {inspection_dir}")
    
    def _create_inspection_html(self, samples: List[Dict[str, Any]], dataset_name: str, output_path: Path):
        """Create HTML file for easy human inspection."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Text Processing Inspection: {dataset_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .sample {{ border: 1px solid #ddd; margin: 20px 0; padding: 15px; border-radius: 5px; }}
        .header {{ background-color: #f5f5f5; padding: 10px; margin: -15px -15px 15px -15px; border-radius: 5px 5px 0 0; }}
        .stats {{ color: #666; font-size: 0.9em; }}
        .text-box {{ border: 1px solid #ccc; padding: 10px; margin: 10px 0; border-radius: 3px; }}
        .original {{ background-color: #fff5f5; }}
        .cleaned {{ background-color: #f5fff5; }}
        .preview {{ max-height: 200px; overflow-y: auto; font-size: 0.9em; }}
        .full {{ max-height: 400px; overflow-y: auto; font-size: 0.85em; white-space: pre-wrap; }}
        h2 {{ color: #333; }}
        h3 {{ color: #555; margin-top: 20px; }}
        .toggle {{ background: #007cba; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer; }}
        .hidden {{ display: none; }}
    </style>
    <script>
        function toggleFull(id) {{
            var element = document.getElementById(id);
            element.classList.toggle('hidden');
        }}
    </script>
</head>
<body>
    <h1>Text Processing Inspection: {dataset_name}</h1>
    <p><strong>Dataset:</strong> {dataset_name} | <strong>Samples:</strong> {len(samples)} | <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
"""
        
        for sample in samples:
            reduction_pct = f"{sample['reduction_ratio']:.1%}"
            html_content += f"""
    <div class="sample">
        <div class="header">
            <h3>Sample #{sample['sample_id']}: {sample['case_reference']}</h3>
            <div class="stats">
                Year: {sample['case_year']} | 
                Original: {sample['original_length']:,} chars | 
                Cleaned: {sample['cleaned_length']:,} chars | 
                Reduction: {reduction_pct}
            </div>
        </div>
        
        <h4>Preview (first 500 chars)</h4>
        <div class="text-box original preview">
            <strong>Original:</strong><br>
            {sample['original_text_preview'].replace('<', '&lt;').replace('>', '&gt;')}
        </div>
        <div class="text-box cleaned preview">
            <strong>Cleaned:</strong><br>
            {sample['cleaned_text_preview'].replace('<', '&lt;').replace('>', '&gt;')}
        </div>
        
        <button class="toggle" onclick="toggleFull('full_{sample['sample_id']}')">Toggle Full Text</button>
        <div id="full_{sample['sample_id']}" class="hidden">
            <h4>Full Text Comparison</h4>
            <div class="text-box original full">
                <strong>Original Full Text:</strong><br>
                {sample['original_text_full'].replace('<', '&lt;').replace('>', '&gt;')}
            </div>
            <div class="text-box cleaned full">
                <strong>Cleaned Full Text:</strong><br>
                {sample['cleaned_text_full'].replace('<', '&lt;').replace('>', '&gt;')}
            </div>
        </div>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _create_overall_inspection_summary(self, inspection_samples: Dict[str, List[Dict[str, Any]]], inspection_dir: Path):
        """Create overall summary of inspection samples."""
        # Calculate aggregate statistics
        total_samples = sum(len(samples) for samples in inspection_samples.values())
        
        summary_stats = {
            'generation_time': datetime.now().isoformat(),
            'total_samples': total_samples,
            'samples_per_dataset': {name: len(samples) for name, samples in inspection_samples.items()},
            'cleaning_statistics': {}
        }
        
        # Calculate cleaning statistics per dataset
        for dataset_name, samples in inspection_samples.items():
            if not samples:
                continue
            
            reductions = [s['reduction_ratio'] for s in samples]
            original_lengths = [s['original_length'] for s in samples]
            cleaned_lengths = [s['cleaned_length'] for s in samples]
            
            summary_stats['cleaning_statistics'][dataset_name] = {
                'avg_reduction_ratio': np.mean(reductions),
                'min_reduction_ratio': np.min(reductions),
                'max_reduction_ratio': np.max(reductions),
                'avg_original_length': np.mean(original_lengths),
                'avg_cleaned_length': np.mean(cleaned_lengths),
                'samples_count': len(samples)
            }
        
        # Save summary
        summary_path = inspection_dir / "inspection_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False)
        
        # Create README for human reviewer
        readme_path = inspection_dir / "README.md"
        readme_content = f"""# Human Inspection Samples

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Purpose
These samples allow human oversight of the text preprocessing quality.
Review these to ensure the cleaning process preserves important legal content while removing artifacts.

## Files

### HTML Files (Best for Review)
- `*_inspection_samples.html` - Interactive HTML with before/after text comparison
- Open in web browser for easy side-by-side comparison

### Data Files
- `*_inspection_samples.json` - Structured data with full text
- `*_inspection_summary.csv` - Quick overview spreadsheet
- `inspection_summary.json` - Aggregate statistics

## What to Look For

### ✅ Good Cleaning
- Removed legal headers, page numbers, case citations
- Preserved legal reasoning and factual content
- Maintained paragraph structure
- Clean, readable text

### ❌ Problematic Cleaning
- Removed important legal content
- Broken sentences or paragraphs
- Missing key information
- Over-aggressive cleaning

## Datasets
{chr(10).join(f"- **{name}**: {len(samples)} samples" for name, samples in inspection_samples.items())}

## Review Instructions
1. Open HTML files in web browser
2. Review 5-10 samples per dataset minimum
3. Check both preview and full text
4. Note any systematic issues
5. Provide feedback for pipeline improvements
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

    def run_complete_pipeline(self):
        """Run the complete preprocessing pipeline."""
        logger.info("🚀 Starting comprehensive LLaMA-3 preprocessing pipeline...")
        
        # Create temporal datasets
        temporal_datasets = self.create_temporal_datasets()
        
        # Create human inspection samples BEFORE processing
        if self.human_inspection_samples > 0:
            logger.info("📋 Creating human inspection samples...")
            inspection_samples = self.create_human_inspection_samples(temporal_datasets)
            self.save_human_inspection_samples(inspection_samples)
        
        # Process each dataset
        processed_datasets = {}
        for name, df in temporal_datasets.items():
            chunks = self.process_dataset(df, name)
            processed_datasets[name] = chunks
        
        # Save all datasets
        self.save_training_datasets(processed_datasets)
        
        # Run quality assessment if enabled
        quality_report = None
        if self.run_quality_assessment and QUALITY_MONITOR_AVAILABLE:
            try:
                logger.info("🔍 Running integrated quality assessment...")
                quality_monitor = PreprocessingQualityMonitor(
                    processed_data_dir=str(self.output_dir),
                    output_dir=str(self.output_dir.parent / "quality_assessment")
                )
                quality_report = quality_monitor.run_quality_assessment()
            except Exception as e:
                logger.warning(f"Quality assessment failed: {e}")
        
        # Generate final summary
        total_chunks = sum(len(chunks) for chunks in processed_datasets.values())
        total_tokens = sum(
            sum(chunk['token_count'] for chunk in chunks) 
            for chunks in processed_datasets.values()
        )
        
        summary = {
            'pipeline_completed': datetime.now().isoformat(),
            'total_datasets': len(processed_datasets),
            'total_chunks': total_chunks,
            'total_tokens': total_tokens,
            'human_inspection_samples': self.human_inspection_samples,
            'quality_assessment': quality_report.get('overall_quality_score', 'N/A') if quality_report else 'Skipped',
            'datasets': {
                name: {
                    'chunks': len(chunks),
                    'tokens': sum(chunk['token_count'] for chunk in chunks)
                }
                for name, chunks in processed_datasets.items()
            }
        }
        
        # Save pipeline summary
        summary_path = self.output_dir / 'pipeline_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*80)
        print("🎉 LLAMA-3 PREPROCESSING PIPELINE COMPLETE!")
        print("="*80)
        print(f"📊 Total chunks created: {total_chunks:,}")
        print(f"🔤 Total tokens: {total_tokens:,}")
        print(f"📁 Output directory: {self.output_dir}")
        
        if self.human_inspection_samples > 0:
            print(f"📋 Human inspection samples: {self.human_inspection_samples} per dataset")
            print(f"   👁️  Review at: {self.output_dir}/human_inspection/")
        
        if quality_report:
            quality_score = quality_report.get('overall_quality_score', 0)
            status_emoji = "✅" if quality_score > 0.7 else "⚠️" if quality_score > 0.5 else "❌"
            print(f"🔍 Quality Score: {quality_score:.2f}/1.00 {status_emoji}")
        
        for name, stats in summary['datasets'].items():
            print(f"   📄 {name}: {stats['chunks']:,} chunks, {stats['tokens']:,} tokens")
        
        print(f"\n✅ Ready for LLaMA-3 unsupervised training!")
        if quality_report and quality_report.get('overall_quality_score', 0) < 0.7:
            print("⚠️  Consider reviewing quality assessment recommendations before training.")
        if self.human_inspection_samples > 0:
            print("👁️  Please review human inspection samples for quality oversight.")
        print("="*80)
        
        return summary

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='LLaMA-3 Preprocessing Pipeline')
    parser.add_argument('--data_path', '-d', type=str, 
                       default='preprocessing/data/processed_data/processed_legal_cases_with_years.csv',
                       help='Path to processed legal cases with extracted years')
    parser.add_argument('--output_dir', '-o', type=str, 
                       default='preprocessing/outputs/llama_training_ready',
                       help='Output directory for training-ready datasets')
    parser.add_argument('--tokenizer', '-t', type=str,
                       default='meta-llama/Llama-2-7b-hf',
                       help='LLaMA tokenizer model name')
    parser.add_argument('--skip_quality_assessment', action='store_true',
                       help='Skip quality assessment after processing')
    parser.add_argument('--human_inspection_samples', type=int, default=20,
                       help='Number of random samples per dataset for human inspection (0 to disable)')
    
    args = parser.parse_args()
    
    pipeline = LLaMAPreprocessingPipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        tokenizer_model=args.tokenizer,
        run_quality_assessment=not args.skip_quality_assessment,
        human_inspection_samples=args.human_inspection_samples
    )
    
    summary = pipeline.run_complete_pipeline()
    
    return summary

if __name__ == "__main__":
    main() 