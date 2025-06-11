#!/usr/bin/env python3
"""
Brexit Temporal Analysis and Preprocessing Pipeline
=================================================

Analyzes the temporal distribution of legal cases and creates preprocessing
pipeline for pre-Brexit (2013-2016) vs post-Brexit (2018-2025) comparison.

Excludes 2017 as the transitional Brexit year.
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BrexitTemporalAnalyzer:
    """Analyze temporal distribution for Brexit comparison and prepare data splits."""
    
    def __init__(self, data_path: str, output_dir: str = "preprocessing/brexit_analysis"):
        """Initialize the analyzer."""
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        logger.info(f"Loading data from {data_path}")
        if self.data_path.suffix == '.parquet':
            self.df = pd.read_parquet(data_path)
        else:
            self.df = pd.read_csv(data_path, low_memory=False)
        
        logger.info(f"Loaded {len(self.df):,} cases")
        
        # Filter for cases with text
        self.df_with_text = self.df[
            self.df['decision_text_cleaned'].notna() & 
            (self.df['decision_text_cleaned'] != '')
        ]
        logger.info(f"Cases with text: {len(self.df_with_text):,}")
        
    def analyze_temporal_distribution(self) -> Dict[str, Any]:
        """Analyze temporal distribution and Brexit timeframes."""
        logger.info("Analyzing temporal distribution...")
        
        # Get year distribution
        year_dist = self.df_with_text['case_year'].value_counts().sort_index()
        
        # Brexit timeframe splits
        pre_brexit = self.df_with_text[
            (self.df_with_text['case_year'] >= 2013) & 
            (self.df_with_text['case_year'] <= 2016)
        ]
        
        post_brexit = self.df_with_text[
            (self.df_with_text['case_year'] >= 2018) & 
            (self.df_with_text['case_year'] <= 2025)
        ]
        
        excluded_2017 = self.df_with_text[self.df_with_text['case_year'] == 2017]
        
        # Calculate statistics
        analysis = {
            'total_cases': len(self.df),
            'cases_with_text': len(self.df_with_text),
            'year_distribution': year_dist.to_dict(),
            'brexit_analysis': {
                'pre_brexit': {
                    'years': '2013-2016',
                    'case_count': len(pre_brexit),
                    'total_characters': int(pre_brexit['decision_text_length'].sum()),
                    'avg_length': float(pre_brexit['decision_text_length'].mean()),
                    'total_words': int(pre_brexit['decision_text_word_count'].sum()),
                    'avg_words': float(pre_brexit['decision_text_word_count'].mean())
                },
                'post_brexit': {
                    'years': '2018-2025',
                    'case_count': len(post_brexit),
                    'total_characters': int(post_brexit['decision_text_length'].sum()),
                    'avg_length': float(post_brexit['decision_text_length'].mean()),
                    'total_words': int(post_brexit['decision_text_word_count'].sum()),
                    'avg_words': float(post_brexit['decision_text_word_count'].mean())
                },
                'excluded_2017': {
                    'case_count': len(excluded_2017),
                    'total_characters': int(excluded_2017['decision_text_length'].sum()) if len(excluded_2017) > 0 else 0
                }
            }
        }
        
        # Add ratio analysis
        if len(post_brexit) > 0:
            analysis['brexit_analysis']['pre_post_ratio'] = len(pre_brexit) / len(post_brexit)
        
        return analysis
    
    def print_temporal_summary(self, analysis: Dict[str, Any]):
        """Print a comprehensive temporal analysis summary."""
        print("\n" + "="*80)
        print("🕰️  BREXIT TEMPORAL ANALYSIS")
        print("="*80)
        
        print(f"\n📊 OVERALL STATISTICS")
        print(f"   📄 Total cases: {analysis['total_cases']:,}")
        print(f"   ✅ With text: {analysis['cases_with_text']:,}")
        
        print(f"\n🕘 TEMPORAL DISTRIBUTION (2010-2025)")
        year_dist = analysis['year_distribution']
        for year in sorted(year_dist.keys()):
            if pd.notna(year) and year >= 2010:
                count = year_dist[year]
                if 2013 <= year <= 2016:
                    emoji = "🟢"  # Pre-Brexit
                elif year == 2017:
                    emoji = "⚫"  # Excluded
                elif 2018 <= year <= 2025:
                    emoji = "🔵"  # Post-Brexit
                else:
                    emoji = "⚪"  # Other
                print(f"   {emoji} {int(year)}: {count:,} cases")
        
        brexit = analysis['brexit_analysis']
        
        print(f"\n🟢 PRE-BREXIT CORPUS (2013-2016)")
        pre = brexit['pre_brexit']
        print(f"   📄 Cases: {pre['case_count']:,}")
        print(f"   📝 Characters: {pre['total_characters']:,}")
        print(f"   🔤 Words: {pre['total_words']:,}")
        print(f"   📏 Avg length: {pre['avg_length']:.0f} chars")
        
        print(f"\n🔵 POST-BREXIT CORPUS (2018-2025)")
        post = brexit['post_brexit']
        print(f"   📄 Cases: {post['case_count']:,}")
        print(f"   📝 Characters: {post['total_characters']:,}")
        print(f"   🔤 Words: {post['total_words']:,}")
        print(f"   📏 Avg length: {post['avg_length']:.0f} chars")
        
        print(f"\n⚫ EXCLUDED 2017 (Transitional)")
        excluded = brexit['excluded_2017']
        print(f"   📄 Cases: {excluded['case_count']:,}")
        print(f"   📝 Characters: {excluded['total_characters']:,}")
        
        if 'pre_post_ratio' in brexit:
            print(f"\n📊 PRE/POST RATIO: {brexit['pre_post_ratio']:.2f}")
            if brexit['pre_post_ratio'] > 1.5:
                print("   ⚠️  Pre-Brexit corpus significantly larger")
            elif brexit['pre_post_ratio'] < 0.67:
                print("   ⚠️  Post-Brexit corpus significantly larger")
            else:
                print("   ✅ Reasonably balanced corpora")
        
        print("\n" + "="*80)
    
    def create_temporal_splits(self) -> Dict[str, pd.DataFrame]:
        """Create pre-Brexit and post-Brexit datasets."""
        logger.info("Creating temporal splits...")
        
        # Filter by years and ensure we have text
        pre_brexit = self.df_with_text[
            (self.df_with_text['case_year'] >= 2013) & 
            (self.df_with_text['case_year'] <= 2016)
        ].copy()
        
        post_brexit = self.df_with_text[
            (self.df_with_text['case_year'] >= 2018) & 
            (self.df_with_text['case_year'] <= 2025)
        ].copy()
        
        # Remove cases before 2013 (too sparse)
        pre_2013 = self.df_with_text[self.df_with_text['case_year'] < 2013]
        logger.info(f"Removing {len(pre_2013):,} cases before 2013 (too sparse)")
        
        # Add temporal labels
        pre_brexit['temporal_period'] = 'pre_brexit'
        post_brexit['temporal_period'] = 'post_brexit'
        
        logger.info(f"Pre-Brexit corpus: {len(pre_brexit):,} cases")
        logger.info(f"Post-Brexit corpus: {len(post_brexit):,} cases")
        
        return {
            'pre_brexit': pre_brexit,
            'post_brexit': post_brexit
        }
    
    def advanced_text_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply advanced text cleaning for language model training."""
        logger.info(f"Applying advanced text cleaning to {len(df):,} cases...")
        
        df_clean = df.copy()
        
        # Track cleaning statistics
        cleaning_stats = {
            'original_cases': len(df_clean),
            'repeated_disclaimers_removed': 0,
            'headers_footers_removed': 0,
            'whitespace_normalized': 0,
            'empty_after_cleaning': 0
        }
        
        for idx, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Advanced cleaning"):
            text = str(row['decision_text_cleaned'])
            original_text = text
            
            # Remove repeated disclaimers and headers
            disclaimer_patterns = [
                r'IMPORTANT NOTICE:.*?(?=\n\n|\n[A-Z])',
                r'This determination contains.*?(?=\n\n|\n[A-Z])',
                r'NOTE: This determination.*?(?=\n\n|\n[A-Z])',
                r'Crown Copyright.*?(?=\n\n|\n[A-Z])',
                r'Neutral Citation Number:.*?(?=\n\n|\n[A-Z])',
            ]
            
            for pattern in disclaimer_patterns:
                if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                    text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
                    cleaning_stats['repeated_disclaimers_removed'] += 1
            
            # Remove headers and footers patterns
            header_footer_patterns = [
                r'^.*?Date of hearing:.*?\n',  # Header info
                r'^.*?Date Determination notified:.*?\n',
                r'^.*?Before\n.*?\n.*?\n',  # Judge info
                r'\n\s*Page \d+ of \d+\s*\n',  # Page numbers
                r'\n\s*\d+\s*\n$',  # Trailing page numbers
            ]
            
            for pattern in header_footer_patterns:
                if re.search(pattern, text, re.MULTILINE):
                    text = re.sub(pattern, '\n', text, flags=re.MULTILINE)
                    cleaning_stats['headers_footers_removed'] += 1
            
            # Normalize whitespace while preserving paragraph boundaries
            # Replace multiple spaces with single space
            text = re.sub(r' +', ' ', text)
            
            # Normalize line breaks - preserve paragraph boundaries
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple line breaks -> double
            text = re.sub(r'\n(?!\n)', ' ', text)  # Single line breaks -> space
            text = re.sub(r'\n\n', '\n\n', text)  # Keep paragraph breaks
            
            # Remove leading/trailing whitespace
            text = text.strip()
            
            if len(text) != len(original_text):
                cleaning_stats['whitespace_normalized'] += 1
            
            # Update the text
            df_clean.at[idx, 'decision_text_cleaned'] = text
            
            # Check if text became empty
            if not text or len(text.strip()) < 100:  # Less than 100 chars
                cleaning_stats['empty_after_cleaning'] += 1
        
        # Remove cases that became too short after cleaning
        original_len = len(df_clean)
        df_clean = df_clean[
            df_clean['decision_text_cleaned'].str.len() >= 100
        ]
        
        cleaning_stats['final_cases'] = len(df_clean)
        cleaning_stats['removed_short'] = original_len - len(df_clean)
        
        logger.info(f"Cleaning completed: {cleaning_stats}")
        
        return df_clean
    
    def save_temporal_datasets(self, splits: Dict[str, pd.DataFrame]):
        """Save the temporal splits to files."""
        logger.info("Saving temporal datasets...")
        
        for period, data in splits.items():
            # Apply advanced cleaning
            data_clean = self.advanced_text_cleaning(data)
            
            # Save as CSV
            output_path = self.output_dir / f"{period}_cases_cleaned.csv"
            data_clean.to_csv(output_path, index=False)
            logger.info(f"Saved {period}: {len(data_clean):,} cases to {output_path}")
            
            # Save text-only version for LM training
            text_only_path = self.output_dir / f"{period}_texts_only.txt"
            with open(text_only_path, 'w', encoding='utf-8') as f:
                for text in data_clean['decision_text_cleaned']:
                    f.write(text + '\n\n' + '='*50 + '\n\n')
            logger.info(f"Saved text-only version to {text_only_path}")
    
    def run_comprehensive_analysis(self):
        """Run complete Brexit temporal analysis and preprocessing."""
        logger.info("Starting comprehensive Brexit temporal analysis...")
        
        # Analyze temporal distribution
        analysis = self.analyze_temporal_distribution()
        
        # Print summary
        self.print_temporal_summary(analysis)
        
        # Create temporal splits
        splits = self.create_temporal_splits()
        
        # Save datasets
        self.save_temporal_datasets(splits)
        
        # Save analysis results
        results_path = self.output_dir / 'brexit_temporal_analysis.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Analysis saved to {results_path}")
        
        return analysis, splits

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Brexit temporal analysis and preprocessing')
    parser.add_argument('--data_path', '-d', type=str, 
                       default='preprocessing/processed_data/processed_legal_cases.csv',
                       help='Path to processed legal cases')
    parser.add_argument('--output_dir', '-o', type=str, 
                       default='preprocessing/brexit_analysis',
                       help='Output directory')
    
    args = parser.parse_args()
    
    analyzer = BrexitTemporalAnalyzer(args.data_path, args.output_dir)
    analysis, splits = analyzer.run_comprehensive_analysis()
    
    print("\n🎉 Brexit temporal analysis complete!")
    print(f"📁 Results saved to: {args.output_dir}")
    
    return analysis, splits

if __name__ == "__main__":
    main() 