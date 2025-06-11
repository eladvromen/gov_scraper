#!/usr/bin/env python3
"""
Comprehensive Quality Assessment for Legal Case Data
Analyzes the quality of extracted decision texts and identifies potential issues.
"""

import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataQualityAssessment:
    """Comprehensive quality assessment for legal case data."""
    
    def __init__(self, data_path: str, output_dir: str = "preprocessing/quality_reports"):
        """
        Initialize quality assessment.
        
        Args:
            data_path: Path to processed legal cases (CSV or Parquet)
            output_dir: Directory to save quality reports
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        if self.data_path.suffix == '.parquet':
            self.df = pd.read_parquet(data_path)
        else:
            self.df = pd.read_csv(data_path)
        
        logger.info(f"Loaded {len(self.df):,} cases for quality assessment")
        
        # Quality metrics storage
        self.quality_report = {}
        
    def check_text_extraction_quality(self) -> Dict[str, Any]:
        """Check the quality of extracted decision texts."""
        logger.info("Analyzing text extraction quality...")
        
        # Basic text statistics
        df_with_text = self.df[self.df['decision_text_cleaned'].notna() & 
                               (self.df['decision_text_cleaned'] != '')]
        
        text_stats = {
            'total_cases': len(self.df),
            'cases_with_text': len(df_with_text),
            'cases_without_text': len(self.df) - len(df_with_text),
            'coverage_percentage': (len(df_with_text) / len(self.df)) * 100,
            'avg_text_length': df_with_text['decision_text_length'].mean(),
            'median_text_length': df_with_text['decision_text_length'].median(),
            'min_text_length': df_with_text['decision_text_length'].min(),
            'max_text_length': df_with_text['decision_text_length'].max(),
            'std_text_length': df_with_text['decision_text_length'].std()
        }
        
        # Check for suspiciously short texts (likely extraction failures)
        short_texts = df_with_text[df_with_text['decision_text_length'] < 500]
        very_short_texts = df_with_text[df_with_text['decision_text_length'] < 100]
        
        text_stats.update({
            'short_texts_count': len(short_texts),
            'short_texts_percentage': (len(short_texts) / len(df_with_text)) * 100,
            'very_short_texts_count': len(very_short_texts),
            'very_short_texts_percentage': (len(very_short_texts) / len(df_with_text)) * 100
        })
        
        return text_stats
    
    def analyze_extraction_methods(self) -> Dict[str, Any]:
        """Analyze performance by extraction method."""
        logger.info("Analyzing extraction method performance...")
        
        # Check if we have extraction method info
        if 'text_extraction_method' not in self.df.columns:
            logger.warning("No extraction method information available")
            return {}
        
        method_stats = {}
        
        for method in self.df['text_extraction_method'].unique():
            if pd.isna(method):
                continue
                
            method_cases = self.df[self.df['text_extraction_method'] == method]
            method_with_text = method_cases[
                method_cases['decision_text_cleaned'].notna() & 
                (method_cases['decision_text_cleaned'] != '')
            ]
            
            method_stats[method] = {
                'total_cases': len(method_cases),
                'successful_extractions': len(method_with_text),
                'success_rate': (len(method_with_text) / len(method_cases)) * 100 if len(method_cases) > 0 else 0,
                'avg_text_length': method_with_text['decision_text_length'].mean() if len(method_with_text) > 0 else 0,
                'median_text_length': method_with_text['decision_text_length'].median() if len(method_with_text) > 0 else 0
            }
        
        return method_stats
    
    def check_content_quality(self, sample_size: int = 100) -> Dict[str, Any]:
        """Check the quality of extracted content."""
        logger.info(f"Analyzing content quality on sample of {sample_size} cases...")
        
        # Sample cases with text for content analysis
        df_with_text = self.df[self.df['decision_text_cleaned'].notna() & 
                               (self.df['decision_text_cleaned'] != '')]
        
        if len(df_with_text) == 0:
            return {'error': 'No cases with text found'}
        
        sample_df = df_with_text.sample(n=min(sample_size, len(df_with_text)), random_state=42)
        
        content_issues = {
            'extraction_artifacts': 0,
            'ocr_errors': 0,
            'incomplete_extractions': 0,
            'non_english_content': 0,
            'missing_legal_terms': 0,
            'good_quality': 0
        }
        
        # Legal terms that should appear in tribunal decisions
        legal_terms = [
            'appellant', 'applicant', 'tribunal', 'immigration', 'asylum',
            'appeal', 'decision', 'determination', 'secretary of state',
            'home office', 'paragraph', 'evidence', 'hearing', 'judge'
        ]
        
        # OCR error patterns
        ocr_patterns = [
            r'[Il1]{3,}',  # Multiple I, l, 1 characters (common OCR error)
            r'rn{2,}',     # Multiple 'rn' (m mistaken for rn)
            r'\b[a-z]\s[a-z]\s[a-z]\b',  # Single letters with spaces
            r'[^\w\s]{5,}',  # Long sequences of special characters
        ]
        
        problematic_samples = []
        
        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Checking content quality"):
            text = str(row['decision_text_cleaned']).lower()
            issues = []
            
            # Check for legal terms
            legal_term_count = sum(1 for term in legal_terms if term in text)
            if legal_term_count < 3:
                content_issues['missing_legal_terms'] += 1
                issues.append('missing_legal_terms')
            
            # Check for OCR errors
            ocr_error_count = sum(1 for pattern in ocr_patterns if re.search(pattern, text))
            if ocr_error_count > 2:
                content_issues['ocr_errors'] += 1
                issues.append('ocr_errors')
            
            # Check for extraction artifacts
            if any(artifact in text for artifact in ['pk\x03\x04', 'content_types.xml', 'rels/.rels']):
                content_issues['extraction_artifacts'] += 1
                issues.append('extraction_artifacts')
            
            # Check for incomplete extractions (ends abruptly)
            if len(text) > 100 and not text.strip().endswith(('.', '!', '?', '"', "'")):
                content_issues['incomplete_extractions'] += 1
                issues.append('incomplete_extractions')
            
            # Check for non-English content (rough heuristic)
            english_chars = sum(1 for c in text if c.isascii())
            if len(text) > 0 and (english_chars / len(text)) < 0.9:
                content_issues['non_english_content'] += 1
                issues.append('non_english_content')
            
            if not issues:
                content_issues['good_quality'] += 1
            else:
                problematic_samples.append({
                    'reference_number': row.get('reference_number', 'Unknown'),
                    'issues': issues,
                    'text_length': len(text),
                    'text_preview': text[:200] + '...' if len(text) > 200 else text
                })
        
        # Convert to percentages
        content_quality = {k: (v / len(sample_df)) * 100 for k, v in content_issues.items()}
        content_quality['sample_size'] = len(sample_df)
        content_quality['problematic_samples'] = problematic_samples[:10]  # Keep only first 10 for review
        
        return content_quality
    
    def analyze_temporal_trends(self) -> Dict[str, Any]:
        """Analyze extraction quality trends over time."""
        logger.info("Analyzing temporal trends...")
        
        if 'case_year' not in self.df.columns:
            return {'error': 'No case year information available'}
        
        yearly_stats = {}
        
        for year in sorted(self.df['case_year'].dropna().unique()):
            year_data = self.df[self.df['case_year'] == year]
            year_with_text = year_data[
                year_data['decision_text_cleaned'].notna() & 
                (year_data['decision_text_cleaned'] != '')
            ]
            
            yearly_stats[int(year)] = {
                'total_cases': len(year_data),
                'cases_with_text': len(year_with_text),
                'coverage_percentage': (len(year_with_text) / len(year_data)) * 100 if len(year_data) > 0 else 0,
                'avg_text_length': year_with_text['decision_text_length'].mean() if len(year_with_text) > 0 else 0
            }
        
        return yearly_stats
    
    def identify_duplicate_content(self) -> Dict[str, Any]:
        """Identify potential duplicate content."""
        logger.info("Checking for duplicate content...")
        
        df_with_text = self.df[self.df['decision_text_cleaned'].notna() & 
                               (self.df['decision_text_cleaned'] != '')]
        
        if len(df_with_text) == 0:
            return {'duplicates_found': 0}
        
        # Check for exact duplicates
        exact_duplicates = df_with_text['decision_text_cleaned'].duplicated().sum()
        
        # Check for near-duplicates (first 500 characters)
        text_previews = df_with_text['decision_text_cleaned'].str[:500]
        near_duplicates = text_previews.duplicated().sum()
        
        # Find the most common text snippets
        text_counter = Counter(text_previews)
        most_common = text_counter.most_common(10)
        
        return {
            'exact_duplicates': exact_duplicates,
            'near_duplicates': near_duplicates,
            'total_with_text': len(df_with_text),
            'exact_duplicate_percentage': (exact_duplicates / len(df_with_text)) * 100,
            'near_duplicate_percentage': (near_duplicates / len(df_with_text)) * 100,
            'most_common_snippets': [{'text': text[:100], 'count': count} 
                                   for text, count in most_common if count > 1]
        }
    
    def check_missing_cases_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in cases without extracted text."""
        logger.info("Analyzing patterns in missing cases...")
        
        missing_cases = self.df[
            self.df['decision_text_cleaned'].isna() | 
            (self.df['decision_text_cleaned'] == '')
        ]
        
        if len(missing_cases) == 0:
            return {'missing_cases': 0}
        
        patterns = {
            'total_missing': len(missing_cases),
            'missing_percentage': (len(missing_cases) / len(self.df)) * 100
        }
        
        # Analyze by year
        if 'case_year' in self.df.columns:
            missing_by_year = missing_cases['case_year'].value_counts().sort_index()
            patterns['missing_by_year'] = missing_by_year.to_dict()
        
        # Analyze by URL availability
        if 'has_pdf_url' in self.df.columns and 'has_word_url' in self.df.columns:
            patterns['missing_with_pdf_url'] = missing_cases['has_pdf_url'].sum()
            patterns['missing_with_word_url'] = missing_cases['has_word_url'].sum()
            patterns['missing_with_both_urls'] = (
                missing_cases['has_pdf_url'] & missing_cases['has_word_url']
            ).sum()
        
        # Sample some missing cases for manual inspection
        sample_missing = missing_cases.head(10)[['reference_number', 'url', 'case_year']].to_dict('records')
        patterns['sample_missing_cases'] = sample_missing
        
        return patterns
    
    def generate_visualizations(self):
        """Generate quality assessment visualizations."""
        logger.info("Generating quality visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Legal Case Data Quality Assessment', fontsize=16)
        
        # 1. Text length distribution
        df_with_text = self.df[self.df['decision_text_cleaned'].notna() & 
                               (self.df['decision_text_cleaned'] != '')]
        
        if len(df_with_text) > 0:
            axes[0, 0].hist(df_with_text['decision_text_length'], bins=50, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Distribution of Decision Text Lengths')
            axes[0, 0].set_xlabel('Text Length (characters)')
            axes[0, 0].set_ylabel('Number of Cases')
            axes[0, 0].axvline(df_with_text['decision_text_length'].mean(), color='red', 
                              linestyle='--', label=f'Mean: {df_with_text["decision_text_length"].mean():.0f}')
            axes[0, 0].legend()
        
        # 2. Coverage by year
        if 'case_year' in self.df.columns:
            yearly_coverage = []
            years = []
            for year in sorted(self.df['case_year'].dropna().unique()):
                year_data = self.df[self.df['case_year'] == year]
                year_with_text = year_data[
                    year_data['decision_text_cleaned'].notna() & 
                    (year_data['decision_text_cleaned'] != '')
                ]
                coverage = (len(year_with_text) / len(year_data)) * 100 if len(year_data) > 0 else 0
                yearly_coverage.append(coverage)
                years.append(int(year))
            
            axes[0, 1].plot(years, yearly_coverage, marker='o')
            axes[0, 1].set_title('Text Coverage by Year')
            axes[0, 1].set_xlabel('Year')
            axes[0, 1].set_ylabel('Coverage Percentage')
            axes[0, 1].set_ylim(0, 105)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Text length vs word count correlation
        if len(df_with_text) > 0:
            sample_data = df_with_text.sample(n=min(1000, len(df_with_text)), random_state=42)
            axes[1, 0].scatter(sample_data['decision_text_length'], sample_data['decision_text_word_count'], 
                              alpha=0.6, s=20)
            axes[1, 0].set_title('Text Length vs Word Count')
            axes[1, 0].set_xlabel('Character Count')
            axes[1, 0].set_ylabel('Word Count')
            
            # Add correlation coefficient
            corr = sample_data['decision_text_length'].corr(sample_data['decision_text_word_count'])
            axes[1, 0].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                           transform=axes[1, 0].transAxes, fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Extraction method performance (if available)
        if 'text_extraction_method' in self.df.columns:
            method_stats = self.analyze_extraction_methods()
            if method_stats:
                methods = list(method_stats.keys())
                success_rates = [method_stats[m]['success_rate'] for m in methods]
                
                bars = axes[1, 1].bar(methods, success_rates)
                axes[1, 1].set_title('Success Rate by Extraction Method')
                axes[1, 1].set_ylabel('Success Rate (%)')
                axes[1, 1].set_ylim(0, 105)
                
                # Add value labels on bars
                for bar, rate in zip(bars, success_rates):
                    height = bar.get_height()
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                                   f'{rate:.1f}%', ha='center', va='bottom')
                
                plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            axes[1, 1].text(0.5, 0.5, 'No extraction method\ninformation available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Extraction Method Performance')
        
        plt.tight_layout()
        viz_path = self.output_dir / 'quality_assessment_visualizations.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {viz_path}")
        return viz_path
    
    def run_full_assessment(self) -> Dict[str, Any]:
        """Run complete quality assessment."""
        logger.info("Starting comprehensive quality assessment...")
        
        # Run all quality checks
        self.quality_report['basic_stats'] = self.check_text_extraction_quality()
        self.quality_report['extraction_methods'] = self.analyze_extraction_methods()
        self.quality_report['content_quality'] = self.check_content_quality()
        self.quality_report['temporal_trends'] = self.analyze_temporal_trends()
        self.quality_report['duplicates'] = self.identify_duplicate_content()
        self.quality_report['missing_patterns'] = self.check_missing_cases_patterns()
        
        # Generate visualizations
        viz_path = self.generate_visualizations()
        self.quality_report['visualization_path'] = str(viz_path)
        
        # Add metadata
        self.quality_report['assessment_date'] = datetime.now().isoformat()
        self.quality_report['data_source'] = str(self.data_path)
        
        # Save comprehensive report
        report_path = self.output_dir / 'comprehensive_quality_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.quality_report, f, indent=2, default=str)
        
        logger.info(f"Comprehensive quality report saved to {report_path}")
        
        return self.quality_report
    
    def print_summary(self):
        """Print a summary of quality assessment results."""
        if not self.quality_report:
            logger.error("No quality report available. Run assessment first.")
            return
        
        print("\n" + "="*80)
        print("📊 DATA QUALITY ASSESSMENT SUMMARY")
        print("="*80)
        
        # Basic statistics
        basic = self.quality_report.get('basic_stats', {})
        print(f"📈 Overall Coverage: {basic.get('coverage_percentage', 0):.1f}%")
        print(f"📄 Cases with text: {basic.get('cases_with_text', 0):,} / {basic.get('total_cases', 0):,}")
        print(f"📏 Average text length: {basic.get('avg_text_length', 0):,.0f} characters")
        print(f"⚠️  Short texts (<500 chars): {basic.get('short_texts_percentage', 0):.1f}%")
        
        # Content quality
        content = self.quality_report.get('content_quality', {})
        if content:
            print(f"\n🔍 Content Quality (sample of {content.get('sample_size', 0)} cases):")
            print(f"   ✅ Good quality: {content.get('good_quality', 0):.1f}%")
            print(f"   🔤 OCR errors: {content.get('ocr_errors', 0):.1f}%")
            print(f"   ⚡ Extraction artifacts: {content.get('extraction_artifacts', 0):.1f}%")
            print(f"   📚 Missing legal terms: {content.get('missing_legal_terms', 0):.1f}%")
        
        # Duplicates
        dupe = self.quality_report.get('duplicates', {})
        if dupe:
            print(f"\n🔄 Duplicate Analysis:")
            print(f"   📋 Exact duplicates: {dupe.get('exact_duplicate_percentage', 0):.1f}%")
            print(f"   📝 Near duplicates: {dupe.get('near_duplicate_percentage', 0):.1f}%")
        
        # Extraction methods
        methods = self.quality_report.get('extraction_methods', {})
        if methods:
            print(f"\n⚙️  Extraction Method Performance:")
            for method, stats in methods.items():
                print(f"   {method}: {stats.get('success_rate', 0):.1f}% success rate")
        
        print("\n" + "="*80)
        print("💡 Recommendations:")
        
        # Generate recommendations based on findings
        recommendations = []
        
        if basic.get('short_texts_percentage', 0) > 5:
            recommendations.append("• High percentage of short texts - consider re-extracting failed cases")
        
        if content.get('ocr_errors', 0) > 10:
            recommendations.append("• Significant OCR errors detected - consider text post-processing")
        
        if content.get('extraction_artifacts', 0) > 5:
            recommendations.append("• Extraction artifacts found - improve file format detection")
        
        if dupe.get('exact_duplicate_percentage', 0) > 1:
            recommendations.append("• Duplicate content detected - consider deduplication")
        
        if not recommendations:
            recommendations.append("• Data quality looks good - ready for analysis!")
        
        for rec in recommendations:
            print(rec)
        
        print("="*80)

def main():
    """Run quality assessment on processed legal cases."""
    
    # Configuration
    data_path = "processed_data/processed_legal_cases.csv"
    
    if not Path(data_path).exists():
        # Try parquet format
        data_path = "processed_data/processed_legal_cases.parquet"
        if not Path(data_path).exists():
            print("❌ No processed data found. Run preprocessing first.")
            return
    
    # Run assessment
    assessor = DataQualityAssessment(data_path)
    
    print("🔍 Running comprehensive data quality assessment...")
    print("This may take a few minutes...")
    
    # Run full assessment
    report = assessor.run_full_assessment()
    
    # Print summary
    assessor.print_summary()
    
    print(f"\n📁 Detailed reports saved to: preprocessing/quality_reports/")
    print(f"📊 Visualizations: {report['visualization_path']}")

if __name__ == "__main__":
    main() 