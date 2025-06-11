#!/usr/bin/env python3
"""
Analyze failure patterns and URL tracking for document extraction

This script analyzes:
1. Types of failures and their frequencies
2. URL patterns that fail vs succeed
3. Download vs extraction failures
4. Recommendations for full run
"""

import sys
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import re

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def analyze_checkpoint_failures():
    """Analyze failures from checkpoint data"""
    checkpoint_file = Path("checkpoints/progress_checkpoint.json")
    
    if not checkpoint_file.exists():
        print("⚠️  No checkpoint file found")
        return {}
    
    with open(checkpoint_file, 'r') as f:
        data = json.load(f)
    
    analysis = {
        'total_processed': len(data.get('processed_cases', [])),
        'successful': len(data.get('successful_cases', [])),
        'failed': len(data.get('failed_cases', [])),
        'success_rate': 0,
        'avg_download_time': 0,
        'avg_extraction_time': 0,
        'current_rate_limit': data.get('current_rate_limit', 1.0)
    }
    
    if analysis['total_processed'] > 0:
        analysis['success_rate'] = analysis['successful'] / analysis['total_processed'] * 100
    
    if analysis['total_processed'] > 0:
        analysis['avg_download_time'] = data.get('total_download_time', 0) / analysis['total_processed']
        analysis['avg_extraction_time'] = data.get('total_extraction_time', 0) / analysis['total_processed']
    
    return analysis

def analyze_log_failures():
    """Analyze failure patterns from log files"""
    log_dir = Path("../logs")
    if not log_dir.exists():
        print("⚠️  No logs directory found")
        return {}
    
    # Find the most recent log file
    log_files = list(log_dir.glob("document_extractor_*.log"))
    if not log_files:
        print("⚠️  No log files found")
        return {}
    
    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    print(f"📋 Analyzing log file: {latest_log.name}")
    
    failure_patterns = {
        'download_failures': [],
        'extraction_failures': [],
        'json_update_failures': [],
        'url_fetch_failures': [],
        'rate_limit_hits': 0,
        'html_responses': 0,
        'small_files': 0,
        'ocr_successes': 0
    }
    
    try:
        with open(latest_log, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if 'FAILED' in line and 'All extraction attempts failed' in line:
                    # Extract case reference and failure details
                    if '[' in line and ']' in line:
                        ref_match = re.search(r'\[([^\]]+)\]', line)
                        if ref_match:
                            case_ref = f"[{ref_match.group(1)}]"
                            failure_patterns['extraction_failures'].append({
                                'case': case_ref,
                                'reason': line.split('failed: ')[-1].strip() if 'failed: ' in line else 'unknown',
                                'line': line_num
                            })
                
                elif 'Failed to download' in line:
                    failure_patterns['download_failures'].append(line.strip())
                
                elif 'JSON file not found' in line:
                    failure_patterns['json_update_failures'].append(line.strip())
                
                elif 'Rate limited' in line:
                    failure_patterns['rate_limit_hits'] += 1
                
                elif 'Got HTML response instead of document' in line:
                    failure_patterns['html_responses'] += 1
                
                elif 'very small' in line.lower():
                    failure_patterns['small_files'] += 1
                
                elif 'OCR extracted' in line and 'characters' in line:
                    failure_patterns['ocr_successes'] += 1
    
    except Exception as e:
        print(f"❌ Error reading log file: {e}")
        return {}
    
    return failure_patterns

def analyze_csv_data():
    """Analyze the CSV data for URL patterns"""
    csv_path = Path("../preprocessing/processed_data/processed_legal_cases.csv")
    if not csv_path.exists():
        print("⚠️  CSV file not found")
        return {}
    
    print("📊 Analyzing CSV data...")
    df = pd.read_csv(csv_path)
    
    # Find cases with missing text
    missing_text = (
        df['decision_text_cleaned'].isna() | 
        (df['decision_text_cleaned'] == '') |
        (df['decision_text_cleaned'].str.len() < 50)
    )
    
    missing_df = df[missing_text]
    
    # Analyze URL patterns
    url_analysis = {
        'total_cases': len(df),
        'missing_text_cases': len(missing_df),
        'cases_with_main_url': missing_df['url'].notna().sum(),
        'cases_with_word_url': missing_df['word_url'].notna().sum(),
        'cases_with_pdf_url': missing_df['pdf_url'].notna().sum(),
        'cases_with_either_url': (missing_df['word_url'].notna() | missing_df['pdf_url'].notna()).sum(),
        'cases_with_both_urls': (missing_df['word_url'].notna() & missing_df['pdf_url'].notna()).sum(),
        'processable_cases': 0
    }
    
    # Cases we can actually process (have main URL for fresh URL fetching)
    processable = missing_df[missing_df['url'].notna()]
    url_analysis['processable_cases'] = len(processable)
    
    # Year distribution of missing cases
    if 'case_year' in missing_df.columns:
        year_dist = missing_df['case_year'].value_counts().sort_index()
        url_analysis['year_distribution'] = year_dist.to_dict()
    
    return url_analysis

def generate_recommendations(checkpoint_analysis, log_analysis, csv_analysis):
    """Generate recommendations for the full run"""
    recommendations = []
    
    # Success rate recommendations
    if checkpoint_analysis.get('success_rate', 0) < 20:
        recommendations.append("⚠️  Low success rate - investigate common failure patterns before full run")
    elif checkpoint_analysis.get('success_rate', 0) > 50:
        recommendations.append("✅ Good success rate - ready for full run")
    else:
        recommendations.append("⚖️  Moderate success rate - consider running with smaller batches")
    
    # Rate limiting recommendations
    current_rate = checkpoint_analysis.get('current_rate_limit', 1.0)
    if current_rate > 2.0:
        recommendations.append(f"🐌 Rate limit is high ({current_rate:.2f}s) - expect slower processing")
    elif current_rate < 0.5:
        recommendations.append(f"🚀 Rate limit is low ({current_rate:.2f}s) - good processing speed expected")
    
    # URL availability recommendations
    processable_ratio = csv_analysis.get('processable_cases', 0) / csv_analysis.get('missing_text_cases', 1)
    if processable_ratio > 0.9:
        recommendations.append("✅ Most cases have processable URLs")
    else:
        recommendations.append(f"⚠️  Only {processable_ratio*100:.1f}% of cases have processable URLs")
    
    # OCR recommendations
    ocr_successes = log_analysis.get('ocr_successes', 0)
    if ocr_successes > 0:
        recommendations.append(f"✅ OCR is working - {ocr_successes} successful OCR extractions detected")
    
    # Failure pattern recommendations
    json_failures = len(log_analysis.get('json_update_failures', []))
    if json_failures > 0:
        recommendations.append(f"⚠️  {json_failures} JSON update failures - check data directory structure")
    
    html_responses = log_analysis.get('html_responses', 0)
    if html_responses > 2:
        recommendations.append(f"⚠️  {html_responses} HTML responses instead of documents - some URLs may be invalid")
    
    return recommendations

def main():
    print("🔍 Analyzing Failure Patterns and URL Tracking")
    print("=" * 50)
    
    # Analyze checkpoint data
    print("\n📋 Checkpoint Analysis")
    print("-" * 20)
    checkpoint_analysis = analyze_checkpoint_failures()
    
    if checkpoint_analysis:
        print(f"   Total processed: {checkpoint_analysis['total_processed']}")
        print(f"   Successful: {checkpoint_analysis['successful']}")
        print(f"   Failed: {checkpoint_analysis['failed']}")
        print(f"   Success rate: {checkpoint_analysis['success_rate']:.1f}%")
        print(f"   Avg download time: {checkpoint_analysis['avg_download_time']:.2f}s")
        print(f"   Avg extraction time: {checkpoint_analysis['avg_extraction_time']:.2f}s")
        print(f"   Current rate limit: {checkpoint_analysis['current_rate_limit']:.2f}s")
    
    # Analyze log patterns
    print("\n📋 Log Pattern Analysis")
    print("-" * 20)
    log_analysis = analyze_log_failures()
    
    if log_analysis:
        print(f"   Download failures: {len(log_analysis['download_failures'])}")
        print(f"   Extraction failures: {len(log_analysis['extraction_failures'])}")
        print(f"   JSON update failures: {len(log_analysis['json_update_failures'])}")
        print(f"   Rate limit hits: {log_analysis['rate_limit_hits']}")
        print(f"   HTML responses: {log_analysis['html_responses']}")
        print(f"   Small files: {log_analysis['small_files']}")
        print(f"   OCR successes: {log_analysis['ocr_successes']}")
        
        # Show sample extraction failures
        if log_analysis['extraction_failures']:
            print(f"\n   Sample extraction failures:")
            for failure in log_analysis['extraction_failures'][:3]:
                print(f"     {failure['case']}: {failure['reason']}")
    
    # Analyze CSV data
    print("\n📋 CSV Data Analysis")
    print("-" * 20)
    csv_analysis = analyze_csv_data()
    
    if csv_analysis:
        print(f"   Total cases: {csv_analysis['total_cases']:,}")
        print(f"   Missing text cases: {csv_analysis['missing_text_cases']:,}")
        print(f"   Processable cases: {csv_analysis['processable_cases']:,}")
        print(f"   Cases with main URL: {csv_analysis['cases_with_main_url']:,}")
        print(f"   Cases with word URL: {csv_analysis['cases_with_word_url']:,}")
        print(f"   Cases with PDF URL: {csv_analysis['cases_with_pdf_url']:,}")
        print(f"   Cases with both URLs: {csv_analysis['cases_with_both_urls']:,}")
        
        if 'year_distribution' in csv_analysis:
            print(f"\n   Cases by year (top 5):")
            year_dist = csv_analysis['year_distribution']
            for year in sorted(year_dist.keys(), reverse=True)[:5]:
                print(f"     {year}: {year_dist[year]:,} cases")
    
    # Generate recommendations
    print("\n🎯 Recommendations for Full Run")
    print("-" * 30)
    recommendations = generate_recommendations(checkpoint_analysis, log_analysis, csv_analysis)
    
    for rec in recommendations:
        print(f"   {rec}")
    
    # Suggest optimal batch size
    if csv_analysis.get('processable_cases', 0) > 0:
        total_cases = csv_analysis['processable_cases']
        if total_cases > 5000:
            suggested_batch = 100
        elif total_cases > 1000:
            suggested_batch = 50
        else:
            suggested_batch = 20
        
        print(f"\n💡 Suggested Settings:")
        print(f"   Batch size: {suggested_batch}")
        print(f"   Estimated time: {total_cases / (60 / checkpoint_analysis.get('avg_extraction_time', 10)):.1f} hours")
        
        print(f"\n🚀 Ready to run:")
        print(f"   python run_document_extraction.py --full --optimized --batch-size {suggested_batch}")

if __name__ == "__main__":
    main() 