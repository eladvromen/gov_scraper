#!/usr/bin/env python3
"""
Extraction Results Analyzer

Analyzes the results of document text extraction and creates summaries
for easy evaluation of success rates and failure patterns.
"""

import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import re

class ExtractionResultsAnalyzer:
    """Analyzes extraction results and generates evaluation reports"""
    
    def __init__(self, csv_path: str, json_dir: str = "data/json"):
        self.csv_path = Path(csv_path)
        self.json_dir = Path(json_dir)
        self.results = []
        
    def analyze_results(self) -> dict:
        """Analyze extraction results from updated JSON files"""
        print("🔍 Analyzing extraction results...")
        
        # Load CSV to get the list of cases that needed extraction
        df = pd.read_csv(self.csv_path)
        missing_text = (
            df['decision_text_cleaned'].isna() | 
            (df['decision_text_cleaned'] == '') |
            (df['decision_text_cleaned'].str.len() < 50)
        )
        target_cases = df[missing_text].copy()
        
        stats = {
            'total_target_cases': len(target_cases),
            'extraction_attempts': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'extraction_methods': Counter(),
            'failure_reasons': Counter(),
            'text_length_stats': [],
            'cases_by_status': []
        }
        
        print(f"📊 Analyzing {len(target_cases)} target cases...")
        
        for idx, case_row in target_cases.iterrows():
            reference_number = case_row['reference_number']
            
            # Check if JSON was updated with extracted text
            json_filename = reference_number.replace('/', '_') + '.json'
            json_path = self.json_dir / json_filename
            
            case_result = {
                'reference_number': reference_number,
                'json_exists': json_path.exists(),
                'extraction_attempted': False,
                'extraction_successful': False,
                'extraction_method': None,
                'text_length': 0,
                'extraction_date': None,
                'failure_reason': None
            }
            
            if json_path.exists():
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        case_data = json.load(f)
                    
                    # Check if extraction was attempted (has extraction metadata)
                    if 'text_extraction_date' in case_data:
                        case_result['extraction_attempted'] = True
                        case_result['extraction_date'] = case_data['text_extraction_date']
                        stats['extraction_attempts'] += 1
                        
                        # Check if extraction was successful
                        decision_text = case_data.get('decision_text', '')
                        if decision_text and len(decision_text.strip()) > 100:
                            case_result['extraction_successful'] = True
                            case_result['text_length'] = len(decision_text)
                            case_result['extraction_method'] = case_data.get('text_extraction_method', 'unknown')
                            
                            stats['successful_extractions'] += 1
                            stats['extraction_methods'][case_result['extraction_method']] += 1
                            stats['text_length_stats'].append(len(decision_text))
                            
                            # Determine actual extraction method from text characteristics
                            actual_method = self._detect_extraction_method(decision_text)
                            if actual_method != case_result['extraction_method']:
                                case_result['detected_method'] = actual_method
                        else:
                            case_result['failure_reason'] = 'text_too_short_or_empty'
                            stats['failed_extractions'] += 1
                            stats['failure_reasons']['text_too_short_or_empty'] += 1
                    else:
                        case_result['failure_reason'] = 'no_extraction_attempted'
                        stats['failure_reasons']['no_extraction_attempted'] += 1
                        
                except Exception as e:
                    case_result['failure_reason'] = f'json_read_error: {str(e)}'
                    stats['failure_reasons']['json_read_error'] += 1
            else:
                case_result['failure_reason'] = 'json_file_missing'
                stats['failure_reasons']['json_file_missing'] += 1
            
            self.results.append(case_result)
            stats['cases_by_status'].append(case_result)
        
        # Calculate additional statistics
        if stats['text_length_stats']:
            stats['avg_text_length'] = sum(stats['text_length_stats']) / len(stats['text_length_stats'])
            stats['min_text_length'] = min(stats['text_length_stats'])
            stats['max_text_length'] = max(stats['text_length_stats'])
        
        stats['success_rate'] = (stats['successful_extractions'] / max(stats['extraction_attempts'], 1)) * 100
        
        return stats
    
    def _detect_extraction_method(self, text: str) -> str:
        """Detect the actual extraction method from text characteristics"""
        # OCR indicators
        ocr_indicators = [
            len(re.findall(r'[A-Z]{2,}', text)) > len(text) / 100,  # Many uppercase words
            'l ' in text[:1000],  # OCR often confuses 'I' with 'l '
            ' rn ' in text[:1000],  # OCR often confuses 'm' with 'rn'
            text.count('\n') > len(text) / 50,  # OCR tends to have many line breaks
        ]
        
        if sum(ocr_indicators) >= 2:
            return "OCR"
        elif any(indicator in text.lower() for indicator in ['%pdf', 'endstream', 'endobj']):
            return "Raw PDF binary"
        elif text.count('\n\n') > len(text) / 200:
            return "PDF text extraction"
        else:
            return "Word document extraction"
    
    def generate_summary_report(self, stats: dict) -> str:
        """Generate a human-readable summary report"""
        report = [
            "📈 DOCUMENT EXTRACTION ANALYSIS REPORT",
            "=" * 50,
            f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "📊 OVERVIEW:",
            f"   Total target cases: {stats['total_target_cases']}",
            f"   Extraction attempts: {stats['extraction_attempts']}",
            f"   Successful extractions: {stats['successful_extractions']}",
            f"   Failed extractions: {stats['failed_extractions']}",
            f"   Success rate: {stats['success_rate']:.1f}%",
            "",
            "🔧 EXTRACTION METHODS:",
        ]
        
        for method, count in stats['extraction_methods'].most_common():
            percentage = (count / stats['successful_extractions']) * 100 if stats['successful_extractions'] > 0 else 0
            report.append(f"   {method}: {count} cases ({percentage:.1f}%)")
        
        report.extend([
            "",
            "❌ FAILURE REASONS:",
        ])
        
        for reason, count in stats['failure_reasons'].most_common():
            percentage = (count / stats['total_target_cases']) * 100
            report.append(f"   {reason}: {count} cases ({percentage:.1f}%)")
        
        if stats['text_length_stats']:
            report.extend([
                "",
                "📏 TEXT LENGTH STATISTICS:",
                f"   Average: {stats['avg_text_length']:,.0f} characters",
                f"   Range: {stats['min_text_length']:,} - {stats['max_text_length']:,} characters",
            ])
        
        return "\n".join(report)
    
    def create_detailed_csv(self) -> str:
        """Create detailed CSV for case-by-case evaluation"""
        df = pd.DataFrame(self.results)
        
        # Add readable status column
        def get_status(row):
            if row['extraction_successful']:
                return f"✅ SUCCESS ({row['extraction_method']})"
            elif row['extraction_attempted']:
                return f"❌ FAILED ({row['failure_reason']})"
            else:
                return f"⏭️ SKIPPED ({row['failure_reason']})"
        
        df['status'] = df.apply(get_status, axis=1)
        df['text_length_readable'] = df['text_length'].apply(lambda x: f"{x:,}" if x > 0 else "0")
        
        # Reorder columns for better readability
        columns = ['reference_number', 'status', 'extraction_method', 'text_length_readable', 
                  'extraction_date', 'failure_reason']
        df_display = df[columns].copy()
        
        output_path = f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_display.to_csv(output_path, index=False)
        
        return output_path
    
    def run_analysis(self):
        """Run complete analysis and generate reports"""
        stats = self.analyze_results()
        
        # Generate summary report
        summary = self.generate_summary_report(stats)
        print("\n" + summary)
        
        # Create detailed CSV
        csv_path = self.create_detailed_csv()
        print(f"\n📄 Detailed results saved to: {csv_path}")
        
        # Show sample successful and failed cases
        successful_cases = [r for r in self.results if r['extraction_successful']]
        failed_cases = [r for r in self.results if r['extraction_attempted'] and not r['extraction_successful']]
        
        if successful_cases:
            print(f"\n✅ SAMPLE SUCCESSFUL CASES:")
            for case in successful_cases[:5]:
                print(f"   {case['reference_number']}: {case['text_length']:,} chars via {case['extraction_method']}")
        
        if failed_cases:
            print(f"\n❌ SAMPLE FAILED CASES:")
            for case in failed_cases[:5]:
                print(f"   {case['reference_number']}: {case['failure_reason']}")
        
        return stats, csv_path

if __name__ == "__main__":
    analyzer = ExtractionResultsAnalyzer(
        csv_path="../preprocessing/processed_data/processed_legal_cases.csv"
    )
    analyzer.run_analysis() 