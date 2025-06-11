#!/usr/bin/env python3
"""
Runner script for the document text extraction pipeline.

This script:
1. Extracts missing decision texts from downloadable documents
2. Updates existing JSON files
3. Re-runs preprocessing on updated files
4. Generates new CSV with same structure

Usage:
    python run_document_extraction.py --test          # Test mode (5 cases)
    python run_document_extraction.py --single       # Test single case
    python run_document_extraction.py --full         # Process all cases
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from document_text_extractor import DocumentTextExtractor
from preprocessing.preprocess_legal_cases import LegalCasePreprocessor

def setup_paths():
    """Setup and validate all required paths"""
    base_dir = Path(__file__).parent.parent
    
    paths = {
        'csv_path': base_dir / "preprocessing/processed_data/processed_legal_cases.csv",
        'json_dir': base_dir / "data/json",
        'output_dir': base_dir / "preprocessing/processed_data",
        'backup_dir': base_dir / "preprocessing/processed_data/backups"
    }
    
    # Validate required paths exist
    missing_paths = []
    for name, path in paths.items():
        if name in ['csv_path', 'json_dir'] and not path.exists():
            missing_paths.append(f"{name}: {path}")
    
    if missing_paths:
        print("❌ Missing required paths:")
        for path in missing_paths:
            print(f"   {path}")
        return None
    
    # Create output directories
    paths['backup_dir'].mkdir(parents=True, exist_ok=True)
    
    return paths

def backup_csv(csv_path: Path, backup_dir: Path):
    """Create a backup of the original CSV"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"processed_legal_cases_backup_{timestamp}.csv"
    backup_path = backup_dir / backup_name
    
    import shutil
    shutil.copy2(csv_path, backup_path)
    print(f"📦 Backup created: {backup_path}")
    return backup_path

def analyze_missing_cases(csv_path: Path, test_mode: bool = False, test_count: int = 5):
    """Analyze and report on missing cases"""
    print("🔍 Analyzing missing cases...")
    
    df = pd.read_csv(csv_path)
    
    # Find cases with missing decision text
    missing_text = (
        df['decision_text_cleaned'].isna() | 
        (df['decision_text_cleaned'] == '') |
        (df['decision_text_cleaned'].str.len() < 50)
    )
    
    missing_df = df[missing_text]
    
    # Apply test mode BEFORE checking URLs
    if test_mode:
        missing_df = missing_df.head(test_count)
        print(f"🧪 TEST MODE: Limited to first {len(missing_df)} cases")
    
    # Check for download URLs
    has_word = missing_df['word_url'].notna()
    has_pdf = missing_df['pdf_url'].notna()
    has_either = has_word | has_pdf
    
    print(f"📊 Missing text analysis:")
    print(f"   Total cases in CSV: {len(df):,}")
    if test_mode:
        print(f"   Cases missing decision text (total): {len(df[missing_text]):,}")
        print(f"   Cases missing decision text (test subset): {len(missing_df):,}")
    else:
        print(f"   Cases missing decision text: {len(missing_df):,}")
    print(f"   Cases with Word URL: {has_word.sum():,}")
    print(f"   Cases with PDF URL: {has_pdf.sum():,}")
    print(f"   Cases with either URL: {has_either.sum():,}")
    print(f"   Cases without any URL: {(~has_either).sum():,}")
    
    return has_either.sum()

def test_single_case(paths: dict):
    """Test extraction on a single case"""
    print("🧪 Testing single case extraction...")
    
    extractor = DocumentTextExtractor(
        csv_path=str(paths['csv_path']),
        json_dir=str(paths['json_dir']),
        test_mode=True,
        test_count=1
    )
    
    success = extractor.test_single_download()
    
    if success:
        print("✅ Single case test successful!")
        return True
    else:
        print("❌ Single case test failed!")
        return False

def run_extraction(paths: dict, test_mode: bool = False, test_count: int = 5, 
                   optimized: bool = False, batch_size: int = 500, resume: bool = True):
    """Run the document extraction process"""
    if test_mode:
        print(f"🧪 Running extraction in TEST MODE ({test_count} cases)")
        batch_size = test_count  # Use smaller batch for testing
    elif optimized:
        print(f"🚀 Running OPTIMIZED extraction with batch size {batch_size}")
    else:
        print("🚀 Running standard extraction...")
    
    extractor = DocumentTextExtractor(
        csv_path=str(paths['csv_path']),
        json_dir=str(paths['json_dir']),
        test_mode=test_mode,
        test_count=test_count,
        batch_size=batch_size,
        rate_limit=1.0,
        max_retries=3
    )
    
    # Use optimized processing if requested
    if optimized and not test_mode:
        stats = extractor.process_all_cases_optimized(resume=resume)
    else:
        stats = extractor.process_all_cases()
    
    # Get list of successfully updated cases for reprocessing
    successful_cases = []
    if hasattr(extractor, 'missing_cases'):
        for idx, case_row in extractor.missing_cases.iterrows():
            reference_number = case_row['reference_number']
            # Check if this case was successfully updated
            json_filename = reference_number.replace('/', '_') + '.json'
            json_path = extractor.json_dir / json_filename
            
            if json_path.exists():
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        case_data = json.load(f)
                    
                    # Check if it has our extraction metadata
                    if 'text_extraction_date' in case_data:
                        decision_text = case_data.get('decision_text', '')
                        if decision_text and len(decision_text.strip()) > 100:
                            successful_cases.append(reference_number)
                except:
                    pass
    
    stats['successful_cases'] = successful_cases
    
    print(f"\n📈 Extraction Results:")
    print(f"   Total cases processed: {stats['total']:,}")
    print(f"   Successful extractions: {stats['successful']:,}")
    print(f"   Failed extractions: {stats['failed']:,}")
    print(f"   Success rate: {stats['successful']/stats['total']*100:.1f}%" if stats['total'] > 0 else "   No cases processed")
    
    # Show performance stats for optimized runs
    if optimized and 'start_time' in stats:
        try:
            start_time = datetime.fromisoformat(stats['start_time'])
            end_time = datetime.fromisoformat(stats['end_time'])
            total_time = end_time - start_time
            rate = stats['successful'] / total_time.total_seconds() * 60  # cases per minute
            print(f"   Processing time: {total_time}")
            print(f"   Processing rate: {rate:.2f} cases/minute")
        except:
            pass
    
    return stats

def reprocess_updated_files(paths: dict, updated_cases: list = None):
    """Re-run preprocessing on updated JSON files"""
    if not updated_cases:
        print("⚠️  No updated cases provided - skipping reprocessing")
        return None, None
        
    print(f"♻️  Re-processing {len(updated_cases)} updated JSON files...")
    
    try:
        preprocessor = LegalCasePreprocessor(
            json_dir=str(paths['json_dir']),
            output_dir=str(paths['output_dir'])
        )
        
        # Process only the updated files
        updated_data = []
        for case_ref in updated_cases:
            json_filename = case_ref.replace('/', '_') + '.json'
            json_path = paths['json_dir'] / json_filename
            
            if json_path.exists():
                try:
                    case_data = preprocessor.process_single_file(json_path)
                    if case_data:
                        updated_data.append(case_data)
                        print(f"   ✅ Processed: {case_ref}")
                except Exception as e:
                    print(f"   ❌ Error processing {case_ref}: {str(e)}")
        
        if not updated_data:
            print("❌ No files were successfully processed")
            return None, None
        
        # Convert to DataFrame
        df_updated = pd.DataFrame(updated_data)
        
        # Load original CSV and update the relevant rows
        df_original = pd.read_csv(paths['csv_path'])
        
        # Update the original dataframe with new data
        for _, updated_row in df_updated.iterrows():
            ref_num = updated_row['reference_number']
            mask = df_original['reference_number'] == ref_num
            
            if mask.any():
                # Update the existing row
                for col in updated_row.index:
                    if col in df_original.columns:
                        df_original.loc[mask, col] = updated_row[col]
                print(f"   📝 Updated row for: {ref_num}")
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"processed_legal_cases_updated_{timestamp}.csv"
        output_path = paths['output_dir'] / new_filename
        
        df_original.to_csv(output_path, index=False)
        
        print(f"✅ Reprocessing complete!")
        print(f"📁 Updated CSV saved to: {output_path}")
        
        return df_original, output_path
        
    except Exception as e:
        print(f"❌ Error during reprocessing: {str(e)}")
        return None, None

def compare_before_after(original_csv: Path, updated_csv: Path):
    """Compare the before and after datasets"""
    if not updated_csv or not updated_csv.exists():
        print("⚠️  Cannot compare - updated CSV not available")
        return
    
    print("📊 Comparing before and after...")
    
    try:
        df_original = pd.read_csv(original_csv)
        df_updated = pd.read_csv(updated_csv)
        
        # Count cases with decision text
        orig_with_text = (~df_original['decision_text_cleaned'].isna() & 
                         (df_original['decision_text_cleaned'] != '') &
                         (df_original['decision_text_cleaned'].str.len() >= 50)).sum()
        
        updated_with_text = (~df_updated['decision_text_cleaned'].isna() & 
                           (df_updated['decision_text_cleaned'] != '') &
                           (df_updated['decision_text_cleaned'].str.len() >= 50)).sum()
        
        improvement = updated_with_text - orig_with_text
        
        print(f"📈 Comparison Results:")
        print(f"   Original cases with text: {orig_with_text:,}")
        print(f"   Updated cases with text: {updated_with_text:,}")
        print(f"   Improvement: +{improvement:,} cases")
        print(f"   Coverage improvement: +{improvement/len(df_updated)*100:.2f}%")
        
    except Exception as e:
        print(f"❌ Error comparing datasets: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Extract text from downloadable documents for missing legal cases")
    parser.add_argument('--test', action='store_true', help='Run in test mode (process 5 cases)')
    parser.add_argument('--single', action='store_true', help='Test single case only')
    parser.add_argument('--full', action='store_true', help='Process all missing cases')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze missing cases, don\'t extract')
    parser.add_argument('--test-count', type=int, default=5, help='Number of cases to process in test mode')
    parser.add_argument('--optimized', action='store_true', help='Run optimized extraction')
    parser.add_argument('--batch-size', type=int, default=500, help='Batch size for optimized extraction')
    parser.add_argument('--resume', action='store_true', help='Resume from last processed case')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.test, args.single, args.full, args.analyze_only]):
        print("❌ Please specify one of: --test, --single, --full, or --analyze-only")
        parser.print_help()
        return 1
    
    print("🏛️  UK Immigration Tribunal Document Text Extractor")
    print("=" * 60)
    
    # Setup paths
    paths = setup_paths()
    if not paths:
        return 1
    
    # Analyze missing cases
    missing_count = analyze_missing_cases(paths['csv_path'], args.test, args.test_count)
    
    if args.analyze_only:
        print("📋 Analysis complete. Use --test, --single, or --full to extract documents.")
        return 0
    
    if missing_count == 0:
        print("✅ No cases found with missing text and available download URLs!")
        return 0
    
    # Create backup
    backup_path = backup_csv(paths['csv_path'], paths['backup_dir'])
    
    try:
        if args.single:
            # Test single case
            success = test_single_case(paths)
            if not success:
                return 1
            print("✅ Single case test completed successfully!")
            
        elif args.test or args.full:
            # Run extraction
            stats = run_extraction(paths, args.test, args.test_count, args.optimized, args.batch_size, args.resume)
            
            if stats['successful'] == 0:
                print("❌ No successful extractions. Check logs for details.")
                return 1
            
            # Reprocess updated files
            updated_cases = []
            for case_ref in stats['successful_cases']:
                updated_cases.append(case_ref)
            updated_df, updated_csv_path = reprocess_updated_files(paths, updated_cases)
            
            if updated_csv_path:
                # Compare results
                compare_before_after(paths['csv_path'], updated_csv_path)
                
                print(f"\n🎉 Pipeline completed successfully!")
                print(f"📁 Original CSV: {paths['csv_path']}")
                print(f"📁 Updated CSV: {updated_csv_path}")
                print(f"📁 Backup CSV: {backup_path}")
                print(f"📁 Logs: logs/document_extractor_*.log")
            else:
                print("❌ Reprocessing failed. Check logs for details.")
                return 1
    
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 