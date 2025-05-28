import json
import pandas as pd
from preprocess_legal_cases import LegalCasePreprocessor
from pathlib import Path
import sys
import os

# Add parent directory to path to access the data
sys.path.append('..')

def test_improved_cleaning():
    """Test the improved text cleaning on sample files."""
    
    # Test with a small subset first
    json_dir = "data/json"
    output_dir = "preprocessing/test_output"
    
    # Initialize preprocessor
    preprocessor = LegalCasePreprocessor(json_dir, output_dir)
    
    # Test on the same files to compare improvement
    sample_files = [
        "[2000] UKIAT 5.json",
        "[2002] UKIAT 1869.json", 
        "[2001] UKIAT 4.json"
    ]
    
    print("Testing IMPROVED preprocessing on sample files...")
    print("=" * 60)
    
    processed_cases = []
    
    for filename in sample_files:
        file_path = Path(json_dir) / filename
        if file_path.exists():
            print(f"\n📄 Processing: {filename}")
            result = preprocessor.process_single_file(file_path)
            if result:
                processed_cases.append(result)
                print(f"✅ Success")
                print(f"   📊 Text length: {result['decision_text_length']:,} chars")
                print(f"   📝 Word count: {result['decision_text_word_count']:,} words")
                print(f"   📄 Last section: {result['last_section_length']:,} chars (~{result['last_section_token_count']:,} tokens)")
                print(f"   🌍 Country: {result.get('country', 'N/A')}")
                print(f"   📅 Case year: {result.get('case_year', 'N/A')}")
                
                # Show cleaned text beginning (more substantial now)
                cleaned_text = result['decision_text_cleaned']
                if cleaned_text:
                    # Show first 500 characters of cleaned text
                    snippet_length = 500
                    snippet = cleaned_text[:snippet_length]
                    if len(cleaned_text) > snippet_length:
                        snippet += "..."
                    
                    print(f"   📖 Clean text preview:")
                    print(f"   {'-' * 50}")
                    # Indent each line for better display
                    for line in snippet.split('\n')[:8]:  # Show first 8 lines
                        if line.strip():
                            print(f"   {line}")
                    if len(cleaned_text.split('\n')) > 8:
                        print(f"   ... (continues for {len(cleaned_text.split('\n')) - 8} more lines)")
                    print(f"   {'-' * 50}")
                    
                    # Show last section preview
                    last_section = result['decision_text_last_section']
                    if last_section:
                        print(f"   🎯 Last section preview (for classification):")
                        print(f"   {'-' * 50}")
                        last_snippet = last_section[:400]
                        if len(last_section) > 400:
                            last_snippet += "..."
                        for line in last_snippet.split('\n')[:6]:  # Show first 6 lines
                            if line.strip():
                                print(f"   {line}")
                        print(f"   {'-' * 50}")
            else:
                print(f"❌ Failed to process {filename}")
    
    if processed_cases:
        # Create DataFrame
        df = pd.DataFrame(processed_cases)
        print(f"\n{'=' * 60}")
        print("📈 IMPROVED PROCESSING RESULTS")
        print(f"{'=' * 60}")
        print(f"✅ Successfully processed: {len(df)} files")
        print(f"📊 Columns created: {len(df.columns)}")
        
        # Show text statistics comparison
        print(f"\n📝 TEXT STATISTICS:")
        for _, row in df.iterrows():
            filename = row['source_file']
            print(f"   {filename}:")
            print(f"     📏 Length: {row['decision_text_length']:,} chars")
            print(f"     🔤 Words: {row['decision_text_word_count']:,}")
            print(f"     🎯 Last section: {row['last_section_length']:,} chars (~{row['last_section_token_count']:,} tokens)")
            print(f"     📦 File size: {row['file_size_kb']:.1f} KB")
        
        # Save sample results
        os.makedirs(output_dir, exist_ok=True)
        sample_output = Path(output_dir) / "improved_sample_cases.csv"
        df.to_csv(sample_output, index=False, encoding='utf-8')
        print(f"\n💾 Sample data saved to: {sample_output}")
        
        # Also save just the cleaned text for easy review
        text_output = Path(output_dir) / "cleaned_texts_sample.txt"
        with open(text_output, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                f.write(f"{'='*80}\n")
                f.write(f"FILE: {row['source_file']}\n")
                f.write(f"CASE: {row['reference_number']} ({row['country']})\n")
                f.write(f"{'='*80}\n")
                f.write(row['decision_text_cleaned'])
                f.write(f"\n\n{'='*80}\n\n")
        
        print(f"📄 Cleaned texts saved to: {text_output}")
        
        return df
    else:
        print("❌ No files were successfully processed!")
        return None

def compare_with_original():
    """Show what improvements were made compared to original version."""
    print("\n🔄 IMPROVEMENTS MADE:")
    print("=" * 50)
    print("✅ Better removal of administrative headers")
    print("✅ Removal of judge signatures and trailing content") 
    print("✅ Elimination of case reference artifacts")
    print("✅ Smarter detection of content start")
    print("✅ Removal of very short lines (likely artifacts)")
    print("✅ Better handling of tribunal administrative text")
    print("✅ Organized output into preprocessing/ directory")
    print("=" * 50)

if __name__ == "__main__":
    test_df = test_improved_cleaning()
    compare_with_original() 