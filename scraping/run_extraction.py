#!/usr/bin/env python3
"""
Simple script to run document text extraction with better progress bar visibility.
"""

from document_text_extractor import DocumentTextExtractor

def main():
    print("🔧 Initializing Document Text Extractor...")
    
    # Initialize extractor with production settings
    extractor = DocumentTextExtractor(
        csv_path="../preprocessing/processed_data/processed_legal_cases.csv",
        json_dir="../data/json",
        rate_limit=0.5,   # Faster initial rate limit
        max_retries=3,
        test_mode=False,  # Set to True for testing with small sample
        test_count=10,
        batch_size=50,    # Smaller batches for more frequent progress updates
        checkpoint_dir="checkpoints"
    )
    
    print("📋 Configuration:")
    print(f"   • CSV file: {extractor.csv_path}")
    print(f"   • JSON directory: {extractor.json_dir}")
    print(f"   • Batch size: {extractor.batch_size}")
    print(f"   • Rate limit: {extractor.rate_limit}s")
    print(f"   • Test mode: {extractor.test_mode}")
    
    # Optional: Toggle console verbosity if needed
    # extractor.set_console_verbosity(verbose=True)  # Uncomment for detailed console logs
    
    try:
        # Run the extraction process
        print("\n🏃 Starting extraction process...")
        stats = extractor.process_all_cases_optimized(resume=True)
        
        # Print final results
        print(f"\n{'='*60}")
        print("🎉 EXTRACTION COMPLETE!")
        print(f"📊 Final Statistics:")
        print(f"   • Total cases: {stats['total']:,}")
        print(f"   • Successful: {stats['successful']:,}")
        print(f"   • Failed: {stats['failed']:,}")
        print(f"   • Success rate: {stats['successful']/stats['total']*100:.1f}%")
        print(f"   • Start time: {stats['start_time']}")
        print(f"   • End time: {stats['end_time']}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        print("💾 Progress has been saved - you can resume later")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("📄 Check the log files for detailed error information")

if __name__ == "__main__":
    main() 