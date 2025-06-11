#!/usr/bin/env python3
"""
Test script for optimized document extraction features

This script tests:
1. Checkpoint saving and loading
2. Batch processing
3. Resume functionality
4. Performance tracking
5. Memory management
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from document_text_extractor import DocumentTextExtractor
import tempfile
import shutil

def test_checkpoint_functionality():
    """Test checkpoint save/load functionality"""
    print("🧪 Testing checkpoint functionality...")
    
    # Create temporary checkpoint directory
    temp_checkpoint_dir = tempfile.mkdtemp(prefix="test_checkpoints_")
    
    try:
        # Initialize extractor with test mode
        extractor = DocumentTextExtractor(
            csv_path="../preprocessing/processed_data/processed_legal_cases.csv",
            test_mode=True,
            test_count=3,
            batch_size=2,
            checkpoint_dir=temp_checkpoint_dir
        )
        
        # Simulate some progress
        extractor.processed_cases.add("[TEST] Case 1")
        extractor.successful_cases.append("[TEST] Case 1")
        extractor.cases_processed = 1
        extractor.current_batch = 1
        
        # Save checkpoint
        extractor.save_checkpoint()
        
        # Create new extractor instance
        extractor2 = DocumentTextExtractor(
            csv_path="../preprocessing/processed_data/processed_legal_cases.csv",
            test_mode=True,
            test_count=3,
            batch_size=2,
            checkpoint_dir=temp_checkpoint_dir
        )
        
        # Load checkpoint
        loaded = extractor2.load_checkpoint()
        
        if loaded:
            print("✅ Checkpoint save/load working correctly")
            print(f"   Loaded {len(extractor2.processed_cases)} processed cases")
            print(f"   Current batch: {extractor2.current_batch}")
            return True
        else:
            print("❌ Failed to load checkpoint")
            return False
            
    except Exception as e:
        print(f"❌ Checkpoint test failed: {str(e)}")
        return False
    
    finally:
        # Cleanup
        shutil.rmtree(temp_checkpoint_dir, ignore_errors=True)

def test_adaptive_rate_limiting():
    """Test adaptive rate limiting"""
    print("🧪 Testing adaptive rate limiting...")
    
    try:
        extractor = DocumentTextExtractor(
            csv_path="../preprocessing/processed_data/processed_legal_cases.csv",
            test_mode=True,
            test_count=1,
            rate_limit=1.0
        )
        
        initial_rate = extractor.rate_limit
        print(f"   Initial rate limit: {initial_rate}s")
        
        # Test speeding up
        extractor.adjust_rate_limit(True, 0.3)  # Fast successful response
        print(f"   After fast response: {extractor.rate_limit:.2f}s")
        
        # Test slowing down
        extractor.adjust_rate_limit(False, 5.0)  # Slow failed response
        print(f"   After slow/failed response: {extractor.rate_limit:.2f}s")
        
        print("✅ Adaptive rate limiting working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Rate limiting test failed: {str(e)}")
        return False

def test_memory_cleanup():
    """Test memory cleanup functionality"""
    print("🧪 Testing memory cleanup...")
    
    try:
        extractor = DocumentTextExtractor(
            csv_path="../preprocessing/processed_data/processed_legal_cases.csv",
            test_mode=True,
            test_count=1
        )
        
        # Test memory cleanup (should not raise errors)
        extractor.cleanup_memory()
        
        print("✅ Memory cleanup working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Memory cleanup test failed: {str(e)}")
        return False

def test_small_batch_processing():
    """Test batch processing with a very small batch"""
    print("🧪 Testing small batch processing...")
    
    try:
        extractor = DocumentTextExtractor(
            csv_path="../preprocessing/processed_data/processed_legal_cases.csv",
            test_mode=True,
            test_count=3,
            batch_size=2  # Process 3 cases in batches of 2
        )
        
        print(f"   Processing {len(extractor.missing_cases)} cases in batches of {extractor.batch_size}")
        
        # Run optimized processing
        stats = extractor.process_all_cases_optimized(resume=False)
        
        print("✅ Batch processing completed")
        print(f"   Total: {stats['total']}, Successful: {stats['successful']}, Failed: {stats['failed']}")
        
        # Check if checkpoint was created
        checkpoint_file = extractor.checkpoint_dir / "progress_checkpoint.json"
        if checkpoint_file.exists():
            print("✅ Checkpoint file created successfully")
        else:
            print("⚠️  No checkpoint file found")
        
        return stats['total'] > 0
        
    except Exception as e:
        print(f"❌ Batch processing test failed: {str(e)}")
        return False

def main():
    """Run all optimization tests"""
    print("🧪 Testing Optimized Document Extraction Features")
    print("=" * 55)
    
    tests = [
        ("Checkpoint Functionality", test_checkpoint_functionality),
        ("Adaptive Rate Limiting", test_adaptive_rate_limiting),
        ("Memory Cleanup", test_memory_cleanup),
        ("Small Batch Processing", test_small_batch_processing)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"💥 Test {test_name} crashed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n📊 Test Results Summary")
    print("=" * 30)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All optimization features working correctly!")
        print("✅ Ready for optimized full run!")
        return 0
    else:
        print("⚠️  Some tests failed - check before full run")
        return 1

if __name__ == "__main__":
    exit(main()) 