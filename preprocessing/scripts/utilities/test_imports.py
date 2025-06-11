#!/usr/bin/env python3
"""
Test Imports Utility
===================

Simple script to test if all imports are working correctly
for the preprocessing pipeline.
"""

import sys
from pathlib import Path

def test_main_pipeline_import():
    """Test importing the main preprocessing pipeline."""
    try:
        # Add main directory to path
        main_dir = Path(__file__).parent.parent / "main"
        sys.path.insert(0, str(main_dir))
        
        from llama_preprocessing_pipeline import LLaMAPreprocessingPipeline
        print("✅ Main pipeline import: SUCCESS")
        return True
    except ImportError as e:
        print(f"❌ Main pipeline import: FAILED - {e}")
        return False

def test_quality_monitor_import():
    """Test importing the quality monitor."""
    try:
        # Add analysis directory to path
        analysis_dir = Path(__file__).parent.parent / "analysis"
        sys.path.insert(0, str(analysis_dir))
        
        from preprocessing_quality_monitor import PreprocessingQualityMonitor
        print("✅ Quality monitor import: SUCCESS")
        return True
    except ImportError as e:
        print(f"❌ Quality monitor import: FAILED - {e}")
        return False

def test_dependencies():
    """Test key dependencies."""
    dependencies = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'matplotlib': 'matplotlib.pyplot',
        'seaborn': 'seaborn',
        'tqdm': 'tqdm',
        'pathlib': 'pathlib'
    }
    
    print("\n📦 Testing dependencies:")
    all_good = True
    
    for name, module in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {name}: Available")
        except ImportError:
            print(f"❌ {name}: Missing")
            all_good = False
    
    # Test optional dependencies
    print("\n📦 Testing optional dependencies:")
    optional_deps = {
        'transformers': 'transformers',
        'torch': 'torch'
    }
    
    for name, module in optional_deps.items():
        try:
            __import__(module)
            print(f"✅ {name}: Available")
        except ImportError:
            print(f"⚠️  {name}: Missing (optional)")
    
    return all_good

def main():
    """Run all tests."""
    print("🧪 Testing Preprocessing Pipeline Imports")
    print("=" * 50)
    
    # Test dependencies first
    deps_ok = test_dependencies()
    
    print("\n🔧 Testing module imports:")
    main_ok = test_main_pipeline_import()
    quality_ok = test_quality_monitor_import()
    
    print("\n📋 Summary:")
    if deps_ok and main_ok and quality_ok:
        print("🎉 All tests passed! Pipeline is ready to run.")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        if not deps_ok:
            print("   Install missing dependencies with: pip install pandas numpy matplotlib seaborn tqdm")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 