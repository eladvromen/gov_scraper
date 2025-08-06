#!/usr/bin/env python3
"""
Quick test of the Topic Deep Dive Analyzer with Work Intentions data
"""

import pandas as pd
from pathlib import Path
import sys

def test_data_availability():
    """Test if required data files are available"""
    
    print("🔍 Testing Data Availability for Work Intentions Analysis")
    print("=" * 55)
    
    # Define base paths
    base_dir = Path("..").resolve()
    
    # Required data files
    required_files = {
        "Fairness Comparisons": base_dir / "vector_drift" / "deduplicated_fairness_comparisons.csv",
        "Pattern Transitions": base_dir / "country_analysis" / "detailed_pattern_transitions.csv", 
        "Country Significant": base_dir / "country_analysis" / "significant_country_comparisons_fdr.csv",
        "Topic Attribute Data": base_dir / "visualizations" / "topic_attribute_analysis_data.csv",
        "Gender Bias": base_dir / "gender_analysis" / "persistent_gender_bias_detailed.csv"
    }
    
    all_available = True
    
    for name, filepath in required_files.items():
        if filepath.exists():
            print(f"✅ {name}: {filepath.name}")
            
            # Check for work intentions data
            if filepath.suffix == '.csv':
                try:
                    df = pd.read_csv(filepath)
                    
                    # Look for work intentions content
                    work_patterns = 0
                    for col in df.columns:
                        if df[col].dtype == 'object':  # Text columns
                            work_matches = df[col].astype(str).str.contains(
                                'Intentions regarding work', case=False, na=False
                            ).sum()
                            work_patterns += work_matches
                    
                    if work_patterns > 0:
                        print(f"   📊 Contains {work_patterns} work intention patterns")
                    else:
                        print(f"   ⚠️  No work intention patterns found")
                        
                except Exception as e:
                    print(f"   ❌ Error reading file: {e}")
                    
        else:
            print(f"❌ {name}: Not found at {filepath}")
            all_available = False
    
    print("\n" + "=" * 55)
    
    if all_available:
        print("✅ All required data files are available!")
        print("🚀 Ready to run Work Intentions Deep Dive Analysis")
        
        print("\n📋 To run the analysis:")
        print("python run_work_intentions_analysis.py")
        print("\nOr use the general tool:")
        print('python topic_analyzer.py --topic "Intentions regarding work"')
        
    else:
        print("❌ Some data files are missing. Please check file paths.")
        return False
    
    return all_available

def preview_work_intentions_data():
    """Preview work intentions data to show what will be analyzed"""
    
    print("\n📊 PREVIEW: Work Intentions Data")
    print("-" * 40)
    
    # Load and filter main dataset
    try:
        base_dir = Path("..").resolve()
        df = pd.read_csv(base_dir / "vector_drift" / "deduplicated_fairness_comparisons.csv")
        
        # Filter for work intentions
        work_data = df[df['comparison_label'].str.contains(
            'Intentions regarding work', case=False, na=False
        )]
        
        print(f"📈 Found {len(work_data)} work intention comparisons")
        
        if len(work_data) > 0:
            print("\n🔍 Sample comparisons:")
            for i, row in work_data.head(3).iterrows():
                print(f"• {row['comparison_label']}")
                print(f"  Pre: {row['pre_brexit_sp_magnitude']:.4f} | Post: {row['post_brexit_sp_magnitude']:.4f}")
                print(f"  Pre-sig: {row['pre_brexit_sp_significance']} | Post-sig: {row['post_brexit_sp_significance']}")
                print()
        
        # Check pattern transitions
        try:
            transitions_df = pd.read_csv(base_dir / "country_analysis" / "detailed_pattern_transitions.csv")
            work_transitions = transitions_df[transitions_df['topic'].str.contains(
                'Intentions regarding work', case=False, na=False
            )]
            
            if len(work_transitions) > 0:
                print(f"🔄 Pattern Transitions: {len(work_transitions)} found")
                pattern_counts = work_transitions['pattern_type'].value_counts()
                for pattern_type, count in pattern_counts.items():
                    print(f"  • {pattern_type}: {count}")
                    
                print(f"\n📊 Net Change: {pattern_counts.get('NEWLY_EMERGED', 0) - pattern_counts.get('DISAPPEARED', 0)}")
        except:
            print("⚠️  Pattern transitions data not available for preview")
            
    except Exception as e:
        print(f"❌ Error previewing data: {e}")

if __name__ == "__main__":
    # Test data availability
    available = test_data_availability()
    
    if available:
        # Preview work intentions data
        preview_work_intentions_data()
        
        print("\n" + "=" * 55)
        print("🎯 Ready for Work Intentions Deep Dive Analysis!")
        print("Run: python run_work_intentions_analysis.py")
        print("=" * 55)
    else:
        sys.exit(1) 