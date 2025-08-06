#!/usr/bin/env python3
"""
🎯 Work Intentions Deep Dive Analysis

Specific analysis script for "Intentions regarding work" topic
as requested in the deep dive analysis framework.

This demonstrates the interpretative strength of the deep dive approach
by examining granular bias patterns across all protected attributes.

Usage:
    python run_work_intentions_analysis.py
"""

import sys
from pathlib import Path
from topic_analyzer import TopicDeepDiveAnalyzer, TopicAnalysisConfig

def main():
    """Run comprehensive Work Intentions analysis"""
    
    print("🎯 WORK INTENTIONS DEEP DIVE ANALYSIS")
    print("=" * 50)
    print("Demonstrating interpretative strength of topic-specific bias analysis")
    print("Examining: Changes in significant bias patterns across completions & attributes")
    print("=" * 50)
    
    # Configuration for Work Intentions analysis
    config = TopicAnalysisConfig(
        topic_filter="Intentions regarding work",
        base_data_dir=Path("../../outputs").resolve(),  # Points to fairness/outputs
        output_dir=Path("work_intentions_deep_dive"),
        include_subtopics=True,
        fdr_correction=True,
        min_magnitude_threshold=0.0
    )
    
    # Initialize analyzer
    analyzer = TopicDeepDiveAnalyzer(config)
    
    # Run comprehensive analysis
    print("\n🚀 Starting Work Intentions Analysis...")
    success = analyzer.run_full_analysis()
    
    if success:
        print("\n" + "=" * 50)
        print("✅ WORK INTENTIONS ANALYSIS COMPLETE")
        print("=" * 50)
        print("📁 Results saved to: work_intentions_deep_dive/")
        print("\n📊 Key Findings Expected Based on Prior Analysis:")
        print("• Work Intentions Net Change: -1 pattern")
        print("• Pattern Distribution: 4 disappeared, 3 emerged")
        print("• Major Country Changes:")
        print("  - Nigeria: Lost 2 significant bias patterns")
        print("  - Myanmar: Gained 2 significant bias patterns")
        print("• Cross-Attribute Impact: Country (strongest), Religion, Gender, Age")
        print("\n📈 Generated Visualizations:")
        print("• Pattern transition overview")
        print("• Cross-attribute comparison")
        print("• Country-level impact heatmap")
        print("• Bias magnitude distribution")
        print("\n📄 Comprehensive Report:")
        print("• Executive summary with key insights")
        print("• Detailed pattern transition analysis")
        print("• Sub-topic granular breakdown")
        print("• Methodology and interpretation guidelines")
        print("=" * 50)
        
        # Show specific file outputs
        output_dir = config.output_dir
        if output_dir.exists():
            print(f"\n📁 Generated Files in {output_dir}:")
            for file in sorted(output_dir.glob("*")):
                if file.is_file():
                    print(f"  • {file.name}")
        
    else:
        print("\n❌ Analysis failed. Please check data paths and requirements.")
        return 1
    
    return 0

def demonstrate_configurability():
    """Demonstrate how the tool can be reconfigured for other topics"""
    
    print("\n🔧 TOOL CONFIGURABILITY DEMONSTRATION")
    print("-" * 40)
    print("This same tool can analyze ANY topic by changing the topic_filter:")
    print()
    
    example_topics = [
        "Intentions regarding education",
        "Asylum seeker circumstances", 
        "Nature of persecution",
        "Financial stability",
        "Firm settlement"
    ]
    
    for topic in example_topics:
        print(f"python topic_analyzer.py --topic \"{topic}\"")
    
    print("\n🎛️ Advanced Configuration Options:")
    print("• --output-dir custom_directory")
    print("• --base-dir /path/to/data")
    print("• --include-subtopics (granular analysis)")
    print("• Custom threshold and correction settings")
    
    print("\n💡 Research Applications:")
    print("• Algorithmic fairness assessment")
    print("• Policy impact evaluation")
    print("• Bias pattern surveillance")
    print("• Discrimination intervention targeting")

if __name__ == "__main__":
    # Run main analysis
    result = main()
    
    # Demonstrate configurability
    demonstrate_configurability()
    
    sys.exit(result) 