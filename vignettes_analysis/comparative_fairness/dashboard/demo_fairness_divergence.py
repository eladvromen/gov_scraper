#!/usr/bin/env python3
"""
Fairness Divergence Analysis Demo
Demonstration of the modular architecture for SP-based fairness divergence analysis
"""

import sys
from pathlib import Path
import logging

# Add paths for imports
sys.path.append(str(Path(__file__).parent / "03_fairness_metrics"))
sys.path.append(str(Path(__file__).parent / "05_divergence_analysis"))
sys.path.append(str(Path(__file__).parent / "utils"))

from fairness_divergence_analysis import FairnessDivergenceAnalyzer

def demo_sp_vector_extraction():
    """Demonstrate SP vector extraction capabilities"""
    print("=" * 80)
    print("DEMO: Statistical Parity Vector Extraction")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = FairnessDivergenceAnalyzer()
    
    # Extract SP vectors
    vectors_data = analyzer.extract_sp_vectors()
    
    print(f"\n✅ Successfully extracted SP vectors:")
    print(f"   📊 Total vector types: {len(vectors_data['vectors'])}")
    print(f"   📏 Vector dimensions: {len(list(vectors_data['vectors'].values())[0])}")
    print(f"   🏷️  Comparison labels: {len(vectors_data['labels'])}")
    
    # Show available vector types
    print(f"\n📋 Available vector types:")
    for vector_name in vectors_data['vectors'].keys():
        print(f"   • {vector_name}")
    
    # Show data summary
    info = vectors_data['dataframe_info']
    print(f"\n📈 Dataset summary:")
    print(f"   • Total comparisons: {info['total_comparisons']}")
    print(f"   • Protected attributes: {info['protected_attributes']}")
    print(f"   • Topics: {info['topics']}")
    
    return vectors_data

def demo_magnitude_divergence(analyzer):
    """Demonstrate magnitude divergence analysis"""
    print("\n" + "=" * 80)
    print("DEMO: SP Magnitude Divergence Analysis")
    print("=" * 80)
    
    # Analyze magnitude divergence
    magnitude_results = analyzer.analyze_magnitude_divergence()
    
    metrics = magnitude_results['divergence_metrics']
    print(f"\n✅ SP Magnitude Divergence Results:")
    print(f"   🎯 Cosine Similarity: {metrics.get('cosine_similarity', 'N/A'):.4f}")
    print(f"   📐 Cosine Distance: {metrics.get('cosine_distance', 'N/A'):.4f}")
    
    if 'pearson_correlation' in metrics:
        print(f"   🔗 Pearson Correlation: {metrics['pearson_correlation']:.4f}")
        if metrics.get('pearson_p_value', 1) < 0.05:
            print(f"   ⚡ Statistically significant correlation (p={metrics['pearson_p_value']:.6f})")
        else:
            print(f"   💭 Not statistically significant (p={metrics.get('pearson_p_value', 1):.6f})")
    
    # Interpretation
    cos_sim = metrics.get('cosine_similarity', 0)
    if cos_sim > 0.8:
        interpretation = "🟢 High similarity (low divergence)"
    elif cos_sim > 0.6:
        interpretation = "🟡 Moderate similarity"
    elif cos_sim > 0.0:
        interpretation = "🟠 Low similarity (high divergence)"
    else:
        interpretation = "🔴 Opposite bias patterns"
    
    print(f"   📝 Interpretation: {interpretation}")
    
    return magnitude_results

def demo_significance_divergence(analyzer):
    """Demonstrate significance divergence analysis"""
    print("\n" + "=" * 80)
    print("DEMO: SP Significance Divergence Analysis")
    print("=" * 80)
    
    # Analyze significance divergence
    significance_results = analyzer.analyze_significance_divergence()
    
    sig_stats = significance_results['significance_statistics']
    print(f"\n✅ SP Significance Divergence Results:")
    print(f"   🎭 Pattern Agreement Rate: {sig_stats['pattern_agreement_rate']:.2%}")
    print(f"   🔄 Pattern Change Rate: {sig_stats['pattern_change_rate']:.2%}")
    
    print(f"\n📊 Significance Rates:")
    print(f"   📅 Pre-Brexit: {sig_stats['vector1_significance_rate']:.2%}")
    print(f"   📅 Post-Brexit: {sig_stats['vector2_significance_rate']:.2%}")
    
    # Transitions
    if 'significance_transitions' in significance_results:
        transitions = significance_results['significance_transitions']
        print(f"\n🔄 Significance Transitions:")
        print(f"   ⬆️  Gained significance: {transitions['gained_significance']}")
        print(f"   ⬇️  Lost significance: {transitions['lost_significance']}")
        print(f"   ✅ Remained significant: {transitions['remained_significant']}")
        print(f"   ❌ Remained non-significant: {transitions['remained_non_significant']}")
    
    # Interpretation
    agreement_rate = sig_stats['pattern_agreement_rate']
    if agreement_rate > 0.9:
        interpretation = "🟢 Very consistent significance patterns"
    elif agreement_rate > 0.75:
        interpretation = "🟡 Moderately consistent patterns"
    elif agreement_rate > 0.5:
        interpretation = "🟠 Somewhat inconsistent patterns"
    else:
        interpretation = "🔴 Highly inconsistent patterns"
    
    print(f"   📝 Interpretation: {interpretation}")
    
    return significance_results

def demo_stratified_analysis(analyzer):
    """Demonstrate stratified divergence analysis"""
    print("\n" + "=" * 80)
    print("DEMO: Stratified Divergence Analysis")
    print("=" * 80)
    
    # Analyze by protected attribute
    print("\n🏷️  Analyzing by Protected Attribute...")
    attr_results = analyzer.analyze_stratified_divergence('protected_attribute')
    
    print(f"\n✅ Stratified by Protected Attribute:")
    print(f"{'Attribute':<12} | {'Cos Sim':<8} | {'Cos Dist':<9} | {'Pat Agr':<8} | {'Comps':<6}")
    print("-" * 50)
    
    for attr, results in attr_results.items():
        mag_div = results.get('magnitude_divergence', {})
        sig_div = results.get('significance_divergence', {})
        
        cos_sim = mag_div.get('divergence_metrics', {}).get('cosine_similarity', 0)
        cos_dist = mag_div.get('divergence_metrics', {}).get('cosine_distance', 0)
        pattern_agr = sig_div.get('significance_statistics', {}).get('pattern_agreement_rate', 0) if sig_div else 0
        comparisons = results.get('stratum_info', {}).get('comparisons', 0)
        
        print(f"{attr:<12} | {cos_sim:<8.4f} | {cos_dist:<9.4f} | {pattern_agr:<8.2%} | {comparisons:<6}")
    
    return attr_results

def demo_comprehensive_summary(analyzer):
    """Demonstrate comprehensive summary generation"""
    print("\n" + "=" * 80)
    print("DEMO: Comprehensive Summary Report")
    print("=" * 80)
    
    # Generate summary
    summary = analyzer.generate_summary_report()
    
    # Show first part of summary
    summary_lines = summary.split('\n')
    preview_lines = summary_lines[:50]  # Show first 50 lines
    
    print("\n📄 Summary Report Preview (first 50 lines):")
    print("─" * 80)
    for line in preview_lines:
        print(line)
    
    if len(summary_lines) > 50:
        print(f"\n... and {len(summary_lines) - 50} more lines in the full report")
    
    return summary

def run_comprehensive_demo():
    """Run the complete modular framework demonstration"""
    print("🚀 FAIRNESS DIVERGENCE ANALYSIS - MODULAR FRAMEWORK DEMO")
    print("=" * 80)
    print("Demonstrating the modular architecture for SP-based fairness analysis")
    print("=" * 80)
    
    try:
        # Initialize the analyzer
        print("\n🔧 Initializing Fairness Divergence Analyzer...")
        analyzer = FairnessDivergenceAnalyzer()
        print("✅ Analyzer initialized successfully!")
        
        # Demo 1: SP Vector Extraction
        vectors_data = demo_sp_vector_extraction()
        
        # Demo 2: Magnitude Divergence
        magnitude_results = demo_magnitude_divergence(analyzer)
        
        # Demo 3: Significance Divergence
        significance_results = demo_significance_divergence(analyzer)
        
        # Demo 4: Stratified Analysis
        stratified_results = demo_stratified_analysis(analyzer)
        
        # Demo 5: Comprehensive Summary
        summary = demo_comprehensive_summary(analyzer)
        
        # Final summary
        print("\n" + "=" * 80)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        print("\n✅ Demonstrated capabilities:")
        print("   • Modular SP vector extraction from fairness dataframes")
        print("   • Comprehensive magnitude divergence analysis (cosine similarity)")
        print("   • Significance pattern divergence analysis")
        print("   • Stratified analysis by protected attributes")
        print("   • Automated summary report generation")
        print("   • Configurable and extensible architecture")
        
        print("\n🏗️  Architecture highlights:")
        print("   • 03_fairness_metrics/: Fairness-specific processing")
        print("   • 05_divergence_analysis/: Generic divergence calculations")
        print("   • utils/: Shared vector manipulation utilities")
        print("   • config/: Centralized configuration management")
        
        print("\n💡 Next steps:")
        print("   • Save results with analyzer.save_results()")
        print("   • Integrate with normative divergence analysis")
        print("   • Add bootstrap confidence intervals")
        print("   • Create visualizations")
        
        return {
            'vectors_data': vectors_data,
            'magnitude_results': magnitude_results,
            'significance_results': significance_results,
            'stratified_results': stratified_results,
            'summary': summary,
            'analyzer': analyzer
        }
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        print("\n🔍 Troubleshooting tips:")
        print("   • Ensure fairness dataframe exists at configured path")
        print("   • Check that all required columns are present")
        print("   • Verify file permissions for output directories")
        raise

if __name__ == "__main__":
    # Setup logging for demo
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run the comprehensive demo
    demo_results = run_comprehensive_demo()
    
    print("\n" + "="*80)
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\n✅ Demonstrated capabilities:")
    print("   • Modular SP vector extraction from fairness dataframes")
    print("   • Comprehensive magnitude divergence analysis (cosine similarity)")
    print("   • Significance pattern divergence analysis")
    print("   • Stratified analysis by protected attributes")
    print("   • Automated summary report generation")
    print("   • Configurable and extensible architecture")
    
    print("\n🏗️  Architecture highlights:")
    print("   • 03_fairness_metrics/: Fairness-specific processing")
    print("   • 05_divergence_analysis/: Generic divergence calculations")
    print("   • utils/: Shared vector manipulation utilities")
    print("   • config/: Centralized configuration management")
    
    print("\n💡 Next steps:")
    print("   • Save results with analyzer.save_results()")
    print("   • Integrate with normative divergence analysis")
    print("   • Add bootstrap confidence intervals")
    print("   • Create visualizations")
    
    print("\n🔄 SAVING ANALYSIS RESULTS...")
    demo_results['analyzer'].save_results()
    print("✅ Results saved successfully!") 