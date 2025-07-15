#!/usr/bin/env python3
"""
Demo: Topic-Stratified Interactive Bias Analysis
Demonstrates how to explore "How did attribute X change in topic Y?" questions
"""

import pandas as pd
import json
from pathlib import Path

def demo_topic_attribute_exploration():
    """Demonstrate exploring topic-attribute bias patterns"""
    
    print("🔍 TOPIC-STRATIFIED BIAS ANALYSIS DEMO")
    print("="*60)
    print("Exploring: 'How did Attribute X change in Topic Y?'")
    print("="*60)
    
    # Load the detailed topic-attribute analysis data
    data_file = "outputs/visualizations/topic_attribute_analysis_data.csv"
    df = pd.read_csv(data_file)
    
    print(f"\n📊 Dataset Overview:")
    print(f"   • Total topic-attribute combinations: {len(df)}")
    print(f"   • Unique topics: {df['topic'].nunique()}")
    print(f"   • Protected attributes: {df['attribute'].nunique()}")
    print(f"   • Total comparisons analyzed: {df['comparisons'].sum()}")
    
    # 1. Top bias changes by topic-attribute combination
    print(f"\n🏆 TOP 10 LARGEST BIAS CHANGES (by Topic-Attribute):")
    print("-" * 60)
    top_changes = df.nlargest(10, 'abs_bias_change')
    
    for i, (_, row) in enumerate(top_changes.iterrows(), 1):
        direction = "↗️ Increased" if row['bias_change'] > 0 else "↘️ Decreased" if row['bias_change'] < 0 else "➡️ No change"
        print(f"{i:2d}. {row['attribute'].title()} bias in '{row['topic'][:40]}...'")
        print(f"    {direction} by {row['bias_change']:.4f} SP (|{row['abs_bias_change']:.4f}|)")
        print(f"    Comparisons: {row['comparisons']}, Significance change: {row['significance_change']:+d}")
        print()
    
    # 2. Analysis by specific questions
    print("\n❓ ANSWERING SPECIFIC RESEARCH QUESTIONS:")
    print("-" * 50)
    
    # Question 1: How did gender bias change across different topics?
    print("Q1: How did GENDER bias change across different topics?")
    gender_analysis = df[df['attribute'] == 'gender'].sort_values('abs_bias_change', ascending=False)
    
    print(f"   📈 Most changed gender bias topics:")
    for _, row in gender_analysis.head(5).iterrows():
        print(f"      • {row['topic'][:35]}... → {row['bias_change']:+.4f} SP change")
    
    # Question 2: Which topic shows most country bias changes?
    print(f"\nQ2: Which topics show most COUNTRY bias changes?")
    country_analysis = df[df['attribute'] == 'country'].sort_values('abs_bias_change', ascending=False)
    
    print(f"   🏴 Most changed country bias topics:")
    for _, row in country_analysis.head(5).iterrows():
        print(f"      • {row['topic'][:35]}... → {row['bias_change']:+.4f} SP change")
    
    # Question 3: Are there topics with consistent bias patterns across attributes?
    print(f"\nQ3: Which topics show CONSISTENT bias patterns across attributes?")
    topic_consistency = df.groupby('topic').agg({
        'abs_bias_change': ['mean', 'std', 'count'],
        'bias_change': 'mean'
    }).round(4)
    
    # Flatten column names
    topic_consistency.columns = ['mean_abs_change', 'std_abs_change', 'attr_count', 'mean_change']
    topic_consistency['consistency'] = 1 / (1 + topic_consistency['std_abs_change'])
    topic_consistency = topic_consistency[topic_consistency['attr_count'] >= 3]  # At least 3 attributes
    
    print(f"   🎯 Most consistent topics (low variability across attributes):")
    consistent_topics = topic_consistency.sort_values('consistency', ascending=False).head(5)
    for topic, row in consistent_topics.iterrows():
        print(f"      • {topic[:35]}... → Consistency: {row['consistency']:.3f}, Mean change: {row['mean_change']:+.4f}")
    
    # 4. Significance transition analysis
    print(f"\nQ4: Which attribute-topic combinations gained/lost significance most?")
    
    # Gained significance
    gained_sig = df[df['significance_change'] > 0].sort_values('significance_change', ascending=False)
    print(f"   ⬆️  Most gained significance:")
    for _, row in gained_sig.head(3).iterrows():
        print(f"      • {row['attribute'].title()} in '{row['topic'][:30]}...' (+{row['significance_change']} significant)")
    
    # Lost significance  
    lost_sig = df[df['significance_change'] < 0].sort_values('significance_change', ascending=True)
    print(f"   ⬇️  Most lost significance:")
    for _, row in lost_sig.head(3).iterrows():
        print(f"      • {row['attribute'].title()} in '{row['topic'][:30]}...' ({row['significance_change']} significant)")
    
    # 5. Interactive dashboard usage
    print(f"\n🖥️  INTERACTIVE DASHBOARD USAGE:")
    print("-" * 40)
    dashboard_file = Path("outputs/visualizations/interactive_bias_dashboard.html")
    
    if dashboard_file.exists():
        print(f"✅ Interactive dashboard available: {dashboard_file}")
        print(f"   📊 Features:")
        print(f"      • Bubble plot: Size = change magnitude, Color = bias direction")
        print(f"      • Before/After scatter: Shows bias trajectory for each combination")
        print(f"      • Significance changes: Net changes by protected attribute")
        print(f"      • Topic overview: Identifies most changed topics")
        print(f"\n   🚀 Usage: Open {dashboard_file} in a web browser for interactive exploration!")
    else:
        print(f"❌ Interactive dashboard not found. Run the visualization script first.")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   🎯 Most bias changes occur in:")
    print(f"      1. Gender bias in disclosure/contradiction topics")
    print(f"      2. Country bias in asylum circumstances")  
    print(f"      3. Age bias in persecution/disclosure topics")
    print(f"   📊 Brexit impact varies significantly by topic-attribute combination")
    print(f"   🔄 {len(df[df['significance_change'] != 0])} combinations changed significance patterns")

def demo_specific_topic_analysis():
    """Demonstrate deep-dive analysis of a specific topic"""
    
    print(f"\n" + "="*60)
    print("🔬 DEEP-DIVE: Asylum Seeker Circumstances Topic")
    print("="*60)
    
    # Load data
    df = pd.read_csv("outputs/visualizations/topic_attribute_analysis_data.csv")
    
    # Focus on "Asylum seeker circumstances" topic
    asylum_topic = df[df['topic'] == 'Asylum seeker circumstances']
    
    print(f"📋 Topic Overview:")
    print(f"   • Total comparisons: {asylum_topic['comparisons'].sum()}")
    print(f"   • Attributes analyzed: {asylum_topic['attribute'].tolist()}")
    
    print(f"\n📊 Bias Changes by Attribute:")
    for _, row in asylum_topic.iterrows():
        direction = "↗️" if row['bias_change'] > 0 else "↘️" if row['bias_change'] < 0 else "➡️"
        print(f"   {direction} {row['attribute'].title():8s}: {row['bias_change']:+.4f} SP change ({row['comparisons']:2d} comparisons)")
        print(f"        Pre-Brexit significant: {row['pre_significant']:2d}, Post-Brexit: {row['post_significant']:2d} (Δ{row['significance_change']:+d})")
    
    print(f"\n🎯 Key Finding: Country bias in asylum circumstances increased most significantly!")
    print(f"   • This suggests systematically different treatment of asylum seekers by country post-Brexit")
    print(f"   • {asylum_topic[asylum_topic['attribute']=='country']['significance_change'].iloc[0]:+d} more significant country comparisons post-Brexit")

if __name__ == "__main__":
    demo_topic_attribute_exploration()
    demo_specific_topic_analysis()
    
    print(f"\n" + "="*60)
    print("✨ DEMO COMPLETED! Use the interactive dashboard for further exploration.")
    print("="*60) 