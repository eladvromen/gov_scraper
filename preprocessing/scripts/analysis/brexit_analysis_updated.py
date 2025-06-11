#!/usr/bin/env python3
"""
Updated Brexit Analysis with Properly Extracted Years
====================================================

Now using the extracted_year field for comprehensive temporal analysis.
"""

import pandas as pd
import numpy as np

def brexit_analysis_updated():
    # Load data with extracted years
    print("Loading data with extracted years...")
    df = pd.read_csv('preprocessing/processed_data/processed_legal_cases_with_years.csv', low_memory=False)
    
    # Filter for cases with text
    df_with_text = df[
        df['decision_text_cleaned'].notna() & 
        (df['decision_text_cleaned'] != '')
    ]
    
    print("=== COMPREHENSIVE BREXIT TEMPORAL ANALYSIS ===")
    print(f"Total cases: {len(df):,}")
    print(f"Cases with text: {len(df_with_text):,}")
    
    # Brexit timeframe analysis
    pre_brexit = df_with_text[
        (df_with_text['extracted_year'] >= 2013) & 
        (df_with_text['extracted_year'] <= 2016)
    ]
    
    post_brexit = df_with_text[
        (df_with_text['extracted_year'] >= 2018) & 
        (df_with_text['extracted_year'] <= 2025)
    ]
    
    excluded_2017 = df_with_text[df_with_text['extracted_year'] == 2017]
    pre_2013 = df_with_text[df_with_text['extracted_year'] < 2013]
    
    print(f"\n🟢 PRE-BREXIT CORPUS (2013-2016): {len(pre_brexit):,} cases")
    print(f"🔵 POST-BREXIT CORPUS (2018-2025): {len(post_brexit):,} cases")
    print(f"⚫ EXCLUDED 2017: {len(excluded_2017):,} cases")
    print(f"⚪ PRE-2013 (sparse): {len(pre_2013):,} cases")
    
    # Show year distribution for all years
    print(f"\n📊 FULL YEAR DISTRIBUTION:")
    year_dist = df_with_text['extracted_year'].value_counts().sort_index()
    for year, count in year_dist.items():
        if year >= 2010:  # Focus on recent years
            if 2013 <= year <= 2016:
                emoji = "🟢"  # Pre-Brexit
            elif year == 2017:
                emoji = "⚫"  # Excluded
            elif 2018 <= year <= 2025:
                emoji = "🔵"  # Post-Brexit
            else:
                emoji = "⚪"  # Other
            print(f"   {emoji} {int(year)}: {count:,} cases")
    
    # Calculate corpus statistics
    if len(pre_brexit) > 0:
        print(f"\n🟢 PRE-BREXIT STATISTICS:")
        print(f"   Cases: {len(pre_brexit):,}")
        print(f"   Total characters: {pre_brexit['decision_text_length'].sum():,}")
        print(f"   Total words: {pre_brexit['decision_text_word_count'].sum():,}")
        print(f"   Avg case length: {pre_brexit['decision_text_length'].mean():.0f} chars")
    
    if len(post_brexit) > 0:
        print(f"\n🔵 POST-BREXIT STATISTICS:")
        print(f"   Cases: {len(post_brexit):,}")
        print(f"   Total characters: {post_brexit['decision_text_length'].sum():,}")
        print(f"   Total words: {post_brexit['decision_text_word_count'].sum():,}")
        print(f"   Avg case length: {post_brexit['decision_text_length'].mean():.0f} chars")
    
    # Corpus balance
    if len(pre_brexit) > 0 and len(post_brexit) > 0:
        ratio = len(pre_brexit) / len(post_brexit)
        print(f"\n📊 PRE/POST RATIO: {ratio:.2f}")
        if ratio > 1.5:
            print("   ⚠️  Pre-Brexit corpus significantly larger")
        elif ratio < 0.67:
            print("   ⚠️  Post-Brexit corpus significantly larger")
        else:
            print("   ✅ Reasonably balanced corpora")
    
    return df_with_text, pre_brexit, post_brexit

if __name__ == "__main__":
    brexit_analysis_updated() 