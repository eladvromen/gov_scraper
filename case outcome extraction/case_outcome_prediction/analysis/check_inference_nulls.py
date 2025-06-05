#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
"""
Check null values in inference dataset text columns
"""

import pandas as pd
import numpy as np

def analyze_inference_nulls():
    # Load the original inference data
    df = pd.read_parquet('/data/shil6369/gov_scraper/data/processed_legal_cases.parquet')
    
    print('INFERENCE DATASET NULL ANALYSIS')
    print('='*50)
    print(f'Total cases: {len(df):,}')
    print()
    
    # Check both text columns
    print('NULL VALUES COMPARISON:')
    print('-'*30)
    
    if 'decision_text_last_section' in df.columns:
        null_last = df['decision_text_last_section'].isna().sum()
        empty_last = (df['decision_text_last_section'].fillna('').str.strip() == '').sum()
        valid_last = len(df) - null_last - empty_last
        print(f'decision_text_last_section:')
        print(f'  NULL values: {null_last:,}')
        print(f'  Empty strings: {empty_last:,}')
        print(f'  Total unusable: {null_last + empty_last:,}')
        print(f'  Valid cases: {valid_last:,}')
        print()
    
    if 'decision_text_cleaned' in df.columns:
        null_cleaned = df['decision_text_cleaned'].isna().sum()
        empty_cleaned = (df['decision_text_cleaned'].fillna('').str.strip() == '').sum()
        valid_cleaned = len(df) - null_cleaned - empty_cleaned
        print(f'decision_text_cleaned:')
        print(f'  NULL values: {null_cleaned:,}')
        print(f'  Empty strings: {empty_cleaned:,}')
        print(f'  Total unusable: {null_cleaned + empty_cleaned:,}')
        print(f'  Valid cases: {valid_cleaned:,}')
        print()
    
    # Check overlap of nulls
    if 'decision_text_last_section' in df.columns and 'decision_text_cleaned' in df.columns:
        both_null = (df['decision_text_last_section'].isna() & df['decision_text_cleaned'].isna()).sum()
        last_null_cleaned_good = (
            df['decision_text_last_section'].isna() & 
            df['decision_text_cleaned'].notna() & 
            (df['decision_text_cleaned'].str.strip() != '')
        ).sum()
        
        # Combined strategy potential
        combined_valid = (
            (df['decision_text_last_section'].notna() & (df['decision_text_last_section'].str.strip() != '')) |
            (df['decision_text_cleaned'].notna() & (df['decision_text_cleaned'].str.strip() != ''))
        ).sum()
        
        print('OVERLAP ANALYSIS:')
        print('-'*30)
        print(f'Both columns NULL/empty: {both_null:,}')
        print(f'Last section NULL but cleaned text available: {last_null_cleaned_good:,}')
        print(f'Total recoverable with fallback strategy: {combined_valid:,}')
        print(f'Potential improvement: +{last_null_cleaned_good:,} cases')
        print()
    
    # Show text length comparison for non-null cases
    if 'decision_text_last_section' in df.columns and 'decision_text_cleaned' in df.columns:
        print('TEXT LENGTH COMPARISON (non-null cases):')
        print('-'*30)
        
        last_valid = df[df['decision_text_last_section'].notna() & (df['decision_text_last_section'].str.strip() != '')]
        cleaned_valid = df[df['decision_text_cleaned'].notna() & (df['decision_text_cleaned'].str.strip() != '')]
        
        if len(last_valid) > 0:
            last_lengths = last_valid['decision_text_last_section'].str.len()
            print(f'decision_text_last_section (n={len(last_valid):,}):')
            print(f'  Mean length: {last_lengths.mean():.0f} chars')
            print(f'  Median length: {last_lengths.median():.0f} chars')
            print()
        
        if len(cleaned_valid) > 0:
            cleaned_lengths = cleaned_valid['decision_text_cleaned'].str.len()
            print(f'decision_text_cleaned (n={len(cleaned_valid):,}):')
            print(f'  Mean length: {cleaned_lengths.mean():.0f} chars')
            print(f'  Median length: {cleaned_lengths.median():.0f} chars')
            print()
    
    # Show available columns
    print('AVAILABLE TEXT-RELATED COLUMNS:')
    print('-'*30)
    text_cols = [col for col in df.columns if 'text' in col.lower() or 'decision' in col.lower()]
    for col in text_cols:
        print(f'  {col}')
    
    return df

if __name__ == "__main__":
    df = analyze_inference_nulls() 