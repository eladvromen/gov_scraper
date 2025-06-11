#!/usr/bin/env python3
import pandas as pd
import numpy as np

def investigate_metadata():
    print("🔍 INVESTIGATING METADATA ISSUE")
    print("="*50)
    
    # Load high confidence predictions
    df = pd.read_parquet('../results/predictions/high_confidence_predictions.parquet')
    
    print(f"Total high confidence predictions: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Check case_year column
    if 'case_year' in df.columns:
        print("CASE_YEAR ANALYSIS:")
        print("-" * 30)
        print(f"Total rows: {len(df):,}")
        print(f"Non-null case_year: {df['case_year'].notna().sum():,}")
        print(f"Null case_year: {df['case_year'].isna().sum():,}")
        print(f"Null percentage: {df['case_year'].isna().sum() / len(df) * 100:.1f}%")
        print()
        
        # Show unique values
        print("Case year value distribution:")
        print(df['case_year'].value_counts().sort_index().head(20))
        print()
        
        # Check data types
        print(f"case_year dtype: {df['case_year'].dtype}")
        print(f"case_year describe:")
        print(df['case_year'].describe())
        print()
        
        # Show some sample values
        print("Sample case_year values:")
        sample_years = df['case_year'].dropna().head(10)
        for i, year in enumerate(sample_years):
            print(f"  {i+1}: {year} (type: {type(year)})")
        
    else:
        print("❌ case_year column not found!")
    
    print()
    
    # Check other date-related columns
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'year' in col.lower()]
    print(f"Date-related columns found: {date_columns}")
    
    for col in date_columns:
        print(f"\n{col}:")
        print(f"  Non-null: {df[col].notna().sum():,}")
        print(f"  Sample values: {df[col].dropna().head(3).tolist()}")

if __name__ == "__main__":
    investigate_metadata() 