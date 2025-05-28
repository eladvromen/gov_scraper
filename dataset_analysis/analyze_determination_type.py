import os
import json
import re
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def analyze_determination_types_by_period(json_dir="data/json", split_year=2017):
    """
    Analyzes determination types in scraped UTIAC decisions, split by time period
    
    Args:
        json_dir: Directory containing JSON files (default: 'data/json')
        split_year: Year to split the data (before vs on/after) (default: 2017)
        
    Returns:
        DataFrame with counts and percentages of each determination type by period
    """
    # Initialize counters for both periods
    counts_before = Counter()
    counts_after = Counter()
    records = []
    
    # Known determination types based on your data
    type_patterns = {
        'PA': re.compile(r'(?:^|\s)PA[/_\s]?\d+'),  # Protection Appeal/Asylum
        'UI': re.compile(r'(?:^|\s)UI[-/_\s]?\d+', re.IGNORECASE),
        'HU': re.compile(r'(?:^|\s)HU[/_\s]?\d+'),  # Human Rights
        'AA': re.compile(r'(?:^|\s)AA[/_\s]?\d+'),  # Asylum Appeal
        'IA': re.compile(r'(?:^|\s)IA[/_\s]?\d+'),  # Immigration Appeal
        'DA': re.compile(r'(?:^|\s)DA[/_\s]?\d+'),  # Deportation Appeal
        'EA': re.compile(r'(?:^|\s)EA[/_\s]?\d+'),  # EEA Appeal
        'OA': re.compile(r'(?:^|\s)OA[/_\s]?\d+'),  # Other Appeal
        'VA': re.compile(r'(?:^|\s)VA[/_\s]?\d+'),  # Visit Visa Appeal
        'JR': re.compile(r'(?:^|\s)JR[/_\s]?\d+'),  # Judicial Review
        'RP': re.compile(r'(?:^|\s)RP[/_\s]?\d+'),  # Refugee Protection
        'DC': re.compile(r'(?:^|\s)DC[/_\s]?\d+'),  # Deprivation of Citizenship
        'UT': re.compile(r'(?:^|\s)UT[/_\s]?\d+'),  # Upper Tribunal case
        'FA': re.compile(r'(?:^|\s)FA[/_\s]?\d+'),  # Family Appeal
        'UKIAT': re.compile(r'UKIAT|UKAIT'),        # UK Asylum & Immigration Tribunal
        'UKUT': re.compile(r'UKUT'),                # UK Upper Tribunal
        'EWCA': re.compile(r'EWCA'),                # England and Wales Court of Appeal
        'UKSC': re.compile(r'UKSC'),                # UK Supreme Court
    }
    
    # Process all JSON files
    total_before = 0
    total_after = 0
    
    # Store raw reference numbers to analyze missed patterns
    unmatched_references = []
    
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(json_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                ref_num = data.get('reference_number', '')
                date_str = data.get('promulgation_date', '')
                
                # Extract determination type
                dtype = 'Unknown'
                matched = False
                for type_name, pattern in type_patterns.items():
                    if pattern.search(ref_num):
                        dtype = type_name
                        matched = True
                        break
                
                # Track unmatched reference numbers for analysis
                if not matched:
                    unmatched_references.append(ref_num)
                
                # Parse date to determine period
                date_obj = None
                year = None
                
                if date_str:
                    try:
                        # Try common date formats
                        for fmt in ['%d %b %Y', '%d %B %Y', '%Y-%m-%d', '%d/%m/%Y']:
                            try:
                                date_obj = datetime.strptime(date_str, fmt)
                                year = date_obj.year
                                break
                            except ValueError:
                                continue
                        
                        # If no format worked, try to extract year with regex
                        if not year:
                            year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
                            if year_match:
                                year = int(year_match.group(0))
                    except Exception as e:
                        pass
                
                # If still no year, try to extract from reference number
                if not year:
                    year_match = re.search(r'\b(19|20)\d{2}\b', ref_num)
                    if year_match:
                        year = int(year_match.group(0))
                
                # Store in appropriate counter based on year
                if year and year < split_year:
                    counts_before[dtype] += 1
                    total_before += 1
                elif year and year >= split_year:
                    counts_after[dtype] += 1
                    total_after += 1
                else:
                    # No year found - count as unknown period
                    # For this analysis, we could either skip or count in both periods
                    # Let's count in both for completeness
                    counts_before['Unknown Date'] += 1
                    counts_after['Unknown Date'] += 1
                    total_before += 1
                    total_after += 1
                
                # Add to records for detailed analysis
                records.append({
                    'reference_number': ref_num,
                    'determination_type': dtype,
                    'date': date_str,
                    'year': year,
                    'period': f"Before {split_year}" if year and year < split_year else 
                             (f"{split_year}-2025" if year and year >= split_year else "Unknown"),
                    'url': data.get('url', '')
                })
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Analyze unmatched reference numbers to identify missing patterns
    if unmatched_references:
        print(f"\n{len(unmatched_references)} unmatched reference numbers found.")
        print("Sample of unmatched reference numbers:")
        for ref in unmatched_references[:20]:  # Show first 20 examples
            print(f"  - {ref}")
        
        # Try to find patterns in unmatched references
        prefixes = []
        for ref in unmatched_references:
            # Extract first 2-3 characters if they seem to be a prefix
            match = re.match(r'^([A-Z]{2,3})[-_/\s]?\d+', ref)
            if match:
                prefixes.append(match.group(1))
        
        # Count most common prefixes
        if prefixes:
            prefix_counts = Counter(prefixes)
            print("\nMost common prefixes in unmatched references:")
            for prefix, count in prefix_counts.most_common(10):
                print(f"  - {prefix}: {count} occurrences")
    
    # Create dataframes for both periods
    df_before = pd.DataFrame({
        'Type': list(counts_before.keys()),
        'Count': list(counts_before.values())
    })
    df_before['Percentage'] = df_before['Count'] / total_before * 100 if total_before > 0 else 0
    df_before = df_before.sort_values('Count', ascending=False).reset_index(drop=True)
    df_before['Period'] = f"Before {split_year}"
    
    df_after = pd.DataFrame({
        'Type': list(counts_after.keys()),
        'Count': list(counts_after.values())
    })
    df_after['Percentage'] = df_after['Count'] / total_after * 100 if total_after > 0 else 0
    df_after = df_after.sort_values('Count', ascending=False).reset_index(drop=True)
    df_after['Period'] = f"{split_year}-2025"
    
    # Combine the dataframes
    df_combined = pd.concat([df_before, df_after])
    
    # Print summary
    print(f"\nAnalyzed cases by time period:")
    print(f"Before {split_year}: {total_before} cases")
    print(f"{split_year}-2025: {total_after} cases")
    print(f"Total: {total_before + total_after} cases\n")
    
    print(f"Top determination types before {split_year}:")
    print("----------------------------------------")
    for idx, row in df_before.head(10).iterrows():
        print(f"{row['Type']}: {row['Count']} ({row['Percentage']:.1f}%)")
    
    print(f"\nTop determination types {split_year}-2025:")
    print("----------------------------------------")
    for idx, row in df_after.head(10).iterrows():
        print(f"{row['Type']}: {row['Count']} ({row['Percentage']:.1f}%)")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Get all unique determination types
    all_types = sorted(set(df_before['Type'].tolist() + df_after['Type'].tolist()))
    all_types = [t for t in all_types if t != 'Unknown Date']  # Remove Unknown Date for clarity
    
    # Extract counts for each period, ensuring all types are represented
    before_counts = [df_before[df_before['Type'] == t]['Count'].sum() if t in df_before['Type'].values else 0 for t in all_types]
    after_counts = [df_after[df_after['Type'] == t]['Count'].sum() if t in df_after['Type'].values else 0 for t in all_types]
    
    # Set up bar positions
    x = np.arange(len(all_types))
    width = 0.35
    
    # Create grouped bar chart
    plt.bar(x - width/2, before_counts, width, label=f'Before {split_year}')
    plt.bar(x + width/2, after_counts, width, label=f'{split_year}-2025')
    
    plt.xlabel('Determination Type')
    plt.ylabel('Count')
    plt.title(f'Distribution of Determination Types by Period')
    plt.xticks(x, all_types, rotation=45)
    plt.legend()
    
    # Add counts on bars
    for i, v in enumerate(before_counts):
        if v > 0:
            plt.text(i - width/2, v + 100, str(v), ha='center')
    
    for i, v in enumerate(after_counts):
        if v > 0:
            plt.text(i + width/2, v + 100, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig('determination_types_by_period.png')
    plt.show()
    
    # Create another visualization showing the proportion of each type within its period
    plt.figure(figsize=(15, 10))
    
    # Calculate percentages
    before_total = sum(before_counts)
    after_total = sum(after_counts)
    before_pct = [100 * count / before_total if before_total > 0 else 0 for count in before_counts]
    after_pct = [100 * count / after_total if after_total > 0 else 0 for count in after_counts]
    
    # Create grouped bar chart for percentages
    plt.bar(x - width/2, before_pct, width, label=f'Before {split_year}')
    plt.bar(x + width/2, after_pct, width, label=f'{split_year}-2025')
    
    plt.xlabel('Determination Type')
    plt.ylabel('Percentage (%)')
    plt.title(f'Percentage Distribution of Determination Types by Period')
    plt.xticks(x, all_types, rotation=45)
    plt.legend()
    
    # Add percentages on bars
    for i, v in enumerate(before_pct):
        if v > 1:
            plt.text(i - width/2, v + 1, f"{v:.1f}%", ha='center')
    
    for i, v in enumerate(after_pct):
        if v > 1:
            plt.text(i + width/2, v + 1, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig('determination_types_by_period_percentage.png')
    plt.show()
    
    # Save results to CSV
    df_combined.to_csv('determination_types_by_period.csv', index=False)
    
    # Create a full DataFrame with all records
    df_records = pd.DataFrame(records)
    df_records.to_csv('determination_types_detailed_by_period.csv', index=False)
    
    print("\nSaved results to:")
    print("- determination_types_by_period.csv")
    print("- determination_types_detailed_by_period.csv")
    print("- determination_types_by_period.png")
    print("- determination_types_by_period_percentage.png")
    
    return df_combined

# If running as a script
if __name__ == "__main__":
    analyze_determination_types_by_period(json_dir='C:/Users/shil6369/Documents/social_data_science/Thesis/gov_scraping/data/json')