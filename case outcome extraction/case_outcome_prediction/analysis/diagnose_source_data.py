#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
"""
Diagnose source data issues in original JSON files
"""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

def diagnose_json_files():
    print('DIAGNOSING ORIGINAL JSON FILES')
    print('='*50)
    
    # Find JSON directory
    possible_paths = [
        Path("../data/json"),
        Path("data/json"), 
        Path("../preprocessing/data/json"),
        Path("/data/shil6369/gov_scraper/data/json")
    ]
    
    json_dir = None
    for path in possible_paths:
        if path.exists():
            json_dir = path
            break
    
    if not json_dir:
        print("❌ Could not find JSON directory. Checking current directory structure...")
        # List directories to help locate JSON files
        for root, dirs, files in os.walk("."):
            if any(f.endswith('.json') for f in files):
                print(f"Found JSON files in: {root}")
        return
    
    print(f"✓ Found JSON directory: {json_dir}")
    
    # Get all JSON files
    json_files = list(json_dir.glob("*.json"))
    print(f"✓ Found {len(json_files):,} JSON files")
    
    if not json_files:
        print("❌ No JSON files found!")
        return
    
    # Diagnostic counters
    total_files = len(json_files)
    missing_decision_text = 0
    empty_decision_text = 0
    very_short_text = 0
    normal_text = 0
    read_errors = 0
    
    # Sample some files for detailed analysis
    problem_files = []
    
    print(f"\nAnalyzing {total_files:,} files...")
    
    for file_path in tqdm(json_files, desc="Checking JSON files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            decision_text = data.get('decision_text')
            
            if decision_text is None:
                missing_decision_text += 1
                problem_files.append({
                    'file': file_path.name,
                    'issue': 'missing_field',
                    'available_fields': list(data.keys())
                })
            elif isinstance(decision_text, str) and decision_text.strip() == '':
                empty_decision_text += 1
                problem_files.append({
                    'file': file_path.name,
                    'issue': 'empty_string',
                    'available_fields': list(data.keys())
                })
            elif isinstance(decision_text, str) and len(decision_text.strip()) < 100:
                very_short_text += 1
                problem_files.append({
                    'file': file_path.name,
                    'issue': 'very_short',
                    'length': len(decision_text.strip()),
                    'content_preview': decision_text.strip()[:100]
                })
            else:
                normal_text += 1
                
        except Exception as e:
            read_errors += 1
            print(f"Error reading {file_path.name}: {e}")
    
    # Results
    print(f"\nDIAGNOSTIC RESULTS:")
    print(f"="*30)
    print(f"Total files analyzed: {total_files:,}")
    print(f"  ✓ Normal decision_text: {normal_text:,} ({normal_text/total_files*100:.1f}%)")
    print(f"  ❌ Missing 'decision_text' field: {missing_decision_text:,} ({missing_decision_text/total_files*100:.1f}%)")
    print(f"  ❌ Empty 'decision_text' string: {empty_decision_text:,} ({empty_decision_text/total_files*100:.1f}%)")
    print(f"  ⚠️  Very short text (<100 chars): {very_short_text:,} ({very_short_text/total_files*100:.1f}%)")
    print(f"  💥 Read errors: {read_errors:,}")
    
    total_problematic = missing_decision_text + empty_decision_text + very_short_text
    print(f"\nTotal problematic files: {total_problematic:,} ({total_problematic/total_files*100:.1f}%)")
    
    # Show examples of problem files
    if problem_files:
        print(f"\nSAMPLE PROBLEM FILES (first 10):")
        print(f"-"*40)
        for i, problem in enumerate(problem_files[:10]):
            print(f"{i+1}. {problem['file']} - Issue: {problem['issue']}")
            if 'length' in problem:
                print(f"   Length: {problem['length']} chars")
            if 'content_preview' in problem:
                print(f"   Preview: {problem['content_preview'][:50]}...")
            if 'available_fields' in problem:
                print(f"   Available fields: {problem['available_fields'][:5]}...")
            print()
    
    # Check if there are alternative text fields
    print(f"\nCHECKING ALTERNATIVE TEXT FIELDS:")
    print(f"-"*40)
    
    # Sample first 100 files to check field patterns
    sample_files = json_files[:100]
    field_counter = {}
    
    for file_path in sample_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for field in data.keys():
                if 'text' in field.lower() or 'content' in field.lower():
                    field_counter[field] = field_counter.get(field, 0) + 1
        except:
            continue
    
    print("Text-related fields found in sample:")
    for field, count in sorted(field_counter.items(), key=lambda x: x[1], reverse=True):
        print(f"  {field}: {count}/{len(sample_files)} files ({count/len(sample_files)*100:.1f}%)")
    
    # Expected vs actual empty count
    expected_empty = missing_decision_text + empty_decision_text
    print(f"\nCOMPARISON WITH PROCESSED DATA:")
    print(f"-"*40)
    print(f"Expected empty from source: {expected_empty:,}")
    print(f"Actual empty in processed data: 11,306")
    print(f"Difference: {abs(11306 - expected_empty):,}")
    
    if abs(11306 - expected_empty) > 1000:
        print("⚠️  Large difference suggests preprocessing is creating additional empty cases")
    else:
        print("✓ Numbers are close - likely source data issue")

if __name__ == "__main__":
    diagnose_json_files() 