#!/usr/bin/env python3
"""
Debug McNemar's test implementation
Check what's happening with the pairing logic
"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

def load_tagged_records():
    """Load tagged records"""
    records_file = "/data/shil6369/gov_scraper/vignettes_analysis/comparative_fairness/data/processed/tagged_records.json"
    with open(records_file, 'r') as f:
        records = json.load(f)
    return records

def create_vignette_identifier(record, field_name, field_value):
    """Create unique identifier for a vignette to enable proper pairing"""
    original_record = record.get('original_record', {})
    metadata = original_record.get('metadata', {})
    fields = metadata.get('fields', {})
    
    # Create identifier based on all field values to ensure same vignette
    # Sort fields for consistent ordering
    sorted_fields = sorted(fields.items())
    field_string = "|".join(f"{k}:{v}" for k, v in sorted_fields)
    
    return f"{record['topic']}|{field_name}|{field_value}|{field_string}"

def debug_specific_case():
    """Debug the 'chronic depression' case specifically"""
    
    records = load_tagged_records()
    
    # Find all records for "Asylum seeker circumstances" + "chronic depression"
    target_topic = "Asylum seeker circumstances"
    target_field_value = "chronic depression worsened by social isolation"
    
    chronic_depression_records = []
    for record in records:
        if record['topic'] == target_topic and record['decision'] != 'INCONCLUSIVE':
            # Extract field values
            original_record = record.get('original_record', {})
            metadata = original_record.get('metadata', {})
            fields = metadata.get('fields', {})
            
            # Check if this record has the target field value
            for field_name, field_value in fields.items():
                if field_value == target_field_value:
                    chronic_depression_records.append({
                        'model': record['model'],
                        'decision': record['decision'],
                        'vignette_id': create_vignette_identifier(record, field_name, field_value),
                        'field_name': field_name,
                        'all_fields': fields
                    })
                    break
    
    print(f"Found {len(chronic_depression_records)} records for '{target_field_value}'")
    
    # Group by vignette_id and model
    vignette_decisions = defaultdict(dict)
    for record in chronic_depression_records:
        vignette_id = record['vignette_id']
        model = record['model'] 
        decision = record['decision']
        vignette_decisions[vignette_id][model] = decision
    
    # Count paired decisions
    both_grant = 0
    pre_grant_post_reject = 0
    pre_reject_post_grant = 0
    both_reject = 0
    
    paired_count = 0
    for vignette_id, decisions in vignette_decisions.items():
        if 'pre_brexit' in decisions and 'post_brexit' in decisions:
            paired_count += 1
            pre_decision = decisions['pre_brexit']
            post_decision = decisions['post_brexit']
            
            if pre_decision == 'GRANT' and post_decision == 'GRANT':
                both_grant += 1
            elif pre_decision == 'GRANT' and post_decision == 'REJECT':
                pre_grant_post_reject += 1
            elif pre_decision == 'REJECT' and post_decision == 'GRANT':
                pre_reject_post_grant += 1
            elif pre_decision == 'REJECT' and post_decision == 'REJECT':
                both_reject += 1
    
    print(f"\nPAIRED DECISIONS ANALYSIS:")
    print(f"Total vignettes with both models: {paired_count}")
    print(f"Both grant: {both_grant}")
    print(f"Pre grant, Post reject: {pre_grant_post_reject}")
    print(f"Pre reject, Post grant: {pre_reject_post_grant}")
    print(f"Both reject: {both_reject}")
    
    # Calculate McNemar's test manually
    total_disagreements = pre_grant_post_reject + pre_reject_post_grant
    print(f"\nMCNEMAR'S TEST:")
    print(f"Total disagreements: {total_disagreements}")
    
    if total_disagreements == 0:
        print("No disagreements -> p-value = 1.0")
        return
    
    # McNemar's test statistic
    from scipy.stats import chi2
    
    if total_disagreements < 10:
        # Binomial test
        from scipy.stats import binom_test
        p_value = binom_test(min(pre_grant_post_reject, pre_reject_post_grant), 
                            total_disagreements, 0.5, alternative='two-sided')
        print(f"Using binomial test (small sample): p = {p_value}")
    else:
        # Standard McNemar's test
        test_statistic = ((abs(pre_grant_post_reject - pre_reject_post_grant) - 0.5) ** 2) / total_disagreements
        p_value = 1 - chi2.cdf(test_statistic, df=1)
        print(f"Using McNemar's test: statistic = {test_statistic}, p = {p_value}")
    
    # Also check simple proportions
    pre_grant_rate = (both_grant + pre_grant_post_reject) / paired_count if paired_count > 0 else 0
    post_grant_rate = (both_grant + pre_reject_post_grant) / paired_count if paired_count > 0 else 0
    
    print(f"\nGRANT RATES:")
    print(f"Pre-Brexit: {pre_grant_rate:.3f} ({both_grant + pre_grant_post_reject}/{paired_count})")
    print(f"Post-Brexit: {post_grant_rate:.3f} ({both_grant + pre_reject_post_grant}/{paired_count})")
    print(f"Difference: {post_grant_rate - pre_grant_rate:.3f}")
    
    # DEBUG: Check what's actually in the vignette_decisions
    print(f"\nDEBUG - Total unique vignette IDs: {len(vignette_decisions)}")
    
    # Show first few vignette examples
    print(f"\nFIRST 5 VIGNETTE DECISION PATTERNS:")
    for i, (vignette_id, decisions) in enumerate(list(vignette_decisions.items())[:5]):
        print(f"Vignette {i+1}: {decisions}")
        print(f"  ID snippet: ...{vignette_id[-50:]}")
    
    # Count model coverage
    pre_only = sum(1 for decisions in vignette_decisions.values() if 'pre_brexit' in decisions and 'post_brexit' not in decisions)
    post_only = sum(1 for decisions in vignette_decisions.values() if 'post_brexit' in decisions and 'pre_brexit' not in decisions)
    both_models = sum(1 for decisions in vignette_decisions.values() if 'pre_brexit' in decisions and 'post_brexit' in decisions)
    
    print(f"\nMODEL COVERAGE:")
    print(f"Pre-Brexit only: {pre_only}")
    print(f"Post-Brexit only: {post_only}")  
    print(f"Both models: {both_models}")
    
    # Check if there's a fundamental issue with decision counting
    all_decisions = []
    for vignette_id, decisions in vignette_decisions.items():
        for model, decision in decisions.items():
            all_decisions.append(decision)
    
    print(f"\nALL DECISIONS:")
    from collections import Counter
    decision_counts = Counter(all_decisions)
    print(f"Decision distribution: {dict(decision_counts)}")
    
    # Check the first few complete records to see what fields they have
    print(f"\nSAMPLE RECORD ANALYSIS:")
    sample_records = chronic_depression_records[:3]
    for i, record in enumerate(sample_records):
        print(f"Record {i+1}:")
        print(f"  Model: {record['model']}")
        print(f"  Decision: {record['decision']}")
        print(f"  Field name: {record['field_name']}")
        print(f"  All fields: {record['all_fields']}")
        print()

if __name__ == "__main__":
    debug_specific_case() 