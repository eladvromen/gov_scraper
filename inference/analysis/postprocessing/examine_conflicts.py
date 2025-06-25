#!/usr/bin/env python3

import json
import sys
import re
from pathlib import Path

def examine_conflicts(processed_file_path, max_examples=10):
    """Examine conflicting decisions to identify false positives"""
    
    with open(processed_file_path, 'r') as f:
        data = json.load(f)
    
    # Find all cases with conflicting decisions
    conflicts = []
    for item in data['processed_results']:
        conflict_flags = [flag for flag in item['flags'] if flag['type'] == 'conflicting_decisions']
        if conflict_flags:
            conflicts.append({
                'metadata': item['metadata'],
                'decision': item['decision'],
                'all_decisions': item['all_decisions'],
                'cleaned_response': item['cleaned_response'],
                'original_response': item['original_response'],
                'conflict_info': conflict_flags[0]
            })
    
    print(f"Found {len(conflicts)} cases with conflicting decisions")
    print("="*80)
    
    for i, conflict in enumerate(conflicts[:max_examples]):
        print(f"\n=== CONFLICT CASE {i+1} ===")
        print(f"Name: {conflict['metadata']['fields']['name']}")
        print(f"Age: {conflict['metadata']['fields']['age']}")
        print(f"Country: {conflict['metadata']['fields']['country']}")
        print(f"Primary decision: {conflict['decision']}")
        print(f"All decisions found: {conflict['all_decisions']}")
        print(f"Conflicting decisions: {conflict['conflict_info']['decisions']}")
        print(f"Positions: {conflict['conflict_info']['positions']}")
        
        print(f"\n--- CLEANED RESPONSE ---")
        cleaned = conflict['cleaned_response']
        print(f"Length: {len(cleaned)} chars")
        print(f"Text: {repr(cleaned[:300])}{'...' if len(cleaned) > 300 else ''}")
        
        # Highlight where each decision was found
        print(f"\n--- DECISION LOCATIONS ---")
        for decision, pos in conflict['all_decisions']:
            start = max(0, pos - 20)
            end = min(len(cleaned), pos + 30)
            context = cleaned[start:end]
            # Find the actual word at that position
            word_at_pos = ""
            words = cleaned.split()
            char_count = 0
            for word in words:
                if char_count <= pos < char_count + len(word):
                    word_at_pos = word
                    break
                char_count += len(word) + 1  # +1 for space
            print(f"  {decision} at pos {pos}: '...{context}...' (word: '{word_at_pos}')")
        
        print(f"\n--- ANALYSIS ---")
        # Try to determine if this is a real conflict or false positive
        decisions = [d[0] for d in conflict['all_decisions']]
        unique_decisions = list(set(decisions))
        
        if len(unique_decisions) == 2 and 'Granted' in unique_decisions and 'Denied' in unique_decisions:
            print("  Type: Granted vs Denied conflict")
            
            # Check if it's a pattern like "not granted" being split
            text_lower = cleaned.lower()
            if 'not granted' in text_lower or 'cannot be granted' in text_lower:
                print("  Likely cause: 'not granted' or similar phrase")
            elif 'would be denied' in text_lower or 'should be denied' in text_lower:
                print("  Likely cause: hypothetical denial statement")
            else:
                print("  Likely cause: Unknown - needs manual review")
        
        print("="*80)
    
    # Summary of conflict patterns
    print(f"\n=== CONFLICT PATTERN SUMMARY ===")
    decision_pairs = {}
    for conflict in conflicts:
        decisions = tuple(sorted(conflict['conflict_info']['decisions']))
        decision_pairs[decisions] = decision_pairs.get(decisions, 0) + 1
    
    for pair, count in sorted(decision_pairs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pair[0]} vs {pair[1]}: {count} cases")

def main():
    if len(sys.argv) != 2:
        print("Usage: python examine_conflicts.py <processed_results_file>")
        sys.exit(1)
    
    processed_file = sys.argv[1]
    if not Path(processed_file).exists():
        print(f"File not found: {processed_file}")
        sys.exit(1)
    
    examine_conflicts(processed_file)

if __name__ == "__main__":
    main() 