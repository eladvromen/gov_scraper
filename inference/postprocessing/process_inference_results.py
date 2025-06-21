#!/usr/bin/env python3
"""
Post-processing script for inference results from UK immigration LLM study.

This script processes raw model responses to extract clean decisions and reasoning.
Focuses on the primary decision and legal reasoning for research analysis.

Usage:
    python process_inference_results.py <input_file_path>
    
Example:
    python process_inference_results.py ../results/subset_inference_llama3_8b_pre_brexit_20250620_120959.json
"""

import json
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# --- Decision Extraction ---

# Map decision phrases to canonical forms
DECISION_VARIANTS = {
    'granted': 'Granted',
    'allowed': 'Granted',
    'appeal allowed': 'Granted',
    'application granted': 'Granted',
    'leave to remain': 'Granted',
    'leave granted': 'Granted',
    'approved': 'Granted',
    'accepted': 'Granted',
    'denied': 'Denied',
    'dismissed': 'Denied',
    'refused': 'Denied',
    'not allowed': 'Denied',
    'not granted': 'Denied',
    'not approved': 'Denied',
    'not accepted': 'Denied',
    'not permitted': 'Denied',
    'appeal dismissed': 'Denied',
    'application refused': 'Denied',
    'leave refused': 'Denied',
    'rejected': 'Denied',
    'declined': 'Denied',
}

# Simple, focused patterns for decision extraction
DECISION_PATTERNS = [
    # Explicit decision statements
    r'\b(?:application|case|appeal|claim|request)\s+(?:is\s+)?(?P<word>granted|denied|dismissed|refused|allowed|approved|accepted|rejected|declined)\b',
    r'\b(?:leave\s+to\s+remain|leave\s+granted|leave\s+refused)\b',
    # Standalone decisions at start of response or after "Decision:"
    r'\b(?P<neg>not\s+)?(?P<word2>granted|denied|dismissed|refused|allowed|approved|accepted|rejected|declined)\b',
    # Negation patterns
    r'(?P<neg4>cannot\s+be|should\s+not\s+be|must\s+not\s+be|will\s+not\s+be|is\s+not)\s+(?P<word4>granted|allowed|approved|accepted|permitted)',
]

def normalize_decision(word: str, neg: Optional[str] = None) -> Optional[str]:
    """Normalize a decision word to canonical form"""
    word = word.lower().strip()
    if not word:
        return None
    
    if neg and neg.strip():
        # Negation flips the meaning
        if word in ['granted', 'allowed', 'approved', 'accepted', 'permitted']:
            return 'Denied'
        elif word in ['denied', 'dismissed', 'refused', 'rejected', 'declined']:
            return 'Granted'
    
    # Map to canonical form
    for k, v in DECISION_VARIANTS.items():
        if word == k or word in k:
            return v
    return None

def extract_primary_decision(response_text: str) -> Optional[str]:
    """Extract the primary decision from the response (first valid decision found by position)"""
    if not response_text:
        return None
    
    # Find all decisions with their positions
    all_decisions = []
    
    for pattern in DECISION_PATTERNS:
        for m in re.finditer(pattern, response_text, re.IGNORECASE):
            # Extract word and negation
            word = ''
            neg = ''
            
            for group_name in ['word', 'word2', 'word4']:
                try:
                    if m.group(group_name):
                        word = m.group(group_name)
                        break
                except (IndexError, AttributeError):
                    continue
            
            for group_name in ['neg', 'neg4']:
                try:
                    if m.group(group_name):
                        neg = m.group(group_name)
                        break
                except (IndexError, AttributeError):
                    continue
            
            decision = normalize_decision(word, neg)
            if decision:
                all_decisions.append((decision, m.start()))
    
    # Sort by position and return the first one
    if all_decisions:
        all_decisions.sort(key=lambda x: x[1])  # Sort by position
        return all_decisions[0][0]  # Return the decision (not position)
    
    return None

# --- Text Cleaning ---

def clean_response_text(response_text: str) -> str:
    """Clean response text by removing code contamination"""
    if not response_text:
        return ""
    
    # More comprehensive code contamination patterns
    code_start_patterns = [
        # Triple quotes followed by code
        '\n"""\n\ndef ',
        '\n"""\n\nclass ',
        '\n"""\n\nimport ',
        '\n"""\n\nfrom ',
        "\n'''\n\ndef ",
        "\n'''\n\nclass ",
        # Code blocks
        '\n```\n\ndef ',
        '\n```python\n',
        '\n```java\n',
        '\n```javascript\n',
        # Direct code patterns
        '\n\n# Import ',
        '\n\nimport ',
        '\n\nfrom ',
        '\n\ndef ',
        '\n\nclass ',
        # Comments followed by imports/functions
        '\n# Load data\n',
        '\n# Import libraries\n',
        '\ndata = load_data()',
        '\ndef split_by_sentence(',
        '\ndef get_sentences(',
        '\ndef get_headline(',
        '\ndef make_decision(',
    ]
    
    cleaned_text = response_text
    
    # Find earliest code contamination and cut there
    earliest_code_start = len(cleaned_text)
    for pattern in code_start_patterns:
        pos = cleaned_text.find(pattern)
        if pos != -1 and pos < earliest_code_start:
            earliest_code_start = pos
    
    # Also look for standalone triple quotes that might indicate code
    triple_quote_pos = cleaned_text.find('\n"""\n')
    if triple_quote_pos != -1 and triple_quote_pos < earliest_code_start:
        earliest_code_start = triple_quote_pos
    
    # Look for import statements not at the very beginning
    import_patterns = [r'\nimport \w+', r'\nfrom \w+', r'\n# Import']
    for pattern in import_patterns:
        match = re.search(pattern, cleaned_text)
        if match and match.start() > 50:  # Not at the very beginning
            if match.start() < earliest_code_start:
                earliest_code_start = match.start()
    
    if earliest_code_start < len(cleaned_text):
        cleaned_text = cleaned_text[:earliest_code_start]
    
    # Clean up artifacts
    trailing_artifacts = ['"""', "'''", '```', '---', '===', '"\n']
    for artifact in trailing_artifacts:
        if cleaned_text.rstrip().endswith(artifact):
            cleaned_text = cleaned_text.rstrip()[:-len(artifact)].rstrip()
    
    leading_artifacts = ['"""', "'''", '```', 'Output:', 'Result:', 'Response:']
    cleaned_text = cleaned_text.strip()
    for artifact in leading_artifacts:
        if cleaned_text.startswith(artifact):
            cleaned_text = cleaned_text[len(artifact):].strip()
    
    return cleaned_text.strip()

# --- Reasoning Extraction ---

REASONING_LABELS = [
    r'Reasoning\s*[:\-]',
    r'I\s+find\s+that',
    r'I\s+conclude\s+that',
    r'I\s+determine\s+that',
    r'because',
    r'on\s+the\s+basis\s+that',
    r'for\s+the\s+following\s+reasons',
    r'it\s+is\s+clear\s+that',
    r'I\s+am\s+satisfied\s+that',
    r'I\s+am\s+not\s+satisfied\s+that',
]

def extract_reasoning(response_text: str) -> Optional[str]:
    """Extract legal reasoning from the response"""
    if not response_text:
        return None
    
    # Try explicit reasoning labels first
    for label in REASONING_LABELS:
        pattern = label + r'(.+?)(?=\n\s*Decision|$)'
        match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            reasoning = re.sub(r'\s+', ' ', reasoning)  # Normalize whitespace
            if len(reasoning.split()) > 3:  # Must be substantial
                return reasoning
    
    # Fallback: extract text after first decision
    decision_match = re.search(r'\b(granted|denied)\b', response_text, re.IGNORECASE)
    if decision_match:
        after_decision = response_text[decision_match.end():].strip()
        # Remove any remaining "Decision:" or "Reasoning:" labels
        after_decision = re.sub(r'^(Decision|Reasoning)\s*[:\-]\s*', '', after_decision, flags=re.IGNORECASE)
        # Take first few sentences
        sentences = re.split(r'(?<=[.!?])\s+', after_decision)
        reasoning = ' '.join(sentences[:3]).strip()
        if len(reasoning.split()) > 3:
            return reasoning
    
    return None

# --- Main Processing ---

def analyze_response(response_text: str, case_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single response to extract decision and reasoning"""
    # Clean the response first
    cleaned_text = clean_response_text(response_text)
    
    # Extract decision and reasoning from cleaned text
    decision = extract_primary_decision(cleaned_text)
    reasoning = extract_reasoning(cleaned_text)
    
    # Simple quality metrics
    has_code_contamination = len(cleaned_text) < len(response_text) * 0.9
    
    return {
        'decision': decision,
        'reasoning': reasoning,
        'original_response': response_text,
        'cleaned_response': cleaned_text,
        'has_code_contamination': has_code_contamination,
        'metadata': case_metadata,
        'quality_metrics': {
            'response_length': len(cleaned_text.split()) if cleaned_text else 0,
            'original_length': len(response_text.split()) if response_text else 0,
            'has_decision': decision is not None,
            'has_reasoning': reasoning is not None and len(reasoning.split()) > 3,
            'text_reduction_ratio': len(cleaned_text) / len(response_text) if response_text else 1.0
        }
    }

def calculate_summary_statistics(processed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate summary statistics for the processed results"""
    total = len(processed_results)
    granted = sum(1 for r in processed_results if r['decision'] == 'Granted')
    denied = sum(1 for r in processed_results if r['decision'] == 'Denied')
    no_decision = sum(1 for r in processed_results if r['decision'] is None)
    
    has_reasoning = sum(1 for r in processed_results if r['quality_metrics']['has_reasoning'])
    has_code_contamination = sum(1 for r in processed_results if r['has_code_contamination'])
    
    avg_length = sum(r['quality_metrics']['response_length'] for r in processed_results) / total if total > 0 else 0
    
    return {
        'total_responses': total,
        'decision_stats': {
            'granted': granted,
            'denied': denied,
            'no_decision': no_decision,
            'granted_rate': granted / total if total > 0 else 0,
            'denied_rate': denied / total if total > 0 else 0,
        },
        'quality_metrics': {
            'avg_response_length': avg_length,
            'has_decision_rate': (total - no_decision) / total if total > 0 else 0,
            'has_reasoning_rate': has_reasoning / total if total > 0 else 0,
            'code_contamination_rate': has_code_contamination / total if total > 0 else 0,
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Process inference results to extract decisions and reasoning')
    parser.add_argument('input_file', help='Path to the input JSON file')
    parser.add_argument('--max-records', type=int, help='Maximum number of records to process (for testing)')
    parser.add_argument('--verbose', action='store_true', help='Print progress information')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        sys.exit(1)
    
    # Load input data
    with open(input_path, 'r') as f:
        input_data = json.load(f)
    
    if isinstance(input_data, list):
        items = input_data
    elif isinstance(input_data, dict) and 'results' in input_data:
        items = input_data['results']
    else:
        print("Error: Unexpected input format")
        sys.exit(1)
    
    # Limit records if specified
    if args.max_records:
        items = items[:args.max_records]
    
    print(f"Processing {input_path}...")
    
    # Process each item
    processed_results = []
    for i, item in enumerate(items):
        if args.verbose and (i + 1) % 100 == 0:
            print(f"Processing item {i + 1}/{len(items)}...")
        
        response_text = item.get('model_response', '')
        metadata = {k: v for k, v in item.items() if k != 'model_response'}
        
        result = analyze_response(response_text, metadata)
        processed_results.append(result)
    
    # Calculate summary statistics
    summary_stats = calculate_summary_statistics(processed_results)
    
    # Prepare output
    output_data = {
        'metadata': {
            'input_file': str(input_path),
            'processing_timestamp': datetime.now().isoformat(),
            'total_items_processed': len(processed_results)
        },
        'summary_statistics': summary_stats,
        'processed_results': processed_results
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"processed_{input_path.stem}_{timestamp}.json"
    output_path = input_path.parent / 'processed' / output_filename
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("Processing complete!")
    print(f"Results saved to: {output_path}")
    print(f"Total items processed: {len(processed_results)}")
    print("Summary statistics:")
    print(f"  - Granted: {summary_stats['decision_stats']['granted']} ({summary_stats['decision_stats']['granted_rate']:.1%})")
    print(f"  - Denied: {summary_stats['decision_stats']['denied']} ({summary_stats['decision_stats']['denied_rate']:.1%})")
    print(f"  - No decision: {summary_stats['decision_stats']['no_decision']}")
    print(f"  - Has reasoning: {summary_stats['quality_metrics']['has_reasoning_rate']:.1%}")
    print(f"  - Code contamination: {summary_stats['quality_metrics']['code_contamination_rate']:.1%}")

if __name__ == "__main__":
    main() 