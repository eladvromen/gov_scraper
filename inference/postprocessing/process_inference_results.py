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
    'denied': 'Denied',
    'granted': 'Granted',
    'dismissed': 'Denied',
    'allowed': 'Granted',
    'refused': 'Denied',
    'approved': 'Granted',
    'allow': 'Granted',
    'deny': 'Denied',
    'grant': 'Granted',  # "grant" = Granted
    'asylum': 'Granted',  # "grant asylum" = Granted
}

# Patterns specifically for the format: "Decision: DENIED/denied *END*"
DECISION_PATTERNS = [
    # Primary pattern: Decision/Decided: [specific_word] followed by *END*
    r'(?:Decision|DECISION|Decided):\s*(?P<decision>denied|granted|dismissed|allowed|refused|approved)\s*\*END\*',
    # Fallback: Decision/Decided: [specific_word] without *END* requirement
    r'(?:Decision|DECISION|Decided):\s*(?P<decision>denied|granted|dismissed|allowed|refused|approved)',
    # Decision with compound phrases: "Decision: ASYLUM GRANTED", "Decision: Appeal denied"
    r'(?:Decision|DECISION|Decided):\s*(?:ASYLUM\s+|Appeal\s+)?(?P<decision_compound>granted|denied|dismissed|allowed|refused|approved)',
    # Alternative format: *GRANTED* or *DENIED*
    r'\*(?P<decision2>granted|denied|dismissed|allowed|refused|approved)\*',
    # Appeal/claim/application format: "The appeal/claim/application is [accordingly] granted/denied"
    r'(?:appeal|claim|application)\s+is\s+(?:accordingly\s+)?(?P<decision3>granted|denied|dismissed|allowed|refused|approved)',
    # Application for asylum format: "application for asylum is granted/denied"
    r'application\s+for\s+asylum\s+(?:in\s+the\s+UK\s+)?is\s+(?:therefore\s+)?(?P<decision3c>granted|denied|dismissed|allowed|refused|approved)',
    # Claim for asylum format: "claim for asylum is denied/granted"
    r'claim\s+for\s+asylum\s+is\s+(?:accordingly\s+)?(?P<decision3b>granted|denied|dismissed|allowed|refused|approved)',
    # Case format: "case dismissed/granted/denied"
    r'case\s+(?P<decision4>dismissed|granted|denied|allowed|refused|approved)',
    # Judge statements: "I allow/deny/grant [name]'s appeal" or "I allow/deny/grant [the] appeal"
    r'I\s+(?P<decision5>allow|deny|grant)\s+(?:\w+\'s\s+)?(?:appeal|it|her\s+appeal|his\s+appeal|the\s+appeal|this\s+appeal)',
    # Judge statements without object: "I grant/deny/allow" (standalone)
    r'I\s+(?P<decision5b>grant|deny|allow)(?:\s+(?:the|this)\s+appeal)?(?:\s+on\s+[\w\s]+grounds)?[.\s]*(?:\*END\*|$)',
    # "Should be" format: "appeal should be GRANTED/DENIED"
    r'(?:appeal|claim|application)?\s*should\s+be\s+(?P<decision5c>granted|denied|dismissed|allowed|refused|approved)',
    # Leading phrases: "leading me to grant/deny"
    r'leading\s+me\s+to\s+(?P<decision5d>grant|deny)\s+(?:this\s+|his\s+|her\s+|their\s+)?(?:appeal|claim|application)',
    # Appellant format: "The Appellant is granted/denied"
    r'(?:The\s+)?Appellant\s+is\s+(?P<decision6>granted|denied|dismissed|allowed|refused|approved)',
    # Asylum format: "grant/deny asylum"
    r'(?:grant|deny)\s+(?P<decision7>asylum)',
    # Asylum claim format: "deny their asylum claim"
    r'(?P<decision7b>grant|deny)\s+(?:their|his|her)\s+asylum\s+claim',
    # Standalone decisions (more restrictive context)
    r'(?:^|\n)\s*(?P<decision8>Granted|Denied|Allowed|Dismissed|Refused|Approved)\s*(?:\n|\*END\*|$)',
]

def normalize_decision(word: str) -> Optional[str]:
    """Normalize a decision word to canonical form"""
    if not word:
        return None
    
    word = word.lower().strip()
    return DECISION_VARIANTS.get(word, None)

def extract_primary_decision(response_text: str) -> Optional[str]:
    """Extract the primary decision from the response"""
    if not response_text:
        return None
    
    # Find decisions with their positions
    all_decisions = []
    
    for pattern in DECISION_PATTERNS:
        for match in re.finditer(pattern, response_text, re.IGNORECASE):
            try:
                # Try all decision groups
                decision_word = None
                for group_name in ['decision', 'decision_compound', 'decision2', 'decision3', 'decision3b', 'decision3c', 'decision4', 'decision5', 'decision5b', 'decision5c', 'decision5d', 'decision6', 'decision7', 'decision7b', 'decision8']:
                    if group_name in match.groupdict() and match.group(group_name):
                        decision_word = match.group(group_name)
                        break
                
                if decision_word:
                    decision = normalize_decision(decision_word)
                    if decision:
                        all_decisions.append((decision, match.start()))
            except (IndexError, AttributeError):
                continue
    
    # Sort by position and return the first one
    if all_decisions:
        all_decisions.sort(key=lambda x: x[1])  # Sort by position
        return all_decisions[0][0]  # Return the decision (not position)
    
    return None

# --- Text Cleaning ---

def clean_response_text(response_text: str) -> str:
    """Clean response text by removing content after *END* and other artifacts"""
    if not response_text:
        return ""
    
    cleaned_text = response_text.strip()
    
    # Find and cut at *END* marker if present
    end_marker_pos = cleaned_text.find('*END*')
    if end_marker_pos != -1:
        # Keep everything up to and including *END*
        cleaned_text = cleaned_text[:end_marker_pos + 5]  # +5 for "*END*"
    
    # Remove any trailing artifacts after cleaning
    trailing_artifacts = ['"""', "'''", '```', '---', '===']
    for artifact in trailing_artifacts:
        if cleaned_text.rstrip().endswith(artifact):
            cleaned_text = cleaned_text.rstrip()[:-len(artifact)].rstrip()
    
    # Remove leading artifacts
    leading_artifacts = ['"""', "'''", '```', 'Output:', 'Result:', 'Response:']
    for artifact in leading_artifacts:
        if cleaned_text.startswith(artifact):
            cleaned_text = cleaned_text[len(artifact):].strip()
    
    return cleaned_text.strip()

# --- Reasoning Extraction ---

def extract_reasoning(response_text: str) -> Optional[str]:
    """Extract legal reasoning from the response (appears before Decision:)"""
    if not response_text:
        return None
    
    # Clean the text first
    text = response_text.strip()
    
    # Find the Decision: marker to know where reasoning ends
    decision_match = re.search(r'Decision:\s*\w+', text, re.IGNORECASE)
    if decision_match:
        # Extract everything before "Decision:"
        reasoning_text = text[:decision_match.start()].strip()
    else:
        # If no Decision: marker found, take first part of text
        reasoning_text = text
    
    # Remove explicit "Reasoning:" label if present
    reasoning_text = re.sub(r'^Reasoning:\s*', '', reasoning_text, flags=re.IGNORECASE).strip()
    
    # Clean up the reasoning text
    if reasoning_text:
        # Remove excessive whitespace
        reasoning_text = re.sub(r'\s+', ' ', reasoning_text)
        
        # Must be substantial (more than just a few words)
        if len(reasoning_text.split()) > 5:
            return reasoning_text
    
    return None

# --- Main Processing ---

def analyze_response(response_text: str, case_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single response to extract decision and reasoning"""
    # Clean the response first
    cleaned_text = clean_response_text(response_text)
    
    # Extract decision and reasoning from cleaned text
    decision = extract_primary_decision(cleaned_text)
    reasoning = extract_reasoning(cleaned_text)
    
    # Quality metrics specific to this format
    has_end_marker = '*END*' in response_text
    has_decision_format = bool(re.search(r'(?:Decision|DECISION|Decided):\s*\w+', response_text, re.IGNORECASE)) or \
                         bool(re.search(r'(?:Decision|DECISION|Decided):\s*(?:ASYLUM\s+|Appeal\s+)?(?:granted|denied|dismissed|allowed|refused|approved)', response_text, re.IGNORECASE)) or \
                         bool(re.search(r'\*(?:granted|denied|dismissed|allowed|refused|approved)\*', response_text, re.IGNORECASE)) or \
                         bool(re.search(r'(?:appeal|claim|application)\s+is\s+(?:accordingly\s+)?(?:granted|denied|dismissed|allowed|refused|approved)', response_text, re.IGNORECASE)) or \
                         bool(re.search(r'application\s+for\s+asylum\s+(?:in\s+the\s+UK\s+)?is\s+(?:therefore\s+)?(?:granted|denied|dismissed|allowed|refused|approved)', response_text, re.IGNORECASE)) or \
                         bool(re.search(r'claim\s+for\s+asylum\s+is\s+(?:accordingly\s+)?(?:granted|denied)', response_text, re.IGNORECASE)) or \
                         bool(re.search(r'case\s+(?:dismissed|granted|denied|allowed|refused|approved)', response_text, re.IGNORECASE)) or \
                         bool(re.search(r'I\s+(?:allow|deny|grant)\s+(?:\w+\'s\s+)?(?:appeal|it|her\s+appeal|his\s+appeal|the\s+appeal|this\s+appeal)', response_text, re.IGNORECASE)) or \
                         bool(re.search(r'I\s+(?:grant|deny|allow)(?:\s+(?:the|this)\s+appeal)?(?:\s+on\s+[\w\s]+grounds)?', response_text, re.IGNORECASE)) or \
                         bool(re.search(r'(?:appeal|claim|application)?\s*should\s+be\s+(?:granted|denied|dismissed|allowed|refused|approved)', response_text, re.IGNORECASE)) or \
                         bool(re.search(r'leading\s+me\s+to\s+(?:grant|deny)\s+(?:this\s+|his\s+|her\s+|their\s+)?(?:appeal|claim|application)', response_text, re.IGNORECASE)) or \
                         bool(re.search(r'Appellant\s+is\s+(?:granted|denied)', response_text, re.IGNORECASE)) or \
                         bool(re.search(r'(?:grant|deny)\s+asylum', response_text, re.IGNORECASE)) or \
                         bool(re.search(r'(?:grant|deny)\s+(?:their|his|her)\s+asylum\s+claim', response_text, re.IGNORECASE)) or \
                         bool(re.search(r'(?:^|\n)\s*(?:Granted|Denied|Allowed|Dismissed)', response_text, re.IGNORECASE))
    text_after_end = len(response_text) > len(cleaned_text) if cleaned_text else False
    
    # Extract or preserve sample_id for consistent matching
    sample_id = case_metadata.get('sample_id')
    if sample_id is None:
        # Generate sample_id from consistent fields if missing (for older inference files)
        sample_id = generate_sample_id_from_metadata(case_metadata)
    
    return {
        'sample_id': sample_id,  # Preserve or generate sample_id for matching
        'decision': decision,
        'reasoning': reasoning,
        'original_response': response_text,
        'cleaned_response': cleaned_text,
        'has_end_marker': has_end_marker,
        'has_decision_format': has_decision_format,
        'text_after_end': text_after_end,
        'metadata': case_metadata,
        'quality_metrics': {
            'response_length': len(cleaned_text.split()) if cleaned_text else 0,
            'original_length': len(response_text.split()) if response_text else 0,
            'has_decision': decision is not None,
            'has_reasoning': reasoning is not None and len(reasoning.split()) > 5,
            'follows_format': has_end_marker and has_decision_format,
            'text_reduction_ratio': len(cleaned_text) / len(response_text) if response_text else 1.0
        }
    }

def generate_sample_id_from_metadata(metadata: Dict[str, Any]) -> str:
    """
    Generate a consistent sample_id from metadata for older inference files.
    This creates a deterministic ID that can be used for matching across datasets.
    """
    # Extract key fields for ID generation
    topic = metadata.get('topic', '')
    meta_topic = metadata.get('meta_topic', '')
    fields = metadata.get('fields', {})
    
    # Build a deterministic key from consistent fields
    key_parts = [
        meta_topic,
        topic,
        str(fields.get('country', '')),
        str(fields.get('age', '')),
        str(fields.get('gender', '')),
        str(fields.get('religion', '')),
        str(fields.get('name', '')),
    ]
    
    # Add any ordinal fields for additional uniqueness
    for key, value in metadata.items():
        if key.startswith('fields.') and key.endswith('__ordinal'):
            key_parts.append(f"{key}:{value}")
    
    # Create hash-based ID for deterministic but compact representation
    import hashlib
    key_string = '|'.join(key_parts)
    sample_hash = hashlib.md5(key_string.encode()).hexdigest()[:8]
    
    # Return a readable sample_id
    return f"meta_{sample_hash}"

def calculate_summary_statistics(processed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate summary statistics for the processed results"""
    total = len(processed_results)
    granted = sum(1 for r in processed_results if r['decision'] == 'Granted')
    denied = sum(1 for r in processed_results if r['decision'] == 'Denied')
    no_decision = sum(1 for r in processed_results if r['decision'] is None)
    
    has_reasoning = sum(1 for r in processed_results if r['quality_metrics']['has_reasoning'])
    follows_format = sum(1 for r in processed_results if r['quality_metrics']['follows_format'])
    has_end_marker = sum(1 for r in processed_results if r['has_end_marker'])
    has_decision_format = sum(1 for r in processed_results if r['has_decision_format'])
    
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
            'follows_format_rate': follows_format / total if total > 0 else 0,
            'has_end_marker_rate': has_end_marker / total if total > 0 else 0,
            'has_decision_format_rate': has_decision_format / total if total > 0 else 0,
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
    print(f"  - Follows format: {summary_stats['quality_metrics']['follows_format_rate']:.1%}")
    print(f"  - Has *END* marker: {summary_stats['quality_metrics']['has_end_marker_rate']:.1%}")
    print(f"  - Has Decision: format: {summary_stats['quality_metrics']['has_decision_format_rate']:.1%}")

if __name__ == "__main__":
    main() 