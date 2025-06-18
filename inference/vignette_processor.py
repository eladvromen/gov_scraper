#!/usr/bin/env python3
"""
DEPRECATED: This file has been replaced by utils.py

This script is kept for backwards compatibility and to generate analytics-ready JSONL files.
For new inference work, use the functions in utils.py instead.
"""

import os
from utils import load_vignettes, generate_analytics_records

def save_vignettes_jsonl(records, out_path):
    """Save records to JSONL format."""
    import json
    with open(out_path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    vignette_json_path = os.path.join('vignettes', 'complete_vignettes.json')
    output_jsonl_path = os.path.join('inference', 'vignette_analytics_ready.jsonl')

    vignettes = load_vignettes(vignette_json_path)
    analytics_records = generate_analytics_records(vignettes)
    save_vignettes_jsonl(analytics_records, output_jsonl_path)
    print(f"Generated {len(analytics_records)} analytics-ready vignettes and saved to {output_jsonl_path}") 