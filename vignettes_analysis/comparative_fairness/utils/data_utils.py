"""
Data utilities for comparative fairness analysis
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import yaml
import logging

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_inference_data(file_path: str) -> Dict:
    """Load inference results from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_protected_attributes(record: Dict) -> Dict:
    """Extract protected attributes from a record"""
    metadata = record.get('metadata', {})
    fields = metadata.get('fields', {})
    
    # Extract the four protected attributes
    protected_attrs = {
        'country': fields.get('country', 'Unknown'),
        'age': fields.get('age', 'Unknown'),
        'religion': fields.get('religion', 'Unknown'),
        'gender': fields.get('gender', 'Unknown')
    }
    
    return protected_attrs

def normalize_decision(decision: str) -> str:
    """Normalize decision to GRANT/DENY"""
    if decision.lower() in ['granted', 'grant']:
        return 'GRANT'
    elif decision.lower() in ['denied', 'deny']:
        return 'DENY'
    else:
        return 'INCONCLUSIVE'

def create_record_id(record: Dict) -> str:
    """Create unique record identifier"""
    metadata = record.get('metadata', {})
    sample_id = metadata.get('sample_id', 'unknown')
    topic = metadata.get('topic', 'unknown')
    return f"{topic}_{sample_id}"

def tag_record(record: Dict, model_name: str) -> Dict:
    """Tag a record with all necessary information for fairness analysis"""
    metadata = record.get('metadata', {})
    
    tagged_record = {
        'record_id': create_record_id(record),
        'model': model_name,
        'sample_id': metadata.get('sample_id'),
        'topic': metadata.get('topic'),
        'meta_topic': metadata.get('meta_topic'),
        'decision': normalize_decision(record.get('decision', 'Unknown')),
        'protected_attributes': extract_protected_attributes(record),
        'vignette_text': metadata.get('vignette_text', ''),
        'reasoning': record.get('reasoning', ''),
        'original_record': record
    }
    
    return tagged_record

def get_reference_groups(records: List[Dict], config: Dict) -> Dict:
    """Determine reference groups from data"""
    reference_groups = {}
    
    for attr in config['protected_attributes']:
        if config['reference_groups'][attr] == 'TBD':
            # Find most frequent value
            values = [r['protected_attributes'][attr] for r in records 
                     if r['protected_attributes'][attr] != 'Unknown']
            if values:
                reference_groups[attr] = max(set(values), key=values.count)
            else:
                reference_groups[attr] = 'Unknown'
        else:
            reference_groups[attr] = config['reference_groups'][attr]
    
    return reference_groups

def validate_data_quality(records: List[Dict]) -> Dict:
    """Validate data quality for fairness analysis"""
    total_records = len(records)
    
    # Check for missing protected attributes
    missing_attrs = {attr: 0 for attr in ['country', 'age', 'religion', 'gender']}
    missing_decisions = 0
    missing_topics = 0
    
    for record in records:
        if record['decision'] == 'INCONCLUSIVE':
            missing_decisions += 1
        if not record['topic']:
            missing_topics += 1
        for attr in missing_attrs:
            if record['protected_attributes'][attr] == 'Unknown':
                missing_attrs[attr] += 1
    
    quality_report = {
        'total_records': total_records,
        'missing_decisions': missing_decisions,
        'missing_topics': missing_topics,
        'missing_attributes': missing_attrs,
        'missing_decisions_pct': (missing_decisions / total_records) * 100,
        'missing_topics_pct': (missing_topics / total_records) * 100,
        'missing_attributes_pct': {
            attr: (count / total_records) * 100 
            for attr, count in missing_attrs.items()
        }
    }
    
    return quality_report

def setup_logging(log_file: str = None) -> logging.Logger:
    """Setup logging for the analysis"""
    logger = logging.getLogger('comparative_fairness')
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 