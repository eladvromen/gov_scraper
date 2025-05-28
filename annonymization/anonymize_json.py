import spacy
import json
import os
import argparse
from collections import defaultdict

def load_json_file(file_path):
    """Load and parse a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def anonymize_text(text, nlp, entity_types=None, custom_replacements=None):
    """
    Anonymize entities in text using spaCy.
    
    Args:
        text: Text to anonymize
        nlp: spaCy language model
        entity_types: List of entity types to anonymize (None for all)
        custom_replacements: Dict mapping entity types to custom replacement text
    """
    if not text or not isinstance(text, str):
        return text, {}
    
    doc = nlp(text)
    anonymized = text
    found_entities = defaultdict(list)
    
    # Default replacements
    replacements = {
        'PERSON': '<<PERSON>>',
        'ORG': '<<ORGANIZATION>>',
        'GPE': '<<LOCATION>>',
        'LOC': '<<LOCATION>>',
        'FAC': '<<FACILITY>>',
        'NORP': '<<GROUP>>',
        'DATE': '<<DATE>>',
    }
    
    # Update with custom replacements if provided
    if custom_replacements:
        replacements.update(custom_replacements)
    
    # Sort entities by position (reversed to avoid index shifting)
    entities = sorted(doc.ents, key=lambda e: e.start_char, reverse=True)
    
    for ent in entities:
        # Skip if we're only processing specific entity types and this isn't one of them
        if entity_types and ent.label_ not in entity_types:
            continue
            
        # Use custom replacement if available, otherwise use the entity label
        if ent.label_ in replacements:
            replacement = replacements[ent.label_]
        else:
            replacement = f"<<{ent.label_}>>"
            
        found_entities[ent.label_].append(ent.text)
        anonymized = anonymized[:ent.start_char] + replacement + anonymized[ent.end_char:]
    
    return anonymized, dict(found_entities)

def anonymize_json(json_data, nlp, entity_types=None, custom_replacements=None, sensitive_keys=None):
    """
    Recursively anonymize all string values in a JSON object.
    
    Args:
        json_data: JSON data to anonymize
        nlp: spaCy language model
        entity_types: List of entity types to anonymize (None for all)
        custom_replacements: Dict mapping entity types to custom replacement text
        sensitive_keys: List of keys that should always be anonymized completely
    """
    all_entities = defaultdict(list)
    
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            # If the key is in sensitive_keys, replace the whole value
            if sensitive_keys and key in sensitive_keys and isinstance(value, str):
                if key.lower() in ['name', 'person', 'appellant', 'appellant_name']:
                    json_data[key] = '<<PERSON>>'
                    all_entities['SENSITIVE_KEY'].append(f"{key}: {value}")
                elif 'judge' in key.lower():
                    json_data[key] = '<<JUDGES>>'
                    all_entities['SENSITIVE_KEY'].append(f"{key}: {value}")
                else:
                    json_data[key] = f'<<{key.upper()}>>'
                    all_entities['SENSITIVE_KEY'].append(f"{key}: {value}")
            elif isinstance(value, str):
                json_data[key], entities = anonymize_text(value, nlp, entity_types, custom_replacements)
                for entity_type, items in entities.items():
                    all_entities[entity_type].extend(items)
            elif isinstance(value, (dict, list)):
                entities = anonymize_json(value, nlp, entity_types, custom_replacements, sensitive_keys)
                for entity_type, items in entities.items():
                    all_entities[entity_type].extend(items)
    
    elif isinstance(json_data, list):
        for i, item in enumerate(json_data):
            if isinstance(item, str):
                json_data[i], entities = anonymize_text(item, nlp, entity_types, custom_replacements)
                for entity_type, items in entities.items():
                    all_entities[entity_type].extend(items)
            elif isinstance(item, (dict, list)):
                entities = anonymize_json(item, nlp, entity_types, custom_replacements, sensitive_keys)
                for entity_type, items in entities.items():
                    all_entities[entity_type].extend(items)
    
    return all_entities

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Anonymize entities in JSON files using spaCy')
    parser.add_argument('--file', '-f', type=str, default='data/json/[2000] UKIAT 1.json',
                        help='Path to the JSON file to anonymize')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save the anonymized JSON file (default: add "anonymized_" prefix)')
    parser.add_argument('--model', '-m', type=str, default='en_core_web_sm',
                        help='spaCy model to use for entity recognition')
    parser.add_argument('--entities', '-e', type=str, nargs='+',
                        help='Entity types to anonymize (default: all)')
    
    args = parser.parse_args()
    
    # Set file paths
    file_path = args.file
    output_path = args.output
    if not output_path:
        output_path = os.path.join(os.path.dirname(file_path), "anonymized_" + os.path.basename(file_path))
    
    # Load spaCy model
    print(f"Loading spaCy model: {args.model}...")
    try:
        nlp = spacy.load(args.model)
    except OSError:
        print(f"Model {args.model} not found. Please download it using:")
        print(f"python -m spacy download {args.model}")
        return
    
    # Load the JSON data
    print(f"Loading file: {file_path}")
    try:
        json_data = load_json_file(file_path)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON file: {e}")
        return
    
    # Define sensitive keys that should always be anonymized
    sensitive_keys = [
        'appellant_name', 'judges', 'appellant', 'respondent', 'case_title',
        'applicant', 'defendant', 'claimant', 'petitioner'
    ]
    
    # Custom replacements for specific entity types
    custom_replacements = {
        'PERSON': '<<PERSON>>',
        'ORG': '<<ORGANIZATION>>',
        'GPE': '<<LOCATION>>',
        'LOC': '<<LOCATION>>',
        'DATE': '<<DATE>>',
        'LAW': '<<LEGAL_REFERENCE>>'
    }
    
    # Anonymize the data
    print("Anonymizing data...")
    all_entities = anonymize_json(
        json_data, 
        nlp, 
        entity_types=args.entities,
        custom_replacements=custom_replacements,
        sensitive_keys=sensitive_keys
    )
    
    # Save the anonymized data
    print(f"Saving anonymized data to: {output_path}")
    save_json_file(json_data, output_path)
    
    # Print summary of anonymized entities
    print("\nEntities found and anonymized:")
    total_unique = 0
    
    for entity_type, items in sorted(all_entities.items()):
        # Remove duplicates while preserving order
        unique_items = []
        for item in items:
            if item not in unique_items:
                unique_items.append(item)
        
        total_unique += len(unique_items)
        print(f"\n{entity_type} ({len(unique_items)} unique entities):")
        for item in unique_items:
            print(f"  - {item}")
    
    print(f"\nTotal unique entities anonymized: {total_unique}")

if __name__ == "__main__":
    main() 