import json
import sys
import os

# Add vignettes directory to path for imports
vignettes_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../vignettes'))
sys.path.insert(0, vignettes_path)

from field_definitions import *

def get_list_size(list_name):
    """Get the size of a list from field_definitions.py"""
    list_mapping = {
        'short_names': len(short_names),
        'short_countries': len(short_countries),
        'short_religions': len(short_religions),
        'full_religions': len(full_religions),
        'short_genders': len(short_genders),
        'full_genders': len(full_genders),
        'short_ages_vulnerability': len(short_ages_vulnerability),
        'short_ages_workforce': len(short_ages_workforce),
        'country_b': len(country_b)
    }
    return list_mapping.get(list_name, 1)

def calculate_vignette_permutations(vignette):
    """Calculate total permutations for a single vignette"""
    
    # Count ordinal field permutations
    ordinal_count = 1
    ordinal_details = {}
    if 'ordinal_fields' in vignette and vignette['ordinal_fields']:
        for field_name, options in vignette['ordinal_fields'].items():
            field_size = len(options)
            ordinal_count *= field_size
            ordinal_details[field_name] = field_size
    
    # Count horizontal field permutations
    horizontal_count = 1
    horizontal_details = {}
    if 'horizontal_fields' in vignette and vignette['horizontal_fields']:
        for field_name, options in vignette['horizontal_fields'].items():
            field_size = len(options)
            horizontal_count *= field_size
            horizontal_details[field_name] = field_size
    
    # Count generic field permutations
    generic_count = 1
    generic_details = {}
    if 'generic_fields' in vignette and vignette['generic_fields']:
        for field_name, list_name in vignette['generic_fields'].items():
            field_size = get_list_size(list_name)
            generic_count *= field_size
            generic_details[field_name] = f"{list_name} ({field_size})"
    
    total = ordinal_count * horizontal_count * generic_count
    
    return {
        'ordinal_count': ordinal_count,
        'ordinal_details': ordinal_details,
        'horizontal_count': horizontal_count,
        'horizontal_details': horizontal_details,
        'generic_count': generic_count,
        'generic_details': generic_details,
        'total': total
    }

def main():
    # Load the vignettes
    with open('vignettes/complete_vignettes.json', 'r') as f:
        vignettes = json.load(f)
    
    print(f"Found {len(vignettes)} vignettes")
    print("="*80)
    
    total_permutations = 0
    
    for i, vignette in enumerate(vignettes):
        print(f"\nVignette {i+1}: {vignette.get('topic', 'Unknown')}")
        print(f"Meta Topic: {vignette.get('meta_topic', 'Unknown')}")
        
        result = calculate_vignette_permutations(vignette)
        total_permutations += result['total']
        
        print(f"  Ordinal fields: {result['ordinal_count']}")
        if result['ordinal_details']:
            for field, count in result['ordinal_details'].items():
                print(f"    - {field}: {count} options")
        
        print(f"  Horizontal fields: {result['horizontal_count']}")
        if result['horizontal_details']:
            for field, count in result['horizontal_details'].items():
                print(f"    - {field}: {count} options")
                
        print(f"  Generic fields: {result['generic_count']}")
        if result['generic_details']:
            for field, details in result['generic_details'].items():
                print(f"    - {field}: {details}")
        
        print(f"  → Total for this vignette: {result['total']:,}")
    
    print("\n" + "="*80)
    print(f"TOTAL PERMUTATIONS ACROSS ALL VIGNETTES: {total_permutations:,}")
    print("="*80)
    
    # Show field definitions summary
    print(f"\nField Definitions Summary:")
    print(f"  short_names: {len(short_names)} names")
    print(f"  short_countries: {len(short_countries)} countries") 
    print(f"  short_religions: {len(short_religions)} religions")
    print(f"  full_religions: {len(full_religions)} religions")
    print(f"  short_genders: {len(short_genders)} genders")
    print(f"  full_genders: {len(full_genders)} genders")
    print(f"  short_ages_vulnerability: {len(short_ages_vulnerability)} ages")
    print(f"  short_ages_workforce: {len(short_ages_workforce)} ages")
    print(f"  country_b: {len(country_b)} countries")

if __name__ == "__main__":
    main() 