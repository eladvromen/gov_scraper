import json
from collections import defaultdict

def load_processed_data(file_path: str):
    """Load processed inference results."""
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data["processed_results"]

def check_key_collisions(data, model_name):
    """Check for key collisions in the data."""
    key_to_cases = defaultdict(list)
    
    for i, case in enumerate(data):
        metadata = case['metadata']
        key = f"{metadata['topic']}_{metadata['fields']['country']}_{metadata['fields']['age']}"
        key_to_cases[key].append({
            'index': i,
            'case': case,
            'full_id': f"{metadata['fields']['name']}_{metadata['fields']['gender']}_{metadata['fields']['religion']}"
        })
    
    print(f"\n=== {model_name} Dataset Analysis ===")
    print(f"Total cases: {len(data)}")
    print(f"Unique keys: {len(key_to_cases)}")
    
    collisions = {k: v for k, v in key_to_cases.items() if len(v) > 1}
    if collisions:
        print(f"KEY COLLISIONS FOUND: {len(collisions)} keys have multiple cases!")
        for key, cases in collisions.items():
            print(f"\nKey: {key}")
            for case_info in cases:
                metadata = case_info['case']['metadata']
                print(f"  - {metadata['fields']['name']} ({metadata['fields']['gender']}, {metadata['fields']['religion']})")
                print(f"    Vignette: {metadata['vignette_text'][:100]}...")
    else:
        print("No key collisions found.")
    
    return key_to_cases

def check_cross_dataset_matching(pre_brexit_keys, post_brexit_keys):
    """Check how cases match between datasets."""
    print(f"\n=== Cross-Dataset Matching Analysis ===")
    
    common_keys = set(pre_brexit_keys.keys()) & set(post_brexit_keys.keys())
    pre_only = set(pre_brexit_keys.keys()) - set(post_brexit_keys.keys())
    post_only = set(post_brexit_keys.keys()) - set(pre_brexit_keys.keys())
    
    print(f"Common keys: {len(common_keys)}")
    print(f"Pre-Brexit only: {len(pre_only)}")
    print(f"Post-Brexit only: {len(post_only)}")
    
    # Check for mismatched vignettes with same key
    mismatched_vignettes = []
    for key in common_keys:
        pre_cases = pre_brexit_keys[key]
        post_cases = post_brexit_keys[key]
        
        # If either side has multiple cases, we need to check carefully
        if len(pre_cases) > 1 or len(post_cases) > 1:
            print(f"\nWARNING: Multiple cases for key {key}")
            print(f"  Pre-Brexit: {len(pre_cases)} cases")
            print(f"  Post-Brexit: {len(post_cases)} cases")
            
        # Compare vignettes for the first case of each (what the viewer does)
        pre_vignette = pre_cases[0]['case']['metadata']['vignette_text']
        post_vignette = post_cases[0]['case']['metadata']['vignette_text']
        
        if pre_vignette != post_vignette:
            mismatched_vignettes.append({
                'key': key,
                'pre_vignette': pre_vignette,
                'post_vignette': post_vignette,
                'pre_name': pre_cases[0]['case']['metadata']['fields']['name'],
                'post_name': post_cases[0]['case']['metadata']['fields']['name']
            })
    
    if mismatched_vignettes:
        print(f"\nMISMATCHED VIGNETTES FOUND: {len(mismatched_vignettes)} cases with same key but different vignettes!")
        for mismatch in mismatched_vignettes[:5]:  # Show first 5
            print(f"\nKey: {mismatch['key']}")
            print(f"Pre-Brexit name: {mismatch['pre_name']}")
            print(f"Post-Brexit name: {mismatch['post_name']}")
            print(f"Pre-Brexit vignette: {mismatch['pre_vignette'][:150]}...")
            print(f"Post-Brexit vignette: {mismatch['post_vignette'][:150]}...")
    else:
        print("No mismatched vignettes found.")
    
    return mismatched_vignettes

def main():
    # File paths
    pre_brexit_file = "../inference/results/processed/processed_subset_inference_llama3_8b_pre_brexit_2013_2016_instruct_20250623_120225_20250623_122325.json"
    post_brexit_file = "../inference/results/processed/processed_subset_inference_llama3_8b_post_brexit_2019_2025_instruct_20250623_123821_20250623_124925.json"
    
    # Load data
    print("Loading data...")
    pre_brexit_data = load_processed_data(pre_brexit_file)
    post_brexit_data = load_processed_data(post_brexit_file)
    
    # Check for key collisions in each dataset
    pre_brexit_keys = check_key_collisions(pre_brexit_data, "Pre-Brexit")
    post_brexit_keys = check_key_collisions(post_brexit_data, "Post-Brexit")
    
    # Check cross-dataset matching
    mismatches = check_cross_dataset_matching(pre_brexit_keys, post_brexit_keys)
    
    # Suggest better key generation
    print(f"\n=== Recommendation ===")
    if mismatches or any(len(cases) > 1 for cases in pre_brexit_keys.values()) or any(len(cases) > 1 for cases in post_brexit_keys.values()):
        print("ISSUE DETECTED: Current key generation is insufficient!")
        print("Recommended fix: Include more fields in the key generation:")
        print("  - Add gender, religion, and name to the key")
        print("  - Or use a hash of the entire vignette text")
        print("  - This will ensure proper case matching between datasets")
    else:
        print("Key generation appears to be working correctly.")

if __name__ == "__main__":
    main() 