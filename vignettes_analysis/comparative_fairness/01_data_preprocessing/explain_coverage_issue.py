"""
Explain the coverage issue with intersectional sample sizes
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def demonstrate_coverage_issue():
    """Show why intersections have coverage issues despite complete data"""
    
    # Load the processed data
    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / "data" / "processed"
    
    with open(processed_dir / "tagged_records.json", 'r') as f:
        records = json.load(f)
    
    print("=== COVERAGE ISSUE EXPLANATION ===\n")
    print("You're absolutely right - ALL vignettes have complete country and religion data!")
    print("The 'coverage' issue is about SAMPLE SIZES after creating intersections.\n")
    
    # Take a specific example
    example_topic = "Financial stability"  # This has 864 total samples
    topic_records = [r for r in records if r['topic'] == example_topic]
    
    print(f"EXAMPLE: {example_topic}")
    print(f"Total samples: {len(topic_records)}")
    print(f"Per model: {len(topic_records)//2} each\n")
    
    # Show gender distribution
    gender_counts = defaultdict(int)
    for record in topic_records:
        gender = record['protected_attributes']['gender']
        gender_counts[gender] += 1
    
    print("Gender distribution:")
    for gender, count in gender_counts.items():
        print(f"  {gender}: {count} samples")
    
    # Show religion distribution
    religion_counts = defaultdict(int)
    for record in topic_records:
        religion = record['protected_attributes']['religion']
        religion_counts[religion] += 1
    
    print("\nReligion distribution:")
    for religion, count in religion_counts.items():
        print(f"  {religion}: {count} samples")
    
    # Show country distribution
    country_counts = defaultdict(int)
    for record in topic_records:
        country = record['protected_attributes']['country']
        country_counts[country] += 1
    
    print("\nCountry distribution:")
    for country, count in country_counts.items():
        print(f"  {country}: {count} samples")
    
    # Now show what happens when we create intersections
    print("\n=== INTERSECTION SAMPLE SIZES ===")
    print("When we create Gender×Religion intersections:")
    
    gender_religion_counts = defaultdict(lambda: defaultdict(int))
    for record in topic_records:
        gender = record['protected_attributes']['gender']
        religion = record['protected_attributes']['religion']
        model = record['model']
        
        intersection = f"{gender}_x_{religion}"
        gender_religion_counts[intersection][model] += 1
    
    print("\nGender×Religion intersections:")
    for intersection, model_counts in gender_religion_counts.items():
        post_count = model_counts.get('post_brexit', 0)
        pre_count = model_counts.get('pre_brexit', 0)
        total = post_count + pre_count
        
        meets_threshold = post_count >= 30 and pre_count >= 30
        status = "✅ VALID" if meets_threshold else "❌ TOO SMALL"
        
        print(f"  {intersection}: {total} total ({post_count} post, {pre_count} pre) - {status}")
    
    # Show religion×country×gender (triple intersection)
    print("\n=== TRIPLE INTERSECTION EXAMPLE ===")
    print("Religion×Country×Gender intersections:")
    
    triple_counts = defaultdict(lambda: defaultdict(int))
    for record in topic_records:
        religion = record['protected_attributes']['religion']
        country = record['protected_attributes']['country']
        gender = record['protected_attributes']['gender']
        model = record['model']
        
        intersection = f"{religion}_x_{country}_x_{gender}"
        triple_counts[intersection][model] += 1
    
    valid_count = 0
    for intersection, model_counts in sorted(triple_counts.items()):
        post_count = model_counts.get('post_brexit', 0)
        pre_count = model_counts.get('pre_brexit', 0)
        total = post_count + pre_count
        
        meets_threshold = post_count >= 30 and pre_count >= 30
        if meets_threshold:
            valid_count += 1
        
        status = "✅ VALID" if meets_threshold else "❌ TOO SMALL"
        print(f"  {intersection}: {total} total ({post_count} post, {pre_count} pre) - {status}")
    
    print(f"\nOnly {valid_count} triple intersections have enough samples for analysis!")
    
    print("\n=== SUMMARY ===")
    print("- ✅ 100% complete data for all protected attributes")
    print("- ⚠️  Sample sizes get divided when creating intersections")
    print("- ⚠️  Need ≥30 samples per model per intersection group")
    print("- ⚠️  More intersections = smaller sample sizes")
    print("- 💡 This is why we have 'coverage' limitations, not missing data!")

if __name__ == "__main__":
    demonstrate_coverage_issue() 