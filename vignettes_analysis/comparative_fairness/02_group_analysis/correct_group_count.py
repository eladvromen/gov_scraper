"""
Correctly count the actual number of groups for fairness analysis
"""

import json
from pathlib import Path

def correct_group_count():
    """Show the correct number of groups for analysis"""
    
    # Load the comprehensive analysis
    base_dir = Path(__file__).parent.parent
    analysis_file = base_dir / "outputs" / "group_analysis" / "comprehensive_group_analysis.json"
    
    with open(analysis_file, 'r') as f:
        data = json.load(f)
    
    print("=== CORRECTED GROUP COUNT ANALYSIS ===\n")
    
    # Individual attributes
    print("=== INDIVIDUAL PROTECTED ATTRIBUTES ===")
    
    individual_total = 0
    for attr, analysis in data['individual_groups'].items():
        attr_total = 0
        for topic, groups in analysis['valid_groups'].items():
            attr_total += len(groups[attr])
        
        individual_total += attr_total
        print(f"{attr.upper()}:")
        print(f"  Total valid groups across all topics: {attr_total}")
        
        # Show per topic breakdown
        print(f"  Per topic breakdown:")
        for topic, groups in analysis['valid_groups'].items():
            num_groups = len(groups[attr])
            print(f"    {topic}: {num_groups} groups")
        print()
    
    print(f"TOTAL INDIVIDUAL GROUPS: {individual_total}\n")
    
    # Intersectional groups
    print("=== INTERSECTIONAL GROUPS ===")
    
    intersectional_total = 0
    for intersection, analysis in data['intersectional_groups'].items():
        inter_total = 0
        for topic, groups in analysis['valid_groups'].items():
            if intersection in groups:
                inter_total += len(groups[intersection])
        
        intersectional_total += inter_total
        print(f"{intersection.upper()}:")
        print(f"  Total valid groups across all topics: {inter_total}")
        
        # Show per topic breakdown
        print(f"  Per topic breakdown:")
        for topic, groups in analysis['valid_groups'].items():
            if intersection in groups:
                num_groups = len(groups[intersection])
                print(f"    {topic}: {num_groups} groups")
        print()
    
    print(f"TOTAL INTERSECTIONAL GROUPS: {intersectional_total}\n")
    
    # Grand total
    grand_total = individual_total + intersectional_total
    print(f"=== GRAND TOTAL ===")
    print(f"Individual groups: {individual_total}")
    print(f"Intersectional groups: {intersectional_total}")
    print(f"TOTAL GROUPS FOR FAIRNESS ANALYSIS: {grand_total}")
    
    # Show example group values
    print(f"\n=== EXAMPLE GROUP VALUES ===")
    
    # Show country groups for one topic
    country_groups = data['individual_groups']['country']['valid_groups']['Nature of persecution']['country']
    print(f"Country groups for 'Nature of persecution' topic:")
    for group in country_groups:
        print(f"  {group['value']}: {group['post_brexit_count']} post, {group['pre_brexit_count']} pre")
    
    # Show gender groups for one topic
    gender_groups = data['individual_groups']['gender']['valid_groups']['Nature of persecution']['gender']
    print(f"\nGender groups for 'Nature of persecution' topic:")
    for group in gender_groups:
        print(f"  {group['value']}: {group['post_brexit_count']} post, {group['pre_brexit_count']} pre")
    
    return {
        'individual_groups': individual_total,
        'intersectional_groups': intersectional_total,
        'total_groups': grand_total
    }

if __name__ == "__main__":
    results = correct_group_count() 