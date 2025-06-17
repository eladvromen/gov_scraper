import json
import sys
import os

# Add parent directory to path to import field_definitions
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from vignettes.field_definitions import *

def analyze_reduction_scenarios():
    """Analyze different reduction scenarios"""
    
    print("=== CURRENT STATE ===")
    print(f"short_names: {len(short_names)} -> {short_names}")
    print(f"short_countries: {len(short_countries)} -> {short_countries}")
    print(f"short_religions: {len(short_religions)} -> {short_religions}")
    print(f"full_religions: {len(full_religions)} -> {full_religions}")
    print(f"short_genders: {len(short_genders)} -> {short_genders}")
    print(f"full_genders: {len(full_genders)} -> {full_genders}")
    print(f"country_b: {len(country_b)} -> {country_b}")
    print()
    
    # Calculate current baseline generic field combinations
    current_base_generic = len(short_names) * len(short_countries) * len(short_religions) * len(short_genders)
    current_full_generic = len(short_names) * len(short_countries) * len(full_religions) * len(full_genders)
    current_with_country_b = len(short_names) * len(short_countries) * len(short_religions) * len(short_genders) * len(country_b)
    
    print("=== REDUCTION SCENARIOS ===")
    
    # Scenario 1: Reduce names to 3 (keep intersectionality)
    print("SCENARIO 1: Reduce names 5→3 (keep gender diversity)")
    reduced_names = 3  # e.g., keep Daniel, Fatima, Muhammad
    new_base = reduced_names * len(short_countries) * len(short_religions) * len(short_genders)
    reduction_1 = (current_base_generic - new_base) / current_base_generic * 100
    print(f"  Base generic: {current_base_generic} → {new_base} ({reduction_1:.1f}% reduction)")
    
    # Scenario 2: Reduce countries to 4 (keep regional diversity)
    print("\nSCENARIO 2: Reduce countries 6→4 (keep regional diversity)")
    reduced_countries = 4  # e.g., Syria, Nigeria, Pakistan, Ukraine
    new_base = len(short_names) * reduced_countries * len(short_religions) * len(short_genders)
    reduction_2 = (current_base_generic - new_base) / current_base_generic * 100
    print(f"  Base generic: {current_base_generic} → {new_base} ({reduction_2:.1f}% reduction)")
    
    # Scenario 3: Reduce country_b to 3 (keep diversity)
    print("\nSCENARIO 3: Reduce country_b 5→3")
    reduced_country_b = 3  # e.g., France, Rwanda, Turkey
    new_with_b = len(short_names) * len(short_countries) * len(short_religions) * len(short_genders) * reduced_country_b
    reduction_3 = (current_with_country_b - new_with_b) / current_with_country_b * 100
    print(f"  With country_b: {current_with_country_b} → {new_with_b} ({reduction_3:.1f}% reduction)")
    
    # Scenario 4: Combined reduction
    print("\nSCENARIO 4: Combined (names 5→3, countries 6→4, country_b 5→3)")
    combined_base = reduced_names * reduced_countries * len(short_religions) * len(short_genders)
    combined_with_b = reduced_names * reduced_countries * len(short_religions) * len(short_genders) * reduced_country_b
    combined_reduction = (current_base_generic - combined_base) / current_base_generic * 100
    print(f"  Base generic: {current_base_generic} → {combined_base} ({combined_reduction:.1f}% reduction)")
    
    # Calculate impact on top vignettes
    print("\n=== IMPACT ON TOP PERMUTATION VIGNETTES ===")
    
    # Firm settlement (current: 72,900)
    current_firm = 27 * (len(short_names) * 3 * len(short_religions) * len(short_genders) * len(short_countries) * len(country_b))
    new_firm = 27 * (reduced_names * 3 * len(short_religions) * len(short_genders) * reduced_countries * reduced_country_b)
    firm_reduction = (current_firm - new_firm) / current_firm * 100
    print(f"Firm settlement: {current_firm:,} → {new_firm:,} ({firm_reduction:.1f}% reduction)")
    
    # PSG (current: 22,680)
    current_psg = 14 * (len(short_names) * 3 * len(full_religions) * len(full_genders) * len(short_countries))
    new_psg = 14 * (reduced_names * 3 * len(full_religions) * len(full_genders) * reduced_countries)
    psg_reduction = (current_psg - new_psg) / current_psg * 100
    print(f"PSG: {current_psg:,} → {new_psg:,} ({psg_reduction:.1f}% reduction)")
    
    # Estimate total reduction
    print(f"\n=== ESTIMATED TOTAL REDUCTION ===")
    print(f"Current total: ~204,660")
    print(f"Estimated new total: ~{204660 * (1 - combined_reduction/100):,.0f}")
    print(f"Overall reduction: ~{combined_reduction:.1f}%")

def suggest_strategic_reductions():
    """Suggest specific strategic reductions"""
    print("\n" + "="*60)
    print("STRATEGIC REDUCTION RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. NAMES (5→3): Keep maximum intersectionality")
    print("   KEEP: Daniel (M/F, Western), Fatima (F, Muslim), Muhammad (M, Muslim)")
    print("   REMOVE: Kwame, Ling")
    print("   → Preserves: Gender fluidity, religious diversity, cultural diversity")
    
    print("\n2. COUNTRIES (6→4): Keep regional/conflict diversity") 
    print("   KEEP: Syria (Middle East, active conflict), Nigeria (Africa, persecution)")
    print("         Pakistan (South Asia, religious), Ukraine (Europe, recent conflict)")
    print("   REMOVE: China, Myanmar")
    print("   → Preserves: Geographic, conflict type, religious context diversity")
    
    print("\n3. COUNTRY_B (5→3): Keep system diversity")
    print("   KEEP: France (functioning), Rwanda (poor systems), Turkey (discriminatory)")
    print("   REMOVE: Bulgaria, South Sudan")
    print("   → Preserves: Full spectrum of asylum system quality")
    
    print("\n4. ALTERNATIVE: Sampling approach")
    print("   Instead of reducing lists, use stratified random sampling:")
    print("   - Generate all permutations")
    print("   - Sample 20-30K ensuring representation across all intersections")
    print("   - Maintains full intersectionality while reducing computation")

if __name__ == "__main__":
    analyze_reduction_scenarios()
    suggest_strategic_reductions() 