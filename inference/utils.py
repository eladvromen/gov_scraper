"""
Utility functions for the inference pipeline.
This module contains the minimal set of functions needed for vignette processing and inference.
"""

import json
import os
import sys
from itertools import product

# Add vignettes directory to path for imports
vignettes_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../vignettes'))
sys.path.insert(0, vignettes_path)

from field_definitions import *
from field_definitions import get_verb_present_third_person, get_verb_past_be, get_pronoun_possessive

def load_vignettes(json_path):
    """Load vignettes from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def resolve_field_reference(field_ref):
    """Resolve field reference to actual list values."""
    field_map = {
        "short_names": short_names,
        "short_countries": short_countries,
        "short_religions": short_religions,
        "full_religions": full_religions,
        "short_genders": short_genders,
        "full_genders": full_genders,
        "short_ages_vulnerability": short_ages_vulnerability,
        "short_ages_workforce": short_ages_workforce,
        "country_b": country_b
    }
    return field_map.get(field_ref, [field_ref])

def generate_analytics_records(vignettes):
    """
    Generate analytics records for all vignette permutations.
    This is the core function used by the inference pipeline.
    """
    records = []
    for vignette in vignettes:
        generic_fields = vignette.get('generic_fields', {})
        ordinal_fields = vignette.get('ordinal_fields', {})
        horizontal_fields = vignette.get('horizontal_fields', {})
        derived_fields = vignette.get('derived_fields', {})

        generic_keys = list(generic_fields.keys())
        generic_lists = [resolve_field_reference(generic_fields[k]) for k in generic_keys]

        ordinal_keys = list(ordinal_fields.keys())
        ordinal_lists = [list(ordinal_fields[k].keys()) for k in ordinal_keys]

        horizontal_keys = list(horizontal_fields.keys())
        horizontal_lists = [horizontal_fields[k] for k in horizontal_keys]

        for generic_vals in product(*generic_lists):
            for ordinal_vals in product(*ordinal_lists) if ordinal_lists else [()]:
                for horizontal_vals in product(*horizontal_lists) if horizontal_lists else [()]:
                    sample_values = {}
                    # Fill generic
                    for k, v in zip(generic_keys, generic_vals):
                        sample_values[k] = v
                    # Fill ordinal (label and value)
                    for k, v in zip(ordinal_keys, ordinal_vals):
                        sample_values[k] = v
                        sample_values[f"{k}__ordinal"] = ordinal_fields[k][v]
                    # Fill horizontal
                    for k, v in zip(horizontal_keys, horizontal_vals):
                        sample_values[k] = v
                    # Derived fields (e.g., name, country_B)
                    if derived_fields:
                        for dfield, dspec in derived_fields.items():
                            if dfield == "name" and "country" in sample_values and "gender" in sample_values:
                                sample_values["name"] = get_name_for_country_gender(sample_values["country"], sample_values["gender"])
                            if dfield == "country_B":
                                # Use mapping to get possible country_B values
                                mapping = dspec["mapping"]
                                source_field = dspec["source_field"]
                                if mapping == "systems_to_countries_map":
                                    val = sample_values.get(source_field)
                                    if val and val in systems_to_countries_map:
                                        sample_values["country_B"] = systems_to_countries_map[val][0]  # pick first for determinism
                                elif mapping == "safety_to_countries_map":
                                    val = sample_values.get(source_field)
                                    if val and val in safety_to_countries_map:
                                        sample_values["country_B"] = safety_to_countries_map[val][0]
                    # Pronoun and grammar helpers
                    if 'gender' in sample_values:
                        gender = sample_values['gender']
                        sample_values['pronoun'] = get_pronoun(gender)
                        sample_values['pronoun_was_were'] = get_verb_past_be(gender)
                        sample_values['pronoun_possessive'] = get_pronoun_possessive(gender)
                        # Common verb forms
                        sample_values['pronoun_suffers'] = get_verb_present_third_person(gender, 'suffer')
                        sample_values['pronoun_lives'] = get_verb_present_third_person(gender, 'live')
                        sample_values['pronoun_works'] = get_verb_present_third_person(gender, 'work')
                    # Fill template
                    vignette_text = vignette["vignette_template"].format(**sample_values)
                    # Build analytics record
                    record = {
                        'meta_topic': vignette['meta_topic'],
                        'topic': vignette['topic'],
                        'fields': {k: sample_values.get(k) for k in list(generic_keys) + list(ordinal_keys) + list(horizontal_keys) + list(derived_fields.keys()) if k in sample_values or f"{k}__ordinal" in sample_values},
                        # Add ordinal values as separate fields
                        **{f"fields.{k}__ordinal": sample_values[f"{k}__ordinal"] for k in ordinal_keys if f"{k}__ordinal" in sample_values},
                        'vignette_text': vignette_text,
                        'model_response': ''  # Placeholder for model free-text response
                    }
                    records.append(record)
    return records 