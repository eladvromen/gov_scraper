import spacy

# Test text from the case
test_text = """
IMMIGRATION APPEAL TRIBUNAL
Fadil Dyli (Protection – UNMIK – Arif – IFA – Art1D) Kosovo CG *
[2000] UKIAT 00001
Date of hearing: 08/08/2000
Date Determination notified: 30/08/2000
Before:
Mr C. M. G. Ockelton (Deputy President)
Mr P. R. Moulden
Mr M. W. Rapinet
Between
FADIL DYLI
APPELLANT
and
The Secretary of State for the Home Department
RESPONDENT
"""

def demo_anonymization():
    # Load spaCy model
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    
    # Process the text with spaCy
    doc = nlp(test_text)
    
    # Print all entities found
    print("Entities found in the text:")
    for ent in doc.ents:
        print(f"{ent.text} ({ent.label_})")
    
    # Anonymize the text
    anonymized = test_text
    for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
        replacement = f"<<{ent.label_}>>"
        anonymized = anonymized[:ent.start_char] + replacement + anonymized[ent.end_char:]
    
    # Print the anonymized text
    print("\nAnonymized text:")
    print(anonymized)
    
    # Now demonstrate with custom entity replacements
    custom_replacements = {
        'PERSON': '<<PERSON>>',
        'ORG': '<<ORGANIZATION>>',
        'GPE': '<<LOCATION>>',
        'DATE': '<<DATE>>',
    }
    
    anonymized_custom = test_text
    for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
        if ent.label_ in custom_replacements:
            replacement = custom_replacements[ent.label_]
        else:
            replacement = f"<<{ent.label_}>>"
        anonymized_custom = anonymized_custom[:ent.start_char] + replacement + anonymized_custom[ent.end_char:]
    
    print("\nAnonymized text with custom replacements:")
    print(anonymized_custom)

if __name__ == "__main__":
    demo_anonymization() 