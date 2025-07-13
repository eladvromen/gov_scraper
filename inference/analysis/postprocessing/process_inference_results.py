#!/usr/bin/env python3
"""
Post-processing script for inference results from UK immigration LLM study.

This script processes raw model responses to extract clean decisions and reasoning.
Enhanced decision categorization supports:
- Granted: Full approval
- Denied: Full denial  
- Inconclusive: Mixed decisions (PARTIALLY GRANTED) or postponements (Adjourned for Further Hearing)

Usage:
    python process_inference_results.py <input_file_path>
    
Example:
    python process_inference_results.py ../results/subset_inference_llama3_8b_pre_brexit_20250620_120959.json
"""

import json
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# --- Decision Extraction ---

# Map decision phrases to canonical forms
DECISION_VARIANTS = {
    'denied': 'Denied',
    'granted': 'Granted',
    'dismissed': 'Denied',
    'allowed': 'Granted',
    'refused': 'Denied',
    'approved': 'Granted',
    'allow': 'Granted',
    'deny': 'Denied',
    'grant': 'Granted',  # "grant" = Granted
    'asylum': 'Granted',  # "grant asylum" = Granted, "I grant him asylum" = Granted
    'allows': 'Granted',  # "The Tribunal allows" = Granted
    'dismisses': 'Denied',  # "The Tribunal dismisses" = Denied
    'dismiss': 'Denied',   # "leading me to dismiss" = Denied
    'denial': 'Denied',   # "leading to its denial" = Denied
    'grants': 'Granted',  # "the Tribunal grants" = Granted
    # New inconclusive decision categories
    'partially granted': 'Inconclusive',
    'partial': 'Inconclusive',
    'adjourned for further hearing': 'Inconclusive',
    'adjourned': 'Inconclusive',
    # Humanitarian protection is considered granted
    'humanitarian protection': 'Granted',
    # Leave granted is typically considered a positive outcome
    'leave': 'Granted',
    # Pending cases are inconclusive for now
    'pending': 'Inconclusive',
    # Viable applications are considered ongoing/inconclusive
    'viable': 'Inconclusive',
    # Allowed is equivalent to granted
    'allowed': 'Granted',
}

# Patterns specifically for the format: "Decision: DENIED/denied *END*"
DECISION_PATTERNS = [
    # Primary pattern: Decision/Decided: [specific_word] followed by *END*
    r'(?:Decision|DECISION|Decided):\s*(?P<decision>denied|granted|dismissed|allowed|refused|approved|partially\s+granted|adjourned\s+for\s+further\s+hearing|adjourned)\s*\*END\*',
    # Fallback: Decision/Decided: [specific_word] without *END* requirement
    r'(?:Decision|DECISION|Decided):\s*(?P<decision>denied|granted|dismissed|allowed|refused|approved|partially\s+granted|adjourned\s+for\s+further\s+hearing|adjourned)',
    # Decision with compound phrases: "Decision: ASYLUM GRANTED", "Decision: Appeal denied"
    r'(?:Decision|DECISION|Decided):\s*(?:ASYLUM\s+|Appeal\s+)?(?P<decision_compound>granted|denied|dismissed|allowed|refused|approved|partially\s+granted)',
    # NEW: Decision with "Asylum appeal" format: "Decision: Asylum appeal denied"
    r'(?:Decision|DECISION|Decided):\s*Asylum\s+appeal\s+(?P<decision_asylum_appeal>granted|denied|dismissed|allowed|refused|approved)',
    # NEW: Decision with "Asylum application" format: "Decision: Asylum application refused"
    r'(?:Decision|DECISION|Decided):\s*Asylum\s+application\s+(?P<decision_asylum_application>granted|denied|dismissed|allowed|refused|approved)',
    # Alternative format: *GRANTED* or *DENIED* or *PARTIALLY GRANTED*
    r'\*(?P<decision2>granted|denied|dismissed|allowed|refused|approved|partially\s+granted|adjourned)\*',
    # NEW: "I *DECISION*" format: "I *DENY*", "I *GRANT*"
    r'I\s+\*(?P<decision_asterisk>granted|denied|dismissed|allowed|refused|approved|grant|deny)\*',
    # NEW: Asterisk format with ASYLUM: *GRANTED ASYLUM*, *DENIED ASYLUM*
    r'\*(?P<decision20>granted|denied)\s+asylum\*',
    # NEW: Asterisk format with ASYLUM CLAIM: *GRANTED ASYLUM CLAIM*, *DENIED ASYLUM CLAIM*
    r'\*(?P<decision21>granted|denied)\s+asylum\s+claim\*',
    # NEW: The appeal of [name]... is granted/denied format
    r'(?:The\s+)?appeal\s+of\s+\w+,?\s+.*?,?\s+is\s+(?P<decision_name>granted|denied|dismissed|allowed|refused|approved)',
    # NEW: The Tribunal allows/dismisses this appeal format
    r'(?:The\s+)?Tribunal\s+(?P<decision_tribunal>allows|dismisses)\s+(?:this\s+)?appeal',
    # Appeal/claim/application format: "The appeal/claim/application is [accordingly] granted/denied"
    r'(?:appeal|claim|application)\s+is\s+(?:accordingly\s+)?(?P<decision3>granted|denied|dismissed|allowed|refused|approved)',
    # Application for asylum format: "application for asylum is granted/denied"
    r'application\s+for\s+asylum\s+(?:in\s+the\s+UK\s+)?is\s+(?:therefore\s+)?(?P<decision3c>granted|denied|dismissed|allowed|refused|approved)',
    # NEW: "must therefore be" format: "application for asylum must therefore be DENIED"
    r'(?:application|claim)\s+for\s+asylum\s+must\s+therefore\s+be\s+(?P<decision_must_be>granted|denied|dismissed|allowed|refused|approved)',
    # NEW: More general "must be" format: "must be granted/denied"
    r'must\s+be\s+(?P<decision_must>granted|denied|dismissed|allowed|refused|approved)',
    # Claim for asylum format: "claim for asylum is denied/granted"
    r'claim\s+for\s+asylum\s+is\s+(?:accordingly\s+)?(?P<decision3b>granted|denied|dismissed|allowed|refused|approved)',
    # NEW: "claim for asylum must therefore be" format
    r'claim\s+for\s+asylum\s+must\s+therefore\s+be\s+(?P<decision_claim_must>granted|denied|dismissed|allowed|refused|approved)',
    # Case format: "case dismissed/granted/denied"
    r'case\s+(?P<decision4>dismissed|granted|denied|allowed|refused|approved)',
    # Judge statements: "I allow/deny/grant [name]'s appeal" or "I allow/deny/grant [the] appeal"
    r'I\s+(?P<decision5>allow|deny|grant)\s+(?:\w+\'s\s+)?(?:appeal|it|her\s+appeal|his\s+appeal|the\s+appeal|this\s+appeal)',
    # NEW: "I deny/grant [pronoun] application for asylum" format
    r'I\s+(?P<decision_i_pronoun>deny|grant)\s+(?:his|her|their|this)\s+application\s+for\s+asylum',
    # NEW: "I deny/grant application for asylum" format (more general)
    r'I\s+(?P<decision_i_application>deny|grant)\s+(?:the\s+|this\s+)?application\s+for\s+asylum',
    # NEW: "I deny/grant claim for asylum" format
    r'I\s+(?P<decision_i_claim>deny|grant)\s+(?:his|her|their|this|the)\s+claim\s+for\s+asylum',
    # Judge statements without object: "I grant/deny/allow" (standalone)
    r'I\s+(?P<decision5b>grant|deny|allow)(?:\s+(?:the|this)\s+appeal)?(?:\s+on\s+[\w\s]+grounds)?[.\s]*(?:\*END\*|$)',
    # "Should be" format: "appeal should be GRANTED/DENIED"
    r'(?:appeal|claim|application)?\s*should\s+be\s+(?P<decision5c>granted|denied|dismissed|allowed|refused|approved)',
    # Leading phrases: "leading me to grant/deny"
    r'leading\s+me\s+to\s+(?P<decision5d>grant|deny)\s+(?:this\s+|his\s+|her\s+|their\s+)?(?:appeal|claim|application)',
    # Appellant format: "The Appellant is granted/denied"
    r'(?:The\s+)?Appellant\s+is\s+(?P<decision6>granted|denied|dismissed|allowed|refused|approved)',
    # Asylum format: "grant/deny asylum"
    r'(?:grant|deny)\s+(?P<decision7>asylum)',
    # NEW: Humanitarian protection format: "grant humanitarian protection"
    r'(?P<decision_humanitarian>grant|deny)\s+humanitarian\s+protection',
    # NEW: "I grant humanitarian protection" format
    r'I\s+(?P<decision_i_humanitarian>grant)\s+humanitarian\s+protection',
    # Asylum claim format: "deny their asylum claim"
    r'(?P<decision7b>grant|deny)\s+(?:their|his|her)\s+asylum\s+claim',
    # Standalone decisions (more restrictive context)
    r'(?:^|\n)\s*(?P<decision8>Granted|Denied|Allowed|Dismissed|Refused|Approved)\s*(?:\n|\*END\*|$)',
    # Claim/application granted/denied: "her claim for asylum is granted", "application is denied"
    r'(?:her|his|their|the)\s+(?:claim|application)\s+(?:for\s+asylum\s+)?is\s+(?:accordingly\s+)?(?P<decision9>granted|denied)',
    # Protection granted: "her claim for international protection is granted"
    r'(?:claim|application)\s+for\s+international\s+protection\s+is\s+(?P<decision10>granted|denied)',
    # Specific UK asylum patterns: "Her claim for asylum in the UK is granted"
    r'(?:Her|His|Their)\s+claim\s+for\s+asylum\s+in\s+the\s+UK\s+is\s+(?P<decision11>granted|denied)',
    # Direct grant by judge: "I grant him asylum", "I grant her asylum"
    r'I\s+grant\s+(?:him|her|them)\s+(?P<decision12>asylum)',
    # NEW: Flexible "I ... grant/deny asylum" pattern: "I find ... and accordingly grant him asylum"
    r'I\s+.*?\s+(?P<decision25>grant|deny)\s+(?:him|her|them|the\s+appellant)\s+asylum',
    # Credibility denial: "His application lacks credibility and is denied"
    r'(?:His|Her|Their)\s+application\s+lacks\s+credibility\s+and\s+is\s+(?P<decision13>denied|refused)',
    # Appeal granted: "This appeal is therefore granted on asylum grounds"
    r'This\s+appeal\s+is\s+therefore\s+(?P<decision14>granted|denied)\s+on\s+asylum\s+grounds',
    # NEW PATTERNS FROM FAILED EXTRACTIONS:
    # Passive voice with pronouns: "She/He/They has been granted/denied", "She/He/They was granted/denied"
    r'(?:She|He|They|(?:\w+))\s+(?:has\s+been|was|is)\s+(?P<decision15>granted|denied|dismissed|allowed|refused|approved)',
    # Incomplete sentence with grant/deny: "She has been granted" (even without explicit object)
    r'(?:She|He|They|(?:\w+))\s+(?:has\s+been|was|is)\s+(?P<decision16>granted|denied)\s*\*?END\*?',
    # Leading me to deny (more flexible): "leading me to deny/grant [object]"
    r'leading\s+me\s+to\s+(?P<decision17>grant|deny|allow|dismiss)\s+(?:the\s+)?(?:appeal|claim|application|it)?',
    # Missing passive constructions: "is granted/denied", "are granted/denied"
    r'(?:is|are)\s+(?:therefore\s+|accordingly\s+)?(?P<decision18>granted|denied|dismissed|allowed|refused|approved)',
    # Name-specific patterns: "[Name] has been granted/denied"
    r'\w+\s+has\s+been\s+(?P<decision19>granted|denied|dismissed|allowed|refused|approved)',
    # NEW: Appeal against denial pattern: "appeal against denial of asylum is granted/denied"
    r'appeal\s+against\s+denial\s+of\s+asylum\s+is\s+(?P<decision22>granted|denied|dismissed|allowed|refused|approved)',
    # NEW: "Shall be" pattern: "application for asylum shall be granted/denied"
    r'(?:application|claim)\s+for\s+asylum\s+shall\s+be\s+(?P<decision23>granted|denied|dismissed|allowed|refused|approved)',
    # NEW: More general "shall be" pattern: "shall be granted/denied"
    r'shall\s+be\s+(?P<decision24>granted|denied|dismissed|allowed|refused|approved)',
    # NEW: "I conclude that" pattern: "I conclude that her asylum claim is credible and should be granted"
    r'I\s+conclude\s+that\s+.*?\s+(?:is|should\s+be)\s+(?P<decision26>granted|denied|dismissed|allowed|refused|approved)',
    # NEW: "I therefore" pattern: "I therefore grant this appeal"
    r'I\s+therefore\s+(?P<decision27>grant|deny|allow|dismiss)\s+(?:this\s+)?(?:appeal|claim|application)',
    # NEW: "credible and granted/denied" pattern: "Her claim for asylum is credible and granted"
    r'(?:claim|application)\s+(?:for\s+asylum\s+)?is\s+credible\s+and\s+(?P<decision28>granted|denied|dismissed|allowed|refused|approved)',
    # NEW: "I find [pronoun] application for asylum granted" pattern
    r'I\s+find\s+(?:his|her|their|the)\s+application\s+for\s+asylum\s+(?P<decision_i_find_app>granted|denied|dismissed|allowed|refused|approved)',
    # NEW: "I grant this claimant asylum under paragraph" pattern
    r'I\s+(?P<decision_claimant_paragraph>grant)\s+this\s+claimant\s+asylum\s+under\s+paragraph\s+[\w\s()]+',
    # NEW: "I find [description] and thus granted/denied" pattern
    r'I\s+find\s+.*?\s+and\s+thus\s+(?P<decision_i_find_thus>granted|denied|dismissed|allowed|refused|approved)',
    # NEW: "claim is substantiated by credible evidence and thus granted" pattern
    r'claim\s+is\s+substantiated\s+by\s+credible\s+evidence\s+and\s+thus\s+(?P<decision_substantiated>granted|denied|dismissed|allowed|refused|approved)',
    # NEW: "I find his claim credible and grant this appeal" pattern
    r'I\s+find\s+(?:his|her|their|the\s+appellant\'s)\s+claim\s+credible\s+and\s+(?P<decision_credible_and>grant|deny)\s+(?:this\s+)?(?:appeal|claim|application)',
    # NEW: "I find his claim credible and therefore grant" pattern
    r'I\s+find\s+(?:his|her|their|the\s+appellant\'s)\s+claim\s+credible\s+and\s+therefore\s+(?P<decision_credible_therefore>grant|deny)\s+(?:this\s+)?(?:appeal|claim|application)',
    # NEW: "I deny this asylum claim" pattern (more specific than general "I deny")
    r'I\s+(?P<decision_i_asylum_claim>deny|grant)\s+this\s+asylum\s+claim',
    # NEW: "Asylum claim denied/granted" at start of line or after period
    r'(?:^|\.)\s*Asylum\s+claim\s+(?P<decision_asylum_claim_status>denied|granted|dismissed|allowed|refused|approved)',
    # NEW: "I therefore grant [possessive] appeal on asylum grounds" pattern
    r'I\s+therefore\s+(?P<decision_therefore_appeal>grant|deny)\s+(?:his|her|their|this)\s+appeal\s+on\s+asylum\s+grounds',
    # NEW: "I find that there is a [description] and grant this appeal on asylum grounds" pattern
    r'I\s+find\s+that\s+there\s+is\s+.*?\s+and\s+(?P<decision_find_that_grant>grant|deny)\s+this\s+appeal\s+on\s+asylum\s+grounds',
    # NEW: "I grant [pronoun] leave pending [description]" pattern
    r'I\s+(?P<decision_grant_leave>grant|deny)\s+(?:him|her|them|this\s+claimant)\s+leave\s+pending\s+.*?',
    # NEW: "I grant [pronoun] leave pending a full hearing of [possessive] appeal" pattern
    r'I\s+(?P<decision_grant_leave_hearing>grant|deny)\s+(?:him|her|them)\s+leave\s+pending\s+a\s+full\s+hearing\s+of\s+(?:his|her|their)\s+appeal',
    # NEW: "I find this appeal granted/denied" pattern
    r'I\s+find\s+this\s+appeal\s+(?P<decision_find_appeal>granted|denied|dismissed|allowed|refused|approved)',
    # NEW: "I find [description] substantiated, and grant this appeal on asylum grounds" pattern
    r'I\s+find\s+.*?\s+substantiated,?\s+and\s+(?P<decision_substantiated_grant>grant|deny)\s+this\s+appeal\s+on\s+asylum\s+grounds',
    # NEW: "I accept [description] as credible and grant this appeal on asylum grounds" pattern
    r'I\s+accept\s+.*?\s+as\s+credible\s+and\s+(?P<decision_credible_grant>grant|deny)\s+this\s+appeal\s+on\s+asylum\s+grounds',
    # NEW: "*GRANTED [description]*" or "*DENIED [description]*" pattern
    r'\*(?P<decision_asterisk_description>GRANTED|DENIED)\s+.*?\*',
    # NEW: "I deny this asylum appeal" pattern
    r'I\s+(?P<decision_deny_asylum_appeal>deny|grant)\s+this\s+asylum\s+appeal',
    # NEW: "grant this appeal on asylum grounds" (without I) pattern
    r'(?P<decision_grant_appeal_grounds>grant|deny)\s+this\s+appeal\s+on\s+asylum\s+grounds',
    # NEW: "I grant [pronoun] asylum" (more general) pattern
    r'I\s+(?P<decision_i_grant_asylum>grant|deny)\s+(?:him|her|them|the\s+appellant)\s+asylum',
    # NEW: "claim will be granted as a refugee" pattern
    r'(?:claim|appeal)\s+will\s+be\s+(?P<decision_will_be>granted|denied|dismissed|allowed|refused|approved)\s+as\s+a\s+refugee',
    # NEW: "claim will be granted under Article [X]" pattern
    r'(?:claim|appeal)\s+will\s+be\s+(?P<decision_will_be_article>granted|denied|dismissed|allowed|refused|approved)\s+under\s+Article\s+\d+',
    # NEW: "prompting this appeal to be GRANTED/DENIED" pattern
    r'prompting\s+this\s+appeal\s+to\s+be\s+(?P<decision_prompting>GRANTED|DENIED|DISMISSED|ALLOWED|REFUSED|APPROVED)',
    # NEW: "Decision: PENDING/GRANTED/DENIED" format
    r'Decision:\s*(?P<decision_format>PENDING|GRANTED|DENIED|DISMISSED|ALLOWED|REFUSED|APPROVED)',
    # NEW: "His claim will be granted" (more general) pattern
    r'(?:His|Her|Their|The)\s+(?:claim|appeal)\s+will\s+be\s+(?P<decision_possessive_will>granted|denied|dismissed|allowed|refused|approved)',
    # NEW: "remain pending" pattern (for pending cases)
    r'(?:remain|remains)\s+(?P<decision_pending>pending)',
    # NEW: "application remains viable" pattern (for pending/continuing cases)
    r'application\s+remains\s+(?P<decision_viable>viable)',
    # NEW: Final batch - "I deny the application" pattern
    r'I\s+(?P<decision_i_deny_app>deny)\s+the\s+application',
    # NEW: "leading to denial of asylum" pattern
    r'leading\s+to\s+(?P<decision_leading_denial>denial)\s+of\s+asylum',
    # NEW: "I find [description] and grant this appeal" pattern
    r'I\s+find\s+.*?\s+and\s+(?P<decision_i_find_grant>grant)\s+this\s+appeal',
    # NEW: "I find that there is a well-founded fear of persecution and grant the appeal" pattern
    r'I\s+find\s+that\s+there\s+is\s+a\s+well-founded\s+fear\s+of\s+persecution\s+and\s+(?P<decision_well_founded_grant>grant)\s+the\s+appeal',
    # NEW: "I find the appeal GRANTED/DENIED" pattern
    r'I\s+find\s+the\s+appeal\s+(?P<decision_i_find_appeal>GRANTED|DENIED|granted|denied)',
    # NEW: "I hereby GRANTED ASYLUM/DENIED ASYLUM" pattern
    r'I\s+hereby\s+(?P<decision_i_hereby>GRANTED|DENIED|granted|denied)\s+(?:ASYLUM|asylum)',
    # NEW: "has been granted asylum" pattern
    r'has\s+been\s+(?P<decision_has_been_granted>granted)\s+asylum',
    # NEW: "leading to its denial" pattern
    r'leading\s+to\s+its\s+(?P<decision_leading_its>denial)',
    # NEW: "I find that [name] faces persecution and therefore grant this appeal" pattern
    r'I\s+find\s+that\s+.*?\s+faces\s+persecution\s+.*?\s+and\s+therefore\s+(?P<decision_faces_persecution_grant>grant)\s+this\s+appeal',
    # NEW: "has been granted" (general) pattern
    r'(?:appeal|claim)\s+.*?\s+has\s+been\s+(?P<decision_has_been>granted|denied)',
    # NEW: "has been allowed" pattern
    r'has\s+been\s+(?P<decision_has_been_allowed>allowed)\s+as\s+their\s+asylum\s+claim',
    # NEW: "[name]'s fears are well-founded and grant this appeal" pattern
    r'(?:\w+)\'s\s+fears\s+are\s+well-founded\s+and\s+(?P<decision_fears_grant>grant)\s+this\s+appeal',
    # NEW: "*Decision*: GRANTED/DENIED" pattern (with asterisks)
    r'\*Decision\*:\s*(?P<decision_asterisk_decision>GRANTED|DENIED|granted|denied)',
    # NEW: "I deny [possessive] asylum appeal" pattern
    r'I\s+(?P<decision_i_deny_asylum_appeal>deny)\s+(?:his|her|their)\s+asylum\s+appeal',
    # NEW: "I find it reasonable to grant [description]" pattern
    r'I\s+find\s+it\s+reasonable\s+to\s+(?P<decision_reasonable_grant>grant)\s+(?:him|her|them)\s+humanitarian\s+protection',
    # NEW: "[description] claim stands granted" pattern
    r'(?:asylum|appeal)\s+claim\s+stands\s+(?P<decision_stands>granted|denied)',
    # NEW: "the Tribunal grants asylum and humanitarian protection" pattern
    r'the\s+Tribunal\s+(?P<decision_tribunal_grants>grants)\s+asylum\s+and\s+humanitarian\s+protection',
    # NEW: "the Tribunal grants asylum/humanitarian protection/human rights claim" pattern
    r'the\s+Tribunal\s+(?P<decision_tribunal_grants_multiple>grants)\s+asylum[/\s]+humanitarian\s+protection[/\s]+human\s+rights\s+claim',
    # NEW: "we deny [possessive] appeal" pattern
    r'we\s+(?P<decision_we_deny>deny)\s+(?:his|her|their)\s+appeal',
    # NEW: "grant the appeal, allowing [name] to remain" pattern
    r'(?P<decision_grant_appeal_allowing>grant)\s+the\s+appeal,\s+allowing\s+\w+\s+to\s+remain',
    # NEW: "which I hereby grant" pattern
    r'which\s+I\s+hereby\s+(?P<decision_which_hereby>grant)',
    # NEW: "I grant the appeal, allowing" pattern
    r'I\s+(?P<decision_i_grant_appeal_allowing>grant)\s+the\s+appeal,\s+allowing',
    # NEW: "I take into account [description] and grant the appeal" pattern
    r'I\s+take\s+into\s+account\s+.*?\s+and\s+(?P<decision_take_account_grant>grant)\s+the\s+appeal',
    # NEW: "allowing [name] to remain in the United Kingdom as a refugee" pattern
    r'allowing\s+\w+\s+to\s+remain\s+in\s+the\s+United\s+Kingdom\s+as\s+a\s+(?P<decision_remain_refugee>refugee)',
    # NEW: "[description] necessitate [description] which I hereby grant" pattern
    r'necessitate\s+.*?\s+which\s+I\s+hereby\s+(?P<decision_necessitate_grant>grant)',
    # NEW: "I grant the appeal, allowing [name] to remain" pattern
    r'I\s+(?P<decision_i_grant_allowing>grant)\s+the\s+appeal,\s+allowing\s+\w+\s+to\s+remain',
    # NEW: "which I hereby grant/deny" pattern
    r'which\s+I\s+hereby\s+(?P<decision_which_hereby_alt2>grant|deny|GRANT|DENY)',
    # NEW: "I therefore deny [possessive] appeal" pattern
    r'I\s+therefore\s+(?P<decision_therefore_deny>deny)\s+(?:his|her|their)\s+appeal',
    # NEW: "therefore deny the appeal" pattern
    r'therefore\s+(?P<decision_therefore_deny_general>deny)\s+the\s+appeal',
    # NEW: "I deny their appeal for [reason]" pattern
    r'I\s+(?P<decision_i_deny_their>deny)\s+their\s+appeal\s+for',
    # NEW: "I therefore *DENY/GRANT* [possessive] appeal" pattern (with asterisks)
    r'I\s+therefore\s+\*(?P<decision_therefore_asterisk>DENY|GRANT|deny|grant)\*\s+(?:his|her|their)\s+appeal',
    # NEW: "I therefore deny [possessive] appeal under [legal reference]" pattern
    r'I\s+therefore\s+(?P<decision_therefore_under>deny)\s+(?:his|her|their)\s+appeal\s+under\s+the\s+Refugee\s+Convention',
    # NEW: Specific missing patterns from failed extractions
    # Pattern: "I deny the application" (Sample 1855)
    r'I\s+(?P<decision_deny_application>deny)\s+the\s+application',
    # Pattern: "Decision: Asylum claim denied" (Sample 2524)
    r'Decision:\s*Asylum\s+claim\s+(?P<decision_asylum_claim_denied>denied)',
    # Pattern: "I find [description] and grant this appeal" (Sample 2868)
    r'I\s+find\s+.*?\s+and\s+(?P<decision_find_and_grant>grant)\s+this\s+appeal',
    # Pattern: "I find that there is a well-founded fear of persecution and grant the appeal" (Sample 3426)
    r'I\s+find\s+that\s+there\s+is\s+a\s+well-founded\s+fear\s+of\s+persecution\s+and\s+(?P<decision_well_founded_fear>grant)\s+the\s+appeal',
    # Pattern: "I find the appeal GRANTED" (Sample 3682)
    r'I\s+find\s+the\s+appeal\s+(?P<decision_find_appeal_granted>GRANTED|granted|DENIED|denied)',
    # Pattern: "I hereby GRANTED ASYLUM" (Sample 4350)
    r'I\s+hereby\s+(?P<decision_hereby_asylum>GRANTED|granted|DENIED|denied)\s+ASYLUM',
    # Pattern: "has been granted asylum" (Sample 6026)
    r'has\s+been\s+(?P<decision_been_granted_asylum>granted)\s+asylum',
    # Pattern: "leading to its denial" (Sample 6098)
    r'leading\s+to\s+its\s+(?P<decision_leading_its_denial>denial)',
    # Pattern: "therefore grant this appeal" (Sample 6177)
    r'therefore\s+(?P<decision_therefore_grant_appeal>grant)\s+this\s+appeal',
    # Pattern: "has been granted" (general, Sample 6241)
    r'(?:appeal|claim)\s+.*?\s+has\s+been\s+(?P<decision_general_been_granted>granted|denied)',
    # Pattern: "has been allowed" (Sample 6386)
    r'has\s+been\s+(?P<decision_been_allowed>allowed)',
    # NEW: Final few missing patterns
    # Pattern: "*Decision*: GRANTED/DENIED" (with asterisks)
    r'\*Decision\*:\s*(?P<decision_asterisk_colon>GRANTED|DENIED|granted|denied)',
    # Pattern: "I deny his asylum appeal"
    r'I\s+(?P<decision_deny_his_asylum>deny)\s+(?:his|her|their)\s+asylum\s+appeal',
    # Pattern: "I find it reasonable to grant him humanitarian protection"
    r'I\s+find\s+it\s+reasonable\s+to\s+(?P<decision_reasonable_humanitarian>grant)\s+(?:him|her|them)\s+humanitarian\s+protection',
    # Pattern: "asylum claim stands granted"
    r'asylum\s+claim\s+stands\s+(?P<decision_claim_stands>granted|denied)',
    # Pattern: "grant her appeal on asylum grounds"
    r'(?P<decision_grant_appeal_asylum_grounds>grant)\s+(?:her|his|their)\s+appeal\s+on\s+asylum\s+grounds',
    # PATTERNS FROM FAILED EXTRACTIONS:
    # Pattern for "I grant him leave as an acknowledgment" (Sample 21804)
    r'I\s+(?P<decision_grant_leave_acknowledgment>grant)\s+(?:him|her|them)\s+leave\s+as\s+an\s+acknowledgment',
    # Pattern for "credibility is compromised, leading to denial" (Sample 22502)
    r'credibility\s+is\s+compromised,\s+leading\s+to\s+(?P<decision_credibility_compromised>denial)\s+of\s+asylum',
    # Pattern for flexible "Tribunal grants" with slashes (Sample 22511) 
    r'the\s+Tribunal\s+(?P<decision_tribunal_flexible>grants)\s+asylum\/\s+humanitarian\s+protection\/\s+human\s+rights\s+claim',
    # Pattern for "allowing [name] to remain in the United Kingdom as a refugee with access to support services"
    r'(?P<decision_refugee_support>grant)\s+.*?allowing\s+\w+\s+to\s+remain\s+in\s+the\s+United\s+Kingdom\s+as\s+a\s+refugee\s+with\s+access\s+to\s+.*?support\s+services',
    # Pattern for "which I hereby DENY" (Sample 23670) - uppercase version
    r'which\s+I\s+hereby\s+(?P<decision_hereby_caps>DENY|GRANT)',
    # FINAL MISSING PATTERNS FROM LATEST FAILED EXTRACTIONS:
    # Pattern for "I find this appeal denied" - "I find [object] denied/granted"
    r'I\s+find\s+(?:this\s+|the\s+)?(?:appeal|claim|application)\s+(?P<decision_i_find_object>denied|granted|dismissed|allowed|refused|approved)',
    # Pattern for "Decided GRANTED/DENIED" format
    r'Decided\s+(?P<decision_decided>GRANTED|DENIED|granted|denied|dismissed|allowed|refused|approved)',
    # Pattern for "Decisions: GRANTED/DENIED" format (plural)
    r'Decisions:\s*(?P<decision_plural>GRANTED|DENIED|granted|denied|dismissed|allowed|refused|approved)',
    # FINAL BATCH - SPECIFIC PATTERNS FROM LATEST FAILED EXTRACTIONS:
    # Pattern: "therefore, I GRANTED his appeal under Article 1A(2) of the Refugee Convention"
    r'therefore,\s*I\s+(?P<decision_therefore_appeal_1a2>GRANTED|granted)\s+(?:his|her|their)\s+appeal\s+under\s+Article\s+1A\(2\)\s+of\s+the\s+Refugee\s+Convention',
    # Pattern: "it would be unreasonable to expect them to return to Syria; granted"
    r'it\s+would\s+be\s+unreasonable\s+to\s+expect\s+(?:them|him|her)\s+to\s+return\s+to\s+\w+;\s+(?P<decision_unreasonable_return>granted|denied)',
    # Pattern: "I grant their appeal under the Refugee Convention"
    r'I\s+(?P<decision_grant_appeal_convention>grant)\s+(?:their|his|her)\s+appeal\s+under\s+the\s+Refugee\s+Convention',
    # Pattern: "I find that it would be unfair not to grant him an opportunity; Therefore, I GRANTED his appeal"
    r'I\s+find\s+that\s+it\s+would\s+be\s+unfair\s+not\s+to\s+grant\s+(?:him|her|them)\s+an\s+opportunity.*?Therefore,\s*I\s+(?P<decision_unfair_opportunity>GRANTED|granted)\s+(?:his|her|their)\s+appeal',
    # Pattern: "granted asylum status" at end
    r'(?P<decision_granted_asylum_status>granted)\s+asylum\s+status\.\s*\*END\*',
    # Pattern: "granting them international protection"
    r'granting\s+(?:them|him|her)\s+(?P<decision_granting_international>international\s+protection)',
    # Pattern: "leading the appeal being denied"
    r'leading\s+the\s+appeal\s+being\s+(?P<decision_leading_appeal>denied|granted)',
    # Pattern: "denial of asylum claim" at end
    r'(?P<decision_denial_asylum_claim>denial)\s+of\s+asylum\s+claim\.\s*\*END\*',
    # Pattern: "I have denied" at end (incomplete sentence)
    r'\*END\*\s*I\s+have\s+(?P<decision_i_have_denied>denied)',
    # Pattern: "I allow their appeal, granting them international protection"
    r'I\s+(?P<decision_allow_appeal_protection>allow)\s+(?:their|his|her)\s+appeal,\s+granting\s+(?:them|him|her)\s+international\s+protection',
    # Pattern for "which remains granted" - "remains granted/denied"
    r'(?:which\s+)?remains\s+(?P<decision_remains>granted|denied|dismissed|allowed|refused|approved)',
    # Pattern for decisions appearing right after *END* marker: *END*"denied" or *END* denied
    r'\*END\*["\s]*(?P<decision_after_end>denied|granted|dismissed|allowed|refused|approved)',
    # Pattern: "I find her application denied" - "I find [object] denied/granted"
    r'I\s+find\s+(?:her|his|their|the)\s+(?:application|claim|appeal)\s+(?P<decision_i_find_application>denied|granted|dismissed|allowed|refused|approved)',
    # Pattern: "making this appeal GRANTED" - "making this [object] GRANTED/DENIED"
    r'making\s+this\s+(?:appeal|claim|application)\s+(?P<decision_making_this>GRANTED|DENIED|granted|denied|dismissed|allowed|refused|approved)',
    # Pattern: "warranting GRANTED status" - "warranting [decision] status"
    r'warranting\s+(?P<decision_warranting>GRANTED|DENIED|granted|denied|dismissed|allowed|refused|approved)\s+status',
    # Pattern: "I grant [name] asylum status in the United Kingdom as a[n] refugee"
    r'I\s+(?P<decision_grant_refugee_status>grant)\s+\w+\s+asylum\s+status\s+in\s+the\s+United\s+Kingdom\s+as\s+an?\s+refugee',
    # Pattern: "I grant [name] asylum status" (more general)
    r'I\s+(?P<decision_grant_asylum_status>grant)\s+\w+\s+asylum\s+status',
    # Pattern: "as a refugee with access to support services"
    r'as\s+a\s+refugee\s+with\s+access\s+to\s+(?:.*?)support\s+services',
    # Pattern: "I grant [name] asylum status in accordance with paragraph [ref]"
    r'I\s+(?P<decision_grant_asylum_paragraph>GRANTED|granted)\s+asylum\s+status\s+in\s+accordance\s+with\s+paragraph\s+[\w()]+',
    # Pattern: "I grant [name] asylum status under Article [ref]"
    r'I\s+(?P<decision_grant_asylum_article>GRANTED|granted)\s+asylum\s+status\s+under\s+Article\s+[\w()]+',
    # Pattern: "making this appeal GRANTED/DENIED" (more general)
    r'making\s+this\s+(?:appeal|claim|application)\s+(?P<decision_making_appeal>GRANTED|DENIED|granted|denied|dismissed|allowed|refused|approved)',
    # Pattern: "warranting GRANTED/DENIED status" (more general)
    r'warranting\s+(?P<decision_warranting_status>GRANTED|DENIED|granted|denied|dismissed|allowed|refused|approved)\s+status',
    # NEW PATTERNS FOR REMAINING CASES:
    # Pattern: "I find [possessive] application denied" at end of sentence
    r'I\s+find\s+(?:her|his|their|the)\s+application\s+(?P<decision_find_app_end>denied|granted|dismissed|allowed|refused|approved)[.;]\s*\*END\*',
    # Pattern: "falls within paragraph [ref] of the immigration rules; granted"
    r'falls\s+within\s+paragraph\s+[\w()]+\s+of\s+the\s+immigration\s+rules;\s+(?P<decision_falls_within>granted|denied|dismissed|allowed|refused|approved)',
    # Pattern: "breach of Article [X] ECHR; granted"
    r'breach\s+of\s+Article\s+[\w()]+\s+ECHR;\s+(?P<decision_breach_article>granted|denied|dismissed|allowed|refused|approved)',
    # Pattern: "breach of Article [X] ECHR and humanitarian protection grounds; granted"
    r'breach\s+of\s+Article\s+[\w()]+\s+ECHR\s+and\s+humanitarian\s+protection\s+grounds;\s+(?P<decision_breach_humanitarian>granted|denied|dismissed|allowed|refused|approved)',
    # Pattern: "in accordance with paragraph [ref] HC395"
    r'in\s+accordance\s+with\s+paragraph\s+[\w()]+\s+HC395[.;]\s*(?P<decision_accordance_hc395>GRANTED|granted|DENIED|denied)',
    # Pattern: "in the United Kingdom as a[n] refugee"
    r'in\s+the\s+United\s+Kingdom\s+as\s+an?\s+refugee[.;]\s*(?P<decision_uk_refugee>GRANTED|granted|DENIED|denied)',
    # Pattern: "I find [possessive] application denied" with semicolon
    r'I\s+find\s+(?:her|his|their|the)\s+application\s+(?P<decision_find_app_semi>denied|granted|dismissed|allowed|refused|approved);',
    # Pattern: "I find [possessive] claim denied" with semicolon
    r'I\s+find\s+(?:her|his|their|the)\s+claim\s+(?P<decision_find_claim_semi>denied|granted|dismissed|allowed|refused|approved);',
    # NEW PATTERNS FROM FAILED EXTRACTIONS:
    # Pattern: "asylum granted" at end (Sample: "asylum granted. *END*")
    r'asylum\s+(?P<decision_asylum_end>granted|denied)\.\s*\*END\*',
    # Pattern: "I GRANTED" at end (Sample: "I GRANTED *END*")
    r'I\s+(?P<decision_i_granted_end>GRANTED|granted|DENIED|denied)\s*\*END\*',
    # Pattern: "I GRANTED asylum under the Refugee Convention"
    r'I\s+(?P<decision_i_granted_convention>GRANTED|granted)\s+asylum\s+under\s+the\s+Refugee\s+Convention',
    # Pattern: "I find Nour's appeal GRANTED under Article [ref]"
    r'I\s+find\s+\w+\'s\s+appeal\s+(?P<decision_find_name_appeal>GRANTED|granted|DENIED|denied)\s+under\s+Article\s+[\w()]+',
    # Pattern: "making this appeal GRANTED under Article [ref]"
    r'making\s+this\s+appeal\s+(?P<decision_making_appeal_article>GRANTED|granted|DENIED|denied)\s+under\s+Article\s+[\w()]+',
    # Pattern: "I GRANTED" followed by any text
    r';\s*I\s+(?P<decision_semicolon_i_granted>GRANTED|granted|DENIED|denied)(?:\s|$)',
    # Pattern: "breach of Article [X] ECHR; I GRANTED"
    r'breach\s+of\s+Article\s+[\w()]+\s+ECHR;\s*I\s+(?P<decision_breach_i_granted>GRANTED|granted|DENIED|denied)',
    # Pattern: "humanitarian protection grounds; granted"
    r'humanitarian\s+protection\s+grounds;\s+(?P<decision_humanitarian_grounds>granted|denied|dismissed|allowed|refused|approved)',
    # Pattern: "The risk of harm outweighs any potential benefits of return; asylum granted"
    r'risk\s+of\s+harm\s+outweighs\s+.*?;\s*asylum\s+(?P<decision_risk_outweighs>granted|denied)',
    # Pattern: "real risk of persecution; I GRANTED"
    r'real\s+risk\s+of\s+persecution;\s*I\s+(?P<decision_real_risk_persecution>GRANTED|granted|DENIED|denied)',
    # Pattern: "well-founded fear, I GRANTED asylum under the Refugee Convention"
    r'well-founded\s+fear,?\s*I\s+(?P<decision_well_founded_convention>GRANTED|granted)\s+asylum\s+under\s+the\s+Refugee\s+Convention',
    # Pattern: "unduly harsh, making this appeal GRANTED"
    r'unduly\s+harsh,\s*making\s+this\s+appeal\s+(?P<decision_unduly_harsh>GRANTED|granted|DENIED|denied)',
    # Pattern: "potential breach of Article [X] ECHR; I GRANTED"
    r'potential\s+breach\s+of\s+Article\s+[\w()]+\s+ECHR;\s*I\s+(?P<decision_potential_breach>GRANTED|granted|DENIED|denied)',
    # ADDITIONAL PATTERNS FROM REMAINING FAILED EXTRACTIONS:
    # Pattern: "I find her application denied" (direct format)
    r'I\s+find\s+(?:her|his|their|the)\s+application\s+(?P<decision_find_app_direct>denied|granted|dismissed|allowed|refused|approved)(?:\s*\.|$)',
    # Pattern: "falls within paragraph [ref] of the immigration rules; granted" (with semicolon)
    r'falls\s+within\s+paragraph\s+[\w()]+\s+of\s+the\s+immigration\s+rules;\s+(?P<decision_immigration_rules>granted|denied|dismissed|allowed|refused|approved)',
    # Pattern: "breach Article 3 ECHR; granted" (without "of")
    r'breach\s+Article\s+[\w()]+\s+ECHR;\s+(?P<decision_breach_simple>granted|denied|dismissed|allowed|refused|approved)',
    # Pattern: "Article 3 ECHR/Article 8 ECHR; granted" (multiple articles with slash)
    r'Article\s+[\w()]+\s+ECHR/Article\s+[\w()]+\s+ECHR;\s+(?P<decision_multiple_articles>granted|denied|dismissed|allowed|refused|approved)',
    # Pattern: "under the Refugee Convention; granted"
    r'under\s+the\s+Refugee\s+Convention;\s+(?P<decision_refugee_convention>granted|denied|dismissed|allowed|refused|approved)',
    # Pattern: "warranting GRANTED status" (at end)
    r'warranting\s+(?P<decision_warranting_end>GRANTED|granted|DENIED|denied)\s+status[.\s]*\*END\*',
    # Pattern: "under Article 1A(2) Refugee Convention; granted"
    r'under\s+Article\s+[\w()]+\s+Refugee\s+Convention;\s+(?P<decision_article_refugee>granted|denied|dismissed|allowed|refused|approved)',
    # Pattern: "in accordance with paragraph [ref] HC395" (with period at end)
    r'in\s+accordance\s+with\s+paragraph\s+[\w()]+\s+HC395\.\s*(?P<decision_hc395_period>GRANTED|granted|DENIED|denied)',
    # Pattern: "making this appeal GRANTED" (at end with period)
    r'making\s+this\s+appeal\s+(?P<decision_making_appeal_end>GRANTED|granted|DENIED|denied)\.\s*\*END\*',
    # Pattern: "persecution under Article 1A(2) Refugee Convention; granted"
    r'persecution\s+under\s+Article\s+[\w()]+\s+Refugee\s+Convention;\s+(?P<decision_persecution_article>granted|denied|dismissed|allowed|refused|approved)',
    # Pattern: "amounts to persecution under Article [ref]; granted"
    r'amounts\s+to\s+persecution\s+under\s+Article\s+[\w()]+;\s+(?P<decision_amounts_persecution>granted|denied|dismissed|allowed|refused|approved)',
    # Pattern: "engage Article [X] ECHR as well as [description]; I GRANTED"
    r'engage\s+Article\s+[\w()]+\s+ECHR\s+.*?;\s*I\s+(?P<decision_engage_article>GRANTED|granted|DENIED|denied)',
    # Pattern: "under the Refugee Convention; I GRANTED"
    r'under\s+the\s+Refugee\s+Convention;\s*I\s+(?P<decision_convention_i_granted>GRANTED|granted|DENIED|denied)',
    # MORE AGGRESSIVE PATTERNS FOR REMAINING FAILED EXTRACTIONS:
    # Pattern: "breach Article 3 ECHR; granted" (simplified)
    r'breach\s+Article\s+\d+\s+ECHR;\s+(?P<decision_breach_echr_simple>granted|denied)',
    # Pattern: "Article 3 ECHR/Article 8 ECHR; granted" (multiple articles)
    r'Article\s+\d+\s+ECHR/Article\s+\d+\s+ECHR;\s+(?P<decision_dual_echr>granted|denied)',
    # Pattern: "rendering this appeal GRANTED"
    r'rendering\s+this\s+appeal\s+(?P<decision_rendering>GRANTED|granted|DENIED|denied)',
    # Pattern: "I grant Nour international protection under the Refugee Convention"
    r'I\s+(?P<decision_grant_international>grant)\s+\w+\s+international\s+protection\s+under\s+the\s+Refugee\s+Convention',
    # Pattern: "I deny her application for international protection"
    r'I\s+(?P<decision_deny_international>deny)\s+(?:her|his|their)\s+application\s+for\s+international\s+protection',
    # Pattern: "His application for asylum will be granted"
    r'(?:His|Her|Their)\s+application\s+for\s+asylum\s+will\s+be\s+(?P<decision_will_be_asylum>granted|denied)',
    # Pattern: "granted asylum" at end with period
    r'(?P<decision_granted_asylum_period>granted)\s+asylum\.\s*\*END\*',
    # Pattern: "I GRANTED asylum as there are substantial grounds"
    r'I\s+(?P<decision_granted_substantial>GRANTED|granted)\s+asylum\s+as\s+there\s+are\s+substantial\s+grounds',
    # Pattern: "asylum as there are substantial grounds for believing"
    r'asylum\s+as\s+there\s+are\s+substantial\s+grounds\s+for\s+believing.*?(?P<decision_substantial_grounds>granted|GRANTED)',
    # Pattern: Simple "granted asylum" at end
    r';\s*(?P<decision_semicolon_granted>granted)\s+asylum\.\s*\*END\*',
    # Pattern: "it would be unreasonable... granted asylum"
    r'it\s+would\s+be\s+unreasonable.*?;\s*(?P<decision_unreasonable_granted>granted)\s+asylum',
    # Pattern: "without any support network; granted asylum"
    r'without\s+any\s+support\s+network;\s*(?P<decision_no_support_granted>granted)\s+asylum',
    # Pattern: "unduly harsh for her to relocate... I GRANTED"
    r'unduly\s+harsh\s+for\s+(?:her|him|them)\s+to\s+relocate.*?I\s+(?P<decision_unduly_harsh_granted>GRANTED|granted)',
    # Pattern: "real risk of serious harm due to her faith"
    r'real\s+risk\s+of\s+serious\s+harm\s+due\s+to\s+(?:her|his|their)\s+faith.*?(?P<decision_harm_faith>GRANTED|granted)',
    # FINAL BATCH OF SPECIFIC PATTERNS FROM LATEST FAILED EXTRACTIONS:
    # Pattern: "I deny her application under Article 1A(2) Refugee Convention"
    r'I\s+(?P<decision_deny_article_1a2>deny)\s+(?:her|his|their)\s+application\s+under\s+Article\s+1A\(2\)\s+Refugee\s+Convention',
    # Pattern: "therefore, I GRANTED her asylum claim"
    r'therefore,\s*I\s+(?P<decision_therefore_granted>GRANTED|granted)\s+(?:her|his|their)\s+asylum\s+claim',
    # Pattern: "I GRANTED her asylum claim under Article 1A(2)"
    r'I\s+(?P<decision_granted_asylum_1a2>GRANTED|granted)\s+(?:her|his|their)\s+asylum\s+claim\s+under\s+Article\s+1A\(2\)',
    # Pattern: "I GRANTED her asylum claim under Article 1(A)2"
    r'I\s+(?P<decision_granted_asylum_1a2_alt>GRANTED|granted)\s+(?:her|his|their)\s+asylum\s+claim\s+under\s+Article\s+1\(A\)2',
    # Pattern: "therefore, I GRANTED her asylum"
    r'therefore,\s*I\s+(?P<decision_therefore_granted_asylum>GRANTED|granted)\s+(?:her|his|their)\s+asylum',
    # Pattern: "I GRANTED this appeal"
    r'therefore,\s*I\s+(?P<decision_therefore_granted_appeal>GRANTED|granted)\s+this\s+appeal',
    # Pattern: "asylum denied" at end
    r'asylum\s+(?P<decision_asylum_denied_end>denied)\s*\*END\*',
    # Pattern: "reduces risk factors; asylum denied"
    r'reduces\s+risk\s+factors;\s*asylum\s+(?P<decision_reduces_risk_denied>denied)',
    # Pattern: "This reduces risk factors; asylum denied"
    r'This\s+reduces\s+risk\s+factors;\s*asylum\s+(?P<decision_this_reduces_denied>denied)',
    # Pattern: "; asylum denied *END*"
    r';\s*asylum\s+(?P<decision_semicolon_asylum_denied>denied)\s*\*END\*',
    # Pattern: "fundamental to human dignity; therefore, I GRANTED"
    r'fundamental\s+to\s+human\s+dignity;\s*therefore,\s*I\s+(?P<decision_dignity_granted>GRANTED|granted)',
    # Pattern: "which is fundamental to human dignity; therefore, I GRANTED her asylum"
    r'fundamental\s+to\s+human\s+dignity;\s*therefore,\s*I\s+(?P<decision_dignity_granted_asylum>GRANTED|granted)\s+(?:her|his|their)\s+asylum',
    # Pattern: "outweighs any potential risks; therefore, I GRANTED"
    r'outweighs\s+any\s+potential\s+risks;\s*therefore,\s*I\s+(?P<decision_outweighs_granted>GRANTED|granted)',
    # Pattern: "cumulative circumstances, I GRANTED her asylum claim"
    r'cumulative\s+circumstances,\s*I\s+(?P<decision_cumulative_granted>GRANTED|granted)\s+(?:her|his|their)\s+asylum\s+claim',
    # Pattern: "given the cumulative circumstances, I GRANTED"
    r'given\s+the\s+cumulative\s+circumstances,\s*I\s+(?P<decision_given_cumulative>GRANTED|granted)',
    # Pattern: "I allow their appeal, granting them international protection"
    r'I\s+(?P<decision_allow_appeal_protection>allow)\s+(?:their|his|her)\s+appeal,\s+granting\s+(?:them|him|her)\s+international\s+protection',
    # FINAL TWO PATTERNS - LAST REMAINING FAILED EXTRACTIONS:
    # Pattern: "allow their appeal, granting them international protection" (Sample shows this exact pattern)
    r'(?P<decision_allow_granting_final>allow)\s+their\s+appeal,\s+granting\s+them\s+international\s+protection',
]

def normalize_decision(word: str) -> Optional[str]:
    """Normalize a decision word to canonical form"""
    if not word:
        return None
    
    word = word.lower().strip()
    return DECISION_VARIANTS.get(word, None)

def extract_primary_decision(response_text: str) -> Optional[str]:
    """Extract the primary decision from the response"""
    if not response_text:
        return None
    
    # Find decisions with their positions
    all_decisions = []
    
    for pattern in DECISION_PATTERNS:
        for match in re.finditer(pattern, response_text, re.IGNORECASE):
            try:
                # Try all decision groups
                decision_word = None
                for group_name in ['decision', 'decision_compound', 'decision_asylum_appeal', 'decision_asylum_application', 'decision2', 'decision_asterisk', 'decision_must_be', 'decision_must', 'decision_claim_must', 'decision_i_pronoun', 'decision_i_application', 'decision_i_claim', 'decision_humanitarian', 'decision_i_humanitarian', 'decision_name', 'decision_tribunal', 'decision3', 'decision3b', 'decision3c', 'decision4', 'decision5', 'decision5b', 'decision5c', 'decision5d', 'decision6', 'decision7', 'decision7b', 'decision8', 'decision9', 'decision10', 'decision11', 'decision12', 'decision13', 'decision14', 'decision15', 'decision16', 'decision17', 'decision18', 'decision19', 'decision20', 'decision21', 'decision22', 'decision23', 'decision24', 'decision25', 'decision26', 'decision27', 'decision28', 'decision_i_find_app', 'decision_claimant_paragraph', 'decision_i_find_thus', 'decision_substantiated', 'decision_credible_and', 'decision_credible_therefore', 'decision_i_asylum_claim', 'decision_asylum_claim_status', 'decision_therefore_appeal', 'decision_find_that_grant', 'decision_grant_leave', 'decision_grant_leave_hearing', 'decision_find_appeal', 'decision_substantiated_grant', 'decision_credible_grant', 'decision_asterisk_description', 'decision_deny_asylum_appeal', 'decision_grant_appeal_grounds', 'decision_i_grant_asylum', 'decision_will_be', 'decision_will_be_article', 'decision_prompting', 'decision_format', 'decision_possessive_will', 'decision_pending', 'decision_viable', 'decision_i_deny_app', 'decision_leading_denial', 'decision_i_find_grant', 'decision_well_founded_grant', 'decision_i_find_appeal', 'decision_i_hereby', 'decision_has_been_granted', 'decision_leading_its', 'decision_faces_persecution_grant', 'decision_has_been', 'decision_has_been_allowed', 'decision_fears_grant', 'decision_asterisk_decision', 'decision_i_deny_asylum_appeal', 'decision_reasonable_grant', 'decision_stands', 'decision_tribunal_grants', 'decision_tribunal_grants_multiple', 'decision_we_deny', 'decision_i_grant_allowing', 'decision_which_hereby', 'decision_therefore_deny', 'decision_therefore_deny_general', 'decision_i_deny_their', 'decision_therefore_asterisk', 'decision_therefore_under', 'decision_deny_application', 'decision_asylum_claim_denied', 'decision_find_and_grant', 'decision_well_founded_fear', 'decision_find_appeal_granted', 'decision_hereby_asylum', 'decision_been_granted_asylum', 'decision_leading_its_denial', 'decision_therefore_grant_appeal', 'decision_general_been_granted', 'decision_been_allowed', 'decision_asterisk_colon', 'decision_deny_his_asylum', 'decision_reasonable_humanitarian', 'decision_claim_stands', 'decision_grant_appeal_asylum_grounds', 'decision_grant_leave_acknowledgment', 'decision_credibility_compromised', 'decision_tribunal_flexible', 'decision_refugee_support', 'decision_hereby_caps', 'decision_i_find_object', 'decision_decided', 'decision_plural', 'decision_remains', 'decision_after_end', 'decision_i_find_application', 'decision_making_this', 'decision_warranting', 'decision_grant_refugee_status', 'decision_grant_asylum_status', 'decision_grant_asylum_paragraph', 'decision_grant_asylum_article', 'decision_making_appeal', 'decision_warranting_status', 'decision_find_app_end', 'decision_falls_within', 'decision_breach_article', 'decision_breach_humanitarian', 'decision_accordance_hc395', 'decision_uk_refugee', 'decision_find_app_semi', 'decision_find_claim_semi', 'decision_asylum_end', 'decision_i_granted_end', 'decision_i_granted_convention', 'decision_find_name_appeal', 'decision_making_appeal_article', 'decision_semicolon_i_granted', 'decision_breach_i_granted', 'decision_humanitarian_grounds', 'decision_risk_outweighs', 'decision_real_risk_persecution', 'decision_well_founded_convention', 'decision_unduly_harsh', 'decision_potential_breach', 'decision_find_app_direct', 'decision_immigration_rules', 'decision_breach_simple', 'decision_multiple_articles', 'decision_refugee_convention', 'decision_warranting_end', 'decision_article_refugee', 'decision_hc395_period', 'decision_making_appeal_end', 'decision_persecution_article', 'decision_amounts_persecution', 'decision_engage_article', 'decision_convention_i_granted', 'decision_breach_echr_simple', 'decision_dual_echr', 'decision_rendering', 'decision_grant_international', 'decision_deny_international', 'decision_will_be_asylum', 'decision_granted_asylum_period', 'decision_granted_substantial', 'decision_substantial_grounds', 'decision_semicolon_granted', 'decision_unreasonable_granted', 'decision_no_support_granted', 'decision_unduly_harsh_granted', 'decision_harm_faith', 'decision_deny_article_1a2', 'decision_therefore_granted', 'decision_granted_asylum_1a2', 'decision_granted_asylum_1a2_alt', 'decision_therefore_granted_asylum', 'decision_therefore_granted_appeal', 'decision_asylum_denied_end', 'decision_reduces_risk_denied', 'decision_this_reduces_denied', 'decision_semicolon_asylum_denied', 'decision_dignity_granted', 'decision_dignity_granted_asylum', 'decision_outweighs_granted', 'decision_cumulative_granted', 'decision_given_cumulative', 'decision_therefore_appeal_1a2', 'decision_unreasonable_return', 'decision_grant_appeal_convention', 'decision_unfair_opportunity', 'decision_granted_asylum_status', 'decision_granting_international', 'decision_leading_appeal', 'decision_denial_asylum_claim', 'decision_i_have_denied', 'decision_allow_appeal_protection', 'decision_allow_granting_final']:
                    if group_name in match.groupdict() and match.group(group_name):
                        decision_word = match.group(group_name)
                        break
                
                if decision_word:
                    decision = normalize_decision(decision_word)
                    if decision:
                        all_decisions.append((decision, match.start()))
            except (IndexError, AttributeError):
                continue
    
    # Sort by position and return the first one
    if all_decisions:
        all_decisions.sort(key=lambda x: x[1])  # Sort by position
        return all_decisions[0][0]  # Return the decision (not position)
    
    return None

# --- Text Cleaning ---

def clean_response_text(response_text: str) -> str:
    """Clean response text by removing content after *END* and other artifacts"""
    if not response_text:
        return ""
    
    cleaned_text = response_text.strip()
    
    # Find and cut at *END* marker if present, but preserve a small window after for decisions
    end_marker_pos = cleaned_text.find('*END*')
    if end_marker_pos != -1:
        # Keep everything up to *END* plus a small window (50 chars) to catch decisions that appear right after
        window_end = end_marker_pos + 5 + 50  # +5 for "*END*" + 50 chars window
        cleaned_text = cleaned_text[:window_end]
        
        # Clean up the window after *END* - remove obvious junk but keep decision words
        end_part = cleaned_text[end_marker_pos + 5:]  # Text after *END*
        
        # Look for decision words in the immediate aftermath and keep them
        decision_match = re.search(r'["\s]*(?:denied|granted|dismissed|allowed|refused|approved)', end_part, re.IGNORECASE)
        if decision_match:
            # Keep up to the end of the decision word
            keep_until = end_marker_pos + 5 + decision_match.end()
            cleaned_text = cleaned_text[:keep_until]
        else:
            # No decision found, just keep the original *END*
            cleaned_text = cleaned_text[:end_marker_pos + 5]
    
    # Remove any trailing artifacts after cleaning
    trailing_artifacts = ['"""', "'''", '```', '---', '===']
    for artifact in trailing_artifacts:
        if cleaned_text.rstrip().endswith(artifact):
            cleaned_text = cleaned_text.rstrip()[:-len(artifact)].rstrip()
    
    # Remove leading artifacts
    leading_artifacts = ['"""', "'''", '```', 'Output:', 'Result:', 'Response:']
    for artifact in leading_artifacts:
        if cleaned_text.startswith(artifact):
            cleaned_text = cleaned_text[len(artifact):].strip()
    
    return cleaned_text.strip()

# --- Reasoning Extraction ---

def extract_reasoning(response_text: str) -> Optional[str]:
    """Extract legal reasoning from the response (appears before Decision:)"""
    if not response_text:
        return None
    
    # Clean the text first
    text = response_text.strip()
    
    # Find the Decision: marker to know where reasoning ends
    decision_match = re.search(r'Decision:\s*\w+', text, re.IGNORECASE)
    if decision_match:
        # Extract everything before "Decision:"
        reasoning_text = text[:decision_match.start()].strip()
    else:
        # If no Decision: marker found, take first part of text
        reasoning_text = text
    
    # Remove explicit "Reasoning:" label if present
    reasoning_text = re.sub(r'^Reasoning:\s*', '', reasoning_text, flags=re.IGNORECASE).strip()
    
    # Clean up the reasoning text
    if reasoning_text:
        # Remove excessive whitespace
        reasoning_text = re.sub(r'\s+', ' ', reasoning_text)
        
        # Must be substantial (more than just a few words)
        if len(reasoning_text.split()) > 5:
            return reasoning_text
    
    return None

# --- Main Processing ---

def analyze_response(response_text: str, case_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single response to extract decision and reasoning"""
    # Clean the response first
    cleaned_text = clean_response_text(response_text)
    
    # Extract decision and reasoning from cleaned text
    decision = extract_primary_decision(cleaned_text)
    reasoning = extract_reasoning(cleaned_text)
    
    # Quality metrics specific to this format
    has_end_marker = '*END*' in response_text
    has_decision_format = bool(re.search(r'(?:Decision|DECISION|Decided):\s*\w+', response_text, re.IGNORECASE)) or \
                         bool(re.search(r'(?:Decision|DECISION|Decided):\s*(?:ASYLUM\s+|Appeal\s+)?(?:granted|denied|dismissed|allowed|refused|approved|partially\s+granted)', response_text, re.IGNORECASE)) or \
                         bool(re.search(r'\*(?:granted|denied|dismissed|allowed|refused|approved|partially\s+granted|adjourned)\*', response_text, re.IGNORECASE)) or \
                         bool(re.search(r'I\s+\*(?:granted|denied|dismissed|allowed|refused|approved|grant|deny)\*', response_text, re.IGNORECASE)) or \
                         bool(re.search(r'(?:appeal|claim|application)\s+is\s+(?:accordingly\s+)?(?:granted|denied|dismissed|allowed|refused|approved)', response_text, re.IGNORECASE)) or \
                         bool(re.search(r'application\s+for\s+asylum\s+(?:in\s+the\s+UK\s+)?is\s+(?:therefore\s+)?(?:granted|denied|dismissed|allowed|refused|approved)', response_text, re.IGNORECASE)) or \
                         bool(re.search(r'claim\s+for\s+asylum\s+is\s+(?:accordingly\s+)?(?:granted|denied)', response_text, re.IGNORECASE)) or \
                         bool(re.search(r'case\s+(?:dismissed|granted|denied|allowed|refused|approved)', response_text, re.IGNORECASE)) or \
                         bool(re.search(r'I\s+(?:allow|deny|grant)\s+(?:\w+\'s\s+)?(?:appeal|it|her\s+appeal|his\s+appeal|the\s+appeal|this\s+appeal)', response_text, re.IGNORECASE)) or \
                         bool(re.search(r'I\s+(?:grant|deny|allow)(?:\s+(?:the|this)\s+appeal)?(?:\s+on\s+[\w\s]+grounds)?', response_text, re.IGNORECASE)) or \
                         bool(re.search(r'(?:appeal|claim|application)?\s*should\s+be\s+(?:granted|denied|dismissed|allowed|refused|approved)', response_text, re.IGNORECASE))
    text_after_end = len(response_text) > len(cleaned_text) if cleaned_text else False
    
    # Extract or preserve sample_id for consistent matching
    sample_id = case_metadata.get('sample_id')
    if sample_id is None:
        # Generate sample_id from consistent fields if missing (for older inference files)
        sample_id = generate_sample_id_from_metadata(case_metadata)
    
    return {
        'sample_id': sample_id,  # Preserve or generate sample_id for matching
        'decision': decision,
        'reasoning': reasoning,
        'original_response': response_text,
        'cleaned_response': cleaned_text,
        'has_end_marker': has_end_marker,
        'has_decision_format': has_decision_format,
        'text_after_end': text_after_end,
        'metadata': case_metadata,
        'quality_metrics': {
            'response_length': len(cleaned_text.split()) if cleaned_text else 0,
            'original_length': len(response_text.split()) if response_text else 0,
            'has_decision': decision is not None,
            'has_reasoning': reasoning is not None and len(reasoning.split()) > 5,
            'follows_format': has_end_marker and has_decision_format,
            'text_reduction_ratio': len(cleaned_text) / len(response_text) if response_text else 1.0
        }
    }

def generate_sample_id_from_metadata(metadata: Dict[str, Any]) -> str:
    """
    Generate a consistent sample_id from metadata for older inference files.
    This creates a deterministic ID that can be used for matching across datasets.
    """
    # Extract key fields for ID generation
    topic = metadata.get('topic', '')
    meta_topic = metadata.get('meta_topic', '')
    fields = metadata.get('fields', {})
    
    # Build a deterministic key from consistent fields
    key_parts = [
        meta_topic,
        topic,
        str(fields.get('country', '')),
        str(fields.get('age', '')),
        str(fields.get('gender', '')),
        str(fields.get('religion', '')),
        str(fields.get('name', '')),
    ]
    
    # Add any ordinal fields for additional uniqueness
    for key, value in metadata.items():
        if key.startswith('fields.') and key.endswith('__ordinal'):
            key_parts.append(f"{key}:{value}")
    
    # Create hash-based ID for deterministic but compact representation
    import hashlib
    key_string = '|'.join(key_parts)
    sample_hash = hashlib.md5(key_string.encode()).hexdigest()[:8]
    
    # Return a readable sample_id
    return f"meta_{sample_hash}"

def calculate_summary_statistics(processed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate summary statistics for the processed results with enhanced decision categorization"""
    total = len(processed_results)
    granted = sum(1 for r in processed_results if r['decision'] == 'Granted')
    denied = sum(1 for r in processed_results if r['decision'] == 'Denied')
    inconclusive = sum(1 for r in processed_results if r['decision'] == 'Inconclusive')
    no_decision = sum(1 for r in processed_results if r['decision'] is None)
    
    has_reasoning = sum(1 for r in processed_results if r['quality_metrics']['has_reasoning'])
    follows_format = sum(1 for r in processed_results if r['quality_metrics']['follows_format'])
    has_end_marker = sum(1 for r in processed_results if r['has_end_marker'])
    has_decision_format = sum(1 for r in processed_results if r['has_decision_format'])
    
    avg_length = sum(r['quality_metrics']['response_length'] for r in processed_results) / total if total > 0 else 0
    
    return {
        'total_responses': total,
        'decision_stats': {
            'granted': granted,
            'denied': denied,
            'inconclusive': inconclusive,
            'no_decision': no_decision,
            'granted_rate': granted / total if total > 0 else 0,
            'denied_rate': denied / total if total > 0 else 0,
            'inconclusive_rate': inconclusive / total if total > 0 else 0,
            'decided_rate': (granted + denied + inconclusive) / total if total > 0 else 0,
        },
        'quality_metrics': {
            'avg_response_length': avg_length,
            'has_decision_rate': (total - no_decision) / total if total > 0 else 0,
            'has_reasoning_rate': has_reasoning / total if total > 0 else 0,
            'follows_format_rate': follows_format / total if total > 0 else 0,
            'has_end_marker_rate': has_end_marker / total if total > 0 else 0,
            'has_decision_format_rate': has_decision_format / total if total > 0 else 0,
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Process inference results to extract decisions and reasoning')
    parser.add_argument('input_file', help='Path to the input JSON file')
    parser.add_argument('--max-records', type=int, help='Maximum number of records to process (for testing)')
    parser.add_argument('--verbose', action='store_true', help='Print progress information')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        sys.exit(1)
    
    # Load input data
    with open(input_path, 'r') as f:
        input_data = json.load(f)
    
    if isinstance(input_data, list):
        items = input_data
    elif isinstance(input_data, dict) and 'results' in input_data:
        items = input_data['results']
    else:
        print("Error: Unexpected input format")
        sys.exit(1)
    
    # Limit records if specified
    if args.max_records:
        items = items[:args.max_records]
    
    print(f"Processing {input_path}...")
    
    # Process each item
    processed_results = []
    for i, item in enumerate(items):
        if args.verbose and (i + 1) % 100 == 0:
            print(f"Processing item {i + 1}/{len(items)}...")
        
        response_text = item.get('model_response', '')
        metadata = {k: v for k, v in item.items() if k != 'model_response'}
        
        result = analyze_response(response_text, metadata)
        processed_results.append(result)
    
    # Calculate summary statistics
    summary_stats = calculate_summary_statistics(processed_results)
    
    # Split results into successful and failed decision extractions
    successful_extractions = [r for r in processed_results if r['decision'] is not None]
    failed_extractions = [r for r in processed_results if r['decision'] is None]
    
    # Create output directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"processed_{input_path.stem}_{timestamp}"
    run_output_dir = input_path.parent / 'processed' / run_dir_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare metadata for both files
    base_metadata = {
        'input_file': str(input_path),
        'processing_timestamp': datetime.now().isoformat(),
        'total_items_processed': len(processed_results),
        'run_directory': str(run_output_dir)
    }
    
    # Save successful extractions
    successful_data = {
        'metadata': {
            **base_metadata,
            'file_type': 'successful_extractions',
            'count': len(successful_extractions)
        },
        'summary_statistics': summary_stats,
        'processed_results': successful_extractions
    }
    
    successful_path = run_output_dir / 'successful_extractions.json'
    with open(successful_path, 'w') as f:
        json.dump(successful_data, f, indent=2)
    
    # Save failed extractions for reiteration
    failed_data = {
        'metadata': {
            **base_metadata,
            'file_type': 'failed_extractions',
            'count': len(failed_extractions),
            'note': 'Cases where decision extraction failed - suitable for reiteration'
        },
        'processed_results': failed_extractions
    }
    
    failed_path = run_output_dir / 'failed_extractions.json'
    with open(failed_path, 'w') as f:
        json.dump(failed_data, f, indent=2)
    
    # Save overall summary
    summary_data = {
        'metadata': base_metadata,
        'summary_statistics': summary_stats,
        'file_breakdown': {
            'successful_extractions': {
                'file': 'successful_extractions.json',
                'count': len(successful_extractions),
                'description': 'Cases with successfully extracted decisions - ready for analysis'
            },
            'failed_extractions': {
                'file': 'failed_extractions.json', 
                'count': len(failed_extractions),
                'description': 'Cases where decision extraction failed - suitable for reiteration'
            }
        }
    }
    
    summary_path = run_output_dir / 'processing_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print("Processing complete!")
    print(f"Results saved to run directory: {run_output_dir}")
    print(f"Files created:")
    print(f"  - successful_extractions.json: {len(successful_extractions)} cases ready for analysis")
    print(f"  - failed_extractions.json: {len(failed_extractions)} cases for reiteration")
    print(f"  - processing_summary.json: Overall run summary")
    print(f"Total items processed: {len(processed_results)}")
    print("Summary statistics:")
    print(f"  - Granted: {summary_stats['decision_stats']['granted']} ({summary_stats['decision_stats']['granted_rate']:.1%})")
    print(f"  - Denied: {summary_stats['decision_stats']['denied']} ({summary_stats['decision_stats']['denied_rate']:.1%})")
    print(f"  - No decision: {summary_stats['decision_stats']['no_decision']}")
    print(f"  - Has reasoning: {summary_stats['quality_metrics']['has_reasoning_rate']:.1%}")
    print(f"  - Follows format: {summary_stats['quality_metrics']['follows_format_rate']:.1%}")
    print(f"  - Has *END* marker: {summary_stats['quality_metrics']['has_end_marker_rate']:.1%}")
    print(f"  - Has Decision: format: {summary_stats['quality_metrics']['has_decision_format_rate']:.1%}")

if __name__ == "__main__":
    main() 