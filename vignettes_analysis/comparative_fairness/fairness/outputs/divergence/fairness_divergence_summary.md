====================================================================================================
COMPREHENSIVE FAIRNESS DIVERGENCE ANALYSIS SUMMARY
====================================================================================================

Dataset Summary:
  Total comparisons: 1414
  Protected attributes: 4
  Topics: 21

Attribute breakdown:
    country: 630 comparisons
    religion: 438 comparisons
    age: 252 comparisons
    gender: 94 comparisons

================================================================================
OVERALL SP MAGNITUDE DIVERGENCE
================================================================================
Cosine Similarity: 0.4261
Cosine Distance: 0.5739
Pearson Correlation: 0.4046
Mean difference: 0.0000
Standard deviation: 0.0945
Mean absolute difference: 0.0650

================================================================================
OVERALL SP SIGNIFICANCE DIVERGENCE
================================================================================
Pattern Agreement Rate: 65.63%
Pattern Change Rate: 34.37%

Significance Rates:
  Pre-Brexit: 31.40%
  Post-Brexit: 27.02%

Significance Transitions:
  Gained significance: 212
  Lost significance: 274
  Remained significant: 170
  Remained non-significant: 758

================================================================================
STRATIFIED ANALYSIS BY PROTECTED_ATTRIBUTE
================================================================================

Stratum              | Cosine Sim | Cosine Dist | Pattern Agr | Comparisons 
--------------------------------------------------------------------------------
country              | 0.2678     | 0.7322      | 63.17%      | 630         
age                  | 0.5341     | 0.4659      | 60.32%      | 252         
religion             | 0.3176     | 0.6824      | 71.23%      | 438         
gender               | 0.7126     | 0.2874      | 70.21%      | 94          

================================================================================
INTERPRETATION GUIDELINES
================================================================================

Magnitude Divergence (Cosine Similarity):
  1.0 = Identical bias patterns
  0.8-1.0 = High similarity (low divergence)
  0.6-0.8 = Moderate similarity
  0.0-0.6 = Low similarity (high divergence)
  <0.0 = Opposite bias patterns

Significance Divergence (Pattern Agreement Rate):
  90-100% = Very consistent significance patterns
  75-90% = Moderately consistent patterns
  50-75% = Somewhat inconsistent patterns
  <50% = Highly inconsistent patterns