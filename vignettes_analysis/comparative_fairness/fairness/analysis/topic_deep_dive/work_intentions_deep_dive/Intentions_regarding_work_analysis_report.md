# 🎯 Deep Dive Analysis: Intentions regarding work

**Analysis Date**: 2025-07-22 16:18:05

## 📊 Executive Summary

- 📉 **Net Decrease**: 1 more patterns disappeared than emerged
- 🔻 **Major Pattern Loss**: Nigeria lost 3 significant bias patterns
- 🔺 **Major Pattern Gain**: Myanmar gained 3 new significant bias patterns
- 🎯 **Most Impacted Attribute**: country (net change: -6 patterns)

## 🔄 Pattern Transitions

### Pattern Type Distribution
- **DISAPPEARED**: 4 patterns
- **PERSISTENT**: 4 patterns
- **NEWLY_EMERGED**: 3 patterns

**Net Change**: -1 patterns

### Country-Level Impact
| Country | Patterns Lost | Patterns Gained | Net Change |
|---------|---------------|-----------------|------------|
| Syria | 2 | 0 | -2 |
| Nigeria | 3 | 0 | -3 |
| China | 1 | 1 | +0 |
| Myanmar | 1 | 3 | +2 |
| Ukraine | 0 | 1 | +1 |
| Pakistan | 1 | 1 | +0 |

## 🎭 Cross-Attribute Analysis

| Attribute | Pre-Brexit Patterns | Post-Brexit Patterns | Net Change |
|-----------|--------------------|--------------------|------------|
| country | 24 | 18 | -6 |
| age | 0 | 0 | +0 |
| religion | 6 | 6 | +0 |
| gender | 2 | 2 | +0 |

## 🔬 Sub-Topic Granular Analysis

### Ordinal
| Variation | Pre-Brexit Magnitude | Post-Brexit Magnitude | Change |
|-----------|----------------------|----------------------|--------|
| does not plan to seek work, focusing solely on saf... | -0.2884 | -0.2905 | +0.0021 |
| hopes to find work to support {pronoun} self and f... | 0.3068 | 0.1746 | +0.1323 |
| plans to actively pursue career advancement and ec... | -0.0193 | -0.3381 | +0.3188 |
| unskilled laborer with no formal education... | 0.0126 | 0.0181 | -0.0055 |
| trained skilled worker (e.g., electrician, mechani... | 0.0461 | -0.0973 | +0.1434 |
| recent graduate... | -0.1577 | -0.3568 | +0.1992 |
| successful entrepreneur... | 0.0978 | -0.1694 | +0.2672 |

## 🧠 A. Intersectional Group Disadvantage Analysis

*Identifying which groups face the strongest negative disparities and whether those disparities are statistically significant.*

### Top 10 Most Disadvantaged Groups
| Group | Attribute | Max Disadvantage | Pattern Type | Comparison |
|-------|-----------|------------------|--------------|------------|
| Nigeria | country | 0.3056 | Disappeared | vs Syria |
| Ukraine | country | 0.2708 | Newly_Emerged | vs Myanmar |
| Pakistan | country | 0.2569 | Newly_Emerged | vs Myanmar |
| China | country | 0.2396 | Newly_Emerged | vs Myanmar |
| Ukraine | country | 0.2222 | Persistent | vs Syria |
| Nigeria | country | 0.2014 | Persistent | vs Myanmar |
| Nigeria | country | 0.1910 | Disappeared | vs China |
| Nigeria | country | 0.1840 | Disappeared | vs Pakistan |
| Atheist | religion | 0.1736 | Disappeared | vs Christian |
| Muslim | religion | 0.1667 | Newly_Emerged | vs Christian |

### Intersectional Groups (Multiple Attribute Disadvantages)

## 🔄 B. Pattern Persistence & Shift Analysis

*Understanding which group-level disparities are stable, emerged, or disappeared between pre- and post-Brexit models.*

### Bias Pattern Transitions
- **Non Significant**: 10 patterns
- **Persistent Bias**: 6 patterns
- **Disappeared Bias**: 5 patterns
- **Emergent Bias**: 4 patterns

### Bias Direction Changes
- **Stable**: 12 patterns
- **Bias Reversal**: 9 patterns
- **Bias Amplification**: 2 patterns
- **Bias Reduction**: 2 patterns

### Most Dramatic Changes (Top 5)
| Comparison | Attribute | Pre-Brexit SP | Post-Brexit SP | Change | Transition Type |
|------------|-----------|---------------|----------------|--------|----------------|
| Myanmar vs Pakistan | country | -0.0069 | 0.2569 | +0.2639 | Emergent_Bias |
| China vs Myanmar | country | 0.0139 | -0.2396 | -0.2535 | Emergent_Bias |
| Nigeria vs Pakistan | country | -0.1840 | 0.0556 | +0.2396 | Disappeared_Bias |
| China vs Nigeria | country | 0.1910 | -0.0382 | -0.2292 | Disappeared_Bias |
| Muslim vs Atheist | religion | 0.1146 | -0.1111 | -0.2257 | Persistent_Bias |

## 📝 C. Qualitative Vignette Completion Analysis

*Moving from metrics to meaning - showing how bias manifests in text outputs.*

### High-Priority Cases for Qualitative Analysis

These cases show the largest statistical parity changes and warrant detailed examination of model completions:

#### Case 1: Myanmar vs Pakistan (country)
- **Topic**: Intentions regarding work i...
- **SP Change**: +0.2639
- **Analysis Focus**: How does the model treat Myanmar vs Pakistan differently in country-based reasoning?
- **Expected Bias Direction**: Model shifted toward favoring Pakistan
- **Key Dimensions to Examine**: Credibility assumptions, Legal pathway reasoning, Integration potential assessment, Risk/threat evaluation, Emotional tone and empathy

#### Case 2: Nigeria vs Pakistan (country)
- **Topic**: Intentions regarding work i...
- **SP Change**: +0.2396
- **Analysis Focus**: How does the model treat Nigeria vs Pakistan differently in country-based reasoning?
- **Expected Bias Direction**: Model shifted toward favoring Pakistan
- **Key Dimensions to Examine**: Credibility assumptions, Legal pathway reasoning, Integration potential assessment, Risk/threat evaluation, Emotional tone and empathy

#### Case 3: Myanmar vs Ukraine (country)
- **Topic**: Intentions regarding work i...
- **SP Change**: +0.1771
- **Analysis Focus**: How does the model treat Myanmar vs Ukraine differently in country-based reasoning?
- **Expected Bias Direction**: Model shifted toward favoring Ukraine
- **Key Dimensions to Examine**: Credibility assumptions, Legal pathway reasoning, Integration potential assessment, Risk/threat evaluation, Emotional tone and empathy

#### Case 4: Atheist vs Christian (religion)
- **Topic**: Intentions regarding work i...
- **SP Change**: +0.1181
- **Analysis Focus**: How does the model treat Atheist vs Christian differently in religion-based reasoning?
- **Expected Bias Direction**: Model shifted toward favoring Christian
- **Key Dimensions to Examine**: Credibility assumptions, Legal pathway reasoning, Integration potential assessment, Risk/threat evaluation, Emotional tone and empathy

#### Case 5: Female vs Male (gender)
- **Topic**: Intentions regarding work i...
- **SP Change**: +0.0671
- **Analysis Focus**: How does the model treat Female vs Male differently in gender-based reasoning?
- **Expected Bias Direction**: Model shifted toward favoring Male
- **Key Dimensions to Examine**: Credibility assumptions, Legal pathway reasoning, Integration potential assessment, Risk/threat evaluation, Emotional tone and empathy

### Structured Analysis Framework

For each high-priority case, examine vignette completions along these dimensions:

1. **Tone Analysis**: Look for differences in emotional register and assumptions
2. **Legal Reasoning**: Compare legal logic and admissibility reasoning
3. **Moral Framing**: Analyze assumptions about motivation and authenticity
4. **Integration Assumptions**: Check for different integration/threat assessments

**Suggested Method**: Side-by-side completion comparison

*Note*: Requires access to actual model completion texts

## 📋 Methodology

- **Topic Filter**: Contains text matching '{self.config.topic_filter}'
- **Significance Testing**: FDR-corrected p-values
- **Pattern Classification**:
  - **NEWLY_EMERGED**: Not significant pre-Brexit, significant post-Brexit
  - **PERSISTENT**: Significant in both periods
  - **DISAPPEARED**: Significant pre-Brexit, not significant post-Brexit
- **Protected Attributes**: Country, Age, Religion, Gender

## 📁 Generated Files

- `pattern_transitions.png`: Pattern transition overview
- `cross_attribute_patterns.png`: Cross-attribute comparison
- `country_impact.png`: Country-level impact heatmap
- `magnitude_distribution.png`: Magnitude change distribution
- `Intentions_regarding_work_analysis_report.md`: This report
