# 📝 Vignette Completion Bias Analysis

**Focus Countries**: Syria, Nigeria, Myanmar
**Topic**: Intentions regarding work in the UK
**Analysis Date**: 2025-07-22 16:37:56

## 🎯 Executive Summary

- **High Bias Completions Found**: 5
- **Fields Analyzed**: 2
- **Countries Mapped**: 3

## 📊 Field-Level Bias Analysis

*Detailed breakdown of bias by vignette field*

### work intentions

- **Field Type**: ordinal
- **Number of Completions**: 3
- **Bias Range**: 0.3188

**Most Biased Completion**: plans to actively pursue career advancement and economic opportunities
- Pre-Brexit Rate: 0.747
- Post-Brexit Rate: 0.531
- Rate Change: +0.319
- Favors: pre_brexit

#### All Completions

| Completion | Pre-Rate | Post-Rate | Change | Significant | Favors |
|------------|----------|-----------|--------|-------------|--------|
| plans to actively pursue career advancement and ec... | 0.747 | 0.531 | +0.319 | ✓ | pre_brexit |
| hopes to find work to support {pronoun} self and f... | 0.995 | 0.943 | +0.132 | ✓ | pre_brexit |
| does not plan to seek work, focusing solely on saf... | 0.542 | 0.569 | +0.002 | ✗ | pre_brexit |

### profession

- **Field Type**: ordinal
- **Number of Completions**: 4
- **Bias Range**: 0.2672

**Most Biased Completion**: successful entrepreneur
- Pre-Brexit Rate: 0.836
- Post-Brexit Rate: 0.667
- Rate Change: +0.267
- Favors: pre_brexit

#### All Completions

| Completion | Pre-Rate | Post-Rate | Change | Significant | Favors |
|------------|----------|-----------|--------|-------------|--------|
| successful entrepreneur... | 0.836 | 0.667 | +0.267 | ✓ | pre_brexit |
| recent graduate... | 0.641 | 0.516 | +0.199 | ✓ | pre_brexit |
| trained skilled worker (e.g., electrician, mechani... | 0.796 | 0.725 | +0.143 | ✓ | pre_brexit |
| unskilled laborer with no formal education... | 0.771 | 0.817 | -0.005 | ✗ | post_brexit |

## 🔥 High Bias Completions

*Completions with statistically significant bias > 0.1*

| Field | Completion | Bias Magnitude | Direction | Sample Size |
|-------|------------|----------------|-----------|-------------|
| work intentions | plans to actively pursue career advancem... | 0.319 | Post-Brexit | 1152 |
| profession | successful entrepreneur... | 0.267 | Post-Brexit | 864 |
| profession | recent graduate... | 0.199 | Post-Brexit | 864 |
| profession | trained skilled worker (e.g., electricia... | 0.143 | Post-Brexit | 864 |
| work intentions | hopes to find work to support {pronoun} ... | 0.132 | Post-Brexit | 1152 |

## 🌍 Country-Specific Completion Effects

*How specific completions affect each focus country*

### Syria

- **Advantageous Completions**: 0
- **Disadvantageous Completions**: 5
- **Neutral Completions**: 0

#### Top Disadvantageous Completions

**work intentions**: plans to actively pursue career advancement and economic opportunities
- Rate Change: +0.319
- Relevance: indirect

**profession**: successful entrepreneur
- Rate Change: +0.267
- Relevance: indirect

**profession**: recent graduate
- Rate Change: +0.199
- Relevance: indirect

### Nigeria

- **Advantageous Completions**: 0
- **Disadvantageous Completions**: 5
- **Neutral Completions**: 0

#### Top Disadvantageous Completions

**work intentions**: plans to actively pursue career advancement and economic opportunities
- Rate Change: +0.319
- Relevance: direct

**profession**: successful entrepreneur
- Rate Change: +0.267
- Relevance: indirect

**profession**: recent graduate
- Rate Change: +0.199
- Relevance: indirect

### Myanmar

- **Advantageous Completions**: 0
- **Disadvantageous Completions**: 5
- **Neutral Completions**: 0

#### Top Disadvantageous Completions

**work intentions**: plans to actively pursue career advancement and economic opportunities
- Rate Change: +0.319
- Relevance: indirect

**profession**: successful entrepreneur
- Rate Change: +0.267
- Relevance: indirect

**profession**: recent graduate
- Rate Change: +0.199
- Relevance: indirect

## 🔗 Completion Interaction Analysis

### High Bias Completion Combinations

| Field | Completion | Bias | Direction | Sample Size |
|-------|------------|------|-----------|-------------|
| work intentions | plans to actively pursue career advancem... | 0.319 | favors_post | 1152 |
| profession | successful entrepreneur... | 0.267 | favors_post | 864 |
| profession | recent graduate... | 0.199 | favors_post | 864 |

## 🧠 Qualitative Analysis Framework

### Key Questions for Manual Completion Review

1. **Work Intentions Field**:
   - Which work intention completions favor/penalize specific countries?
   - How does 'actively pursue career advancement' vs 'focusing solely on safety' affect decisions?
   - Are certain countries assumed to have different work motivations?

2. **Profession Field**:
   - Do 'successful entrepreneur' vs 'unskilled laborer' completions show country bias?
   - Are certain countries' professional credentials treated differently?
   - How does 'recent graduate' completion interact with country of origin?

3. **Cross-Field Interactions**:
   - Do 'entrepreneur' + 'career advancement' combinations favor certain countries?
   - Are there problematic assumptions about country-profession relationships?
   - Which completion combinations drive the Myanmar/Syria/Nigeria pattern changes?

## 📋 Methodology

- **Vignette Structure**: 22 fields analyzed
- **Grant Rate Data**: Field-level completion analysis
- **Bias Threshold**: |Rate Change| > 0.1 for high bias
- **Significance**: p < 0.05 statistical significance
- **Country Mapping**: Heuristic-based completion-country relevance

