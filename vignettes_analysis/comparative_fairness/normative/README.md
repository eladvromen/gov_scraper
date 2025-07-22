# Normative Analysis Module

## Purpose
This module contains all normative analysis for comparing decision-making patterns between Pre-Brexit and Post-Brexit models. It focuses on grant rate analysis, topic tendencies, and normative divergence across vignette fields and topics.

## Structure

### main_calculation/
Core normative metrics computation - **RUN THESE FIRST**
- `grant_rate_analysis_by_vignette_fields_enhanced.py` - Main grant rate analysis by vignette fields
- `grant_rate_analysis_by_vignette_fields.py` - Original version (for reference)
- `normative_divergence_analysis.py` - Calculates divergence between model behaviors
- `combined_decision_distribution.py` - Combined decision distribution analysis

### analysis/
Specialized normative analysis approaches - **RUN AFTER main_calculation**
- `rq1/` - Research Question 1 focused analysis and visualizations
- `decision_patterns/` - Decision pattern analysis across topics
- `divergence/` - (Reserved for future normative divergence analysis)

### outputs/
Organized normative analysis results
- `grant_rates/` - Grant rate calculations and field-level analysis
- `topic_tendencies/` - Topic-level tendency analysis
- `rq1_results/` - Research Question 1 specific results and visualizations
- `divergence/` - Normative divergence analysis results
- `visualizations/` - Generated plots and visualizations

## Execution Order
1. **Data Prerequisites**: 
   - Ensure `../data/processed/tagged_records.json` exists
   - Ensure `/data/shil6369/vignettes/complete_vignettes.json` exists
2. **Main Calculations**: Run scripts in `main_calculation/` directory
3. **Specialized Analysis**: Run scripts in `analysis/` subdirectories
4. **Results**: Check `outputs/` for generated analysis results

## Key Outputs
- **Grant Rates**: `outputs/grant_rates/grant_rate_analysis_by_vignette_fields_enhanced.csv`
- **Topic Tendencies**: `outputs/grant_rates/topic_tendencies_analysis_enhanced.csv`
- **RQ1 Analysis**: `outputs/rq1_results/` (multiple visualization files)
- **Divergence Metrics**: `outputs/divergence/normative_divergence_results.json`

## Analysis Focus
- **78D Field Vector**: Detailed field-level grant rate comparisons
- **13D Topic Vector**: Topic-level normative tendency analysis  
- **Cosine Similarity**: Measuring behavioral divergence between models
- **Statistical Significance**: Identifying meaningful differences in decision patterns 