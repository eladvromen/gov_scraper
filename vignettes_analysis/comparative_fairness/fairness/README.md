# Fairness Analysis Module

## Purpose
This module contains all fairness-related analyses for the comparative study between Pre-Brexit and Post-Brexit models. It focuses on statistical parity, equal opportunity, and bias pattern analysis across protected attributes (age, gender, religion, country).

## Structure

### main_calculation/
Core fairness metrics computation - **RUN THESE FIRST**
- `statistical_parity_all_pairs.py` - Statistical parity calculations for all group pairs
- `error_based_metrics_all_pairs.py` - Equal opportunity and other error-based metrics
- `create_unified_fairness_dataframe.py` - Combines all metrics into unified format
- `comprehensive_stats_analysis.py` - Statistical analysis of results
- `statistical_validity.py` - Statistical significance testing with corrections

### analysis/
Specialized fairness analysis approaches - **RUN AFTER main_calculation**
- `age/` - Age-specific bias pattern analysis
- `gender/` - Gender-specific bias pattern analysis  
- `vector_drift/` - Bias vector drift and qualitative analysis
- `divergence/` - Fairness divergence analysis using SP vectors
- `visualizations/` - Standalone visualization generation

### outputs/
Organized fairness analysis results
- `metrics/` - Raw metric calculations (JSON format)
- `unified/` - Unified dataframes and analysis vectors
- `age_analysis/` - Age-specific analysis results
- `gender_analysis/` - Gender-specific analysis results
- `vector_drift/` - Vector drift analysis results
- `divergence/` - Divergence analysis results
- `visualizations/` - Generated plots and visualizations
- `statistical_validity/` - Statistical validity assessments

## Execution Order
1. **Data Prerequisites**: Ensure `../data/processed/tagged_records.json` exists
2. **Main Calculations**: Run scripts in `main_calculation/` directory
3. **Specialized Analysis**: Run scripts in `analysis/` subdirectories
4. **Results**: Check `outputs/` for generated analysis results

## Key Outputs
- **Core Metrics**: `outputs/metrics/statistical_parity_all_pairs_results.json`
- **Unified Data**: `outputs/unified/unified_fairness_dataframe_topic_granular.csv`
- **Analysis Results**: Individual subdirectories in `outputs/` 