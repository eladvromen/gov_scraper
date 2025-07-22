# Comparative Fairness Analysis - Refactored Structure

## 🎯 Overview
This refactored codebase provides a clean, organized structure for comparative fairness analysis between Pre-Brexit and Post-Brexit legal LLMs. The structure clearly separates fairness and normative analyses with distinct calculation and analysis phases.

## 📁 New Directory Structure

```
comparative_fairness/
├── 📂 fairness/                    # FAIRNESS ANALYSIS MODULE
│   ├── main_calculation/           # Core fairness metrics (RUN FIRST)
│   ├── analysis/                   # Specialized fairness analysis
│   │   ├── age/                    # Age-specific bias analysis
│   │   ├── gender/                 # Gender-specific bias analysis
│   │   ├── vector_drift/           # Bias vector drift analysis
│   │   ├── divergence/             # Fairness divergence analysis
│   │   └── visualizations/         # Visualization generation
│   └── outputs/                    # Organized fairness results
│       ├── metrics/                # Raw calculations
│       ├── unified/                # Unified dataframes
│       ├── age_analysis/           # Age analysis results
│       ├── gender_analysis/        # Gender analysis results
│       ├── vector_drift/           # Vector drift results
│       ├── divergence/             # Divergence results
│       ├── visualizations/         # Generated plots
│       └── statistical_validity/   # Statistical assessments
│
├── 📂 normative/                   # NORMATIVE ANALYSIS MODULE
│   ├── main_calculation/           # Core normative metrics (RUN FIRST)
│   ├── analysis/                   # Specialized normative analysis
│   │   ├── rq1/                    # Research Question 1 analysis
│   │   ├── decision_patterns/      # Decision pattern analysis
│   │   └── divergence/             # Normative divergence analysis
│   └── outputs/                    # Organized normative results
│       ├── grant_rates/            # Grant rate calculations
│       ├── topic_tendencies/       # Topic tendency analysis
│       ├── rq1_results/            # RQ1 specific results
│       ├── divergence/             # Divergence results
│       └── visualizations/         # Generated plots
│
├── 📂 dashboard/                   # INTERACTIVE DASHBOARD
│   ├── ethical_ai_assessment_dashboard.py
│   ├── launch_dashboard.py
│   └── demo_*.py                   # Demo scripts
│
├── 📂 data/                        # SHARED DATA (unchanged)
├── 📂 config/                      # CONFIGURATION (unchanged)
├── 📂 utils/                       # SHARED UTILITIES (unchanged)
├── 📂 01_data_preprocessing/       # DATA PREPROCESSING (unchanged)
└── 📂 outputs/                     # SHARED OUTPUTS (logs only)
```

## 🚀 Execution Workflow

### Phase 1: Data Preparation
```bash
# Ensure external data dependencies are available
# - Pre/Post Brexit model results
# - Vignette structure data
```

### Phase 2: Fairness Analysis
```bash
cd fairness/

# 1. Run main calculations (REQUIRED FIRST)
python main_calculation/statistical_parity_all_pairs.py
python main_calculation/error_based_metrics_all_pairs.py
python main_calculation/create_unified_fairness_dataframe.py
python main_calculation/comprehensive_stats_analysis.py
python main_calculation/statistical_validity.py

# 2. Run specialized analysis (OPTIONAL)
python analysis/age/age_bias_analysis.py
python analysis/gender/gender_bias_analysis.py
python analysis/vector_drift/bias_vector_drift_analysis.py
python analysis/divergence/fairness_divergence_analysis.py
python analysis/visualizations/standalone_bias_visualizations.py
```

### Phase 3: Normative Analysis
```bash
cd normative/

# 1. Run main calculations (REQUIRED FIRST)
python main_calculation/grant_rate_analysis_by_vignette_fields_enhanced.py
python main_calculation/normative_divergence_analysis.py
python main_calculation/combined_decision_distribution.py

# 2. Run specialized analysis (OPTIONAL)
python analysis/rq1/rq1_focused_analysis.py
python analysis/rq1/rq1_visualizations_separate.py
python analysis/decision_patterns/decision_distribution_analysis.py
```

### Phase 4: Interactive Dashboard
```bash
cd dashboard/
python launch_dashboard.py
# Opens at http://localhost:8501
```

## 📊 Key Benefits

### ✅ Clear Separation
- **Fairness vs Normative**: Distinct modules for different analysis types
- **Calculation vs Analysis**: Clear distinction between core metrics and specialized analysis
- **Organized Outputs**: Results clearly categorized by analysis type

### ✅ Execution Clarity
- **Dependencies**: Clear prerequisite chains
- **Order**: Obvious execution order (main_calculation → analysis)
- **Modularity**: Can run individual components independently

### ✅ Results Organization
- **Findable**: Know exactly where each type of result is stored
- **Structured**: Logical hierarchy for all outputs
- **Documented**: Each module has clear documentation

## 🔍 Quick Reference

### External Dependencies
- **Model Results**: `/data/shil6369/gov_scraper/inference/results/processed/`
- **Vignettes**: `/data/shil6369/vignettes/complete_vignettes.json`

### Key Outputs
- **Fairness Metrics**: `fairness/outputs/metrics/`
- **Unified Fairness Data**: `fairness/outputs/unified/`
- **Grant Rate Analysis**: `normative/outputs/grant_rates/`
- **RQ1 Results**: `normative/outputs/rq1_results/`

### Quick Launch Commands
```bash
# Fairness main calculations
cd fairness/main_calculation && python statistical_parity_all_pairs.py

# Normative main calculations  
cd normative/main_calculation && python grant_rate_analysis_by_vignette_fields_enhanced.py

# Interactive dashboard
cd dashboard && python launch_dashboard.py
```

---

## 📋 Migration Summary
- **Files Moved**: All analysis scripts organized by function
- **Outputs Reorganized**: Results categorized by analysis type
- **Structure Simplified**: Clear main_calculation vs analysis distinction
- **Documentation Added**: Each module documented with execution order 