# Comparative Fairness Analysis

This project conducts a comprehensive comparative fairness analysis between two fine-tuned legal LLMs (Pre-Brexit and Post-Brexit) across asylum decision vignettes.

## Project Structure

```
comparative_fairness/
├── 01_data_preprocessing/      # Data loading and initial processing
├── 02_group_analysis/          # Group identification and filtering
├── 03_fairness_metrics/        # Fairness metric calculations
├── 04_significance_testing/    # Statistical significance testing
├── 05_visualization/           # Plots and visualizations
├── 06_reporting/              # Final reports and summaries
├── data/                      # Data storage
├── outputs/                   # Results and logs
├── config/                    # Configuration files
├── utils/                     # Utility functions
└── tests/                     # Unit tests
```

## Analysis Overview

### Protected Attributes
- **Gender**: Male, Female, Non-binary
- **Religion**: Various religious affiliations
- **Age**: Discrete age values
- **Country**: Country of origin

### Intersectional Analysis
- Gender × Religion
- Gender × Country  
- Religion × Country
- Age × Gender
- Religion × Country × Gender

### Fairness Metrics

**Representation Fairness:**
- Statistical Parity (SP)
- Counterfactual Flip Rate (CF)

**Error-Based Fairness:**
- Equal Opportunity (EO)
- False Positive Rate (FPR)
- Equalized Odds L2 (EOL2)
- Agreement Rate (AGR)

## Usage

### Phase 1: Data Preprocessing
```bash
cd /data/shil6369/gov_scraper/vignettes_analysis/comparative_fairness
python 01_data_preprocessing/load_and_tag_records.py
```

### Configuration
Edit `config/analysis_config.yaml` to modify:
- Minimum group size (default: 30)
- Significance threshold (default: 1.96)
- Protected attributes
- Data paths

## Data Sources
- Post-Brexit Model: `full_inference_llama3_8b_post_brexit_2019_2025_instruct_*`
- Pre-Brexit Model: `full_inference_llama3_8b_pre_brexit_2013_2016_instruct_*`

## Output Files
- `data/processed/tagged_records.json`: All records with fairness tags
- `data/processed/data_quality_report.json`: Data quality assessment
- `data/processed/data_distribution_summary.json`: Distribution analysis
- `outputs/logs/`: Execution logs

## Next Steps
1. Run data preprocessing to validate setup
2. Implement group analysis and filtering
3. Calculate fairness metrics
4. Perform statistical significance testing
5. Generate visualizations and reports 