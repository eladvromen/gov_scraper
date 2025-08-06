# 🎯 Topic Deep Dive Analyzer

A robust, reconfigurable tool for analyzing bias pattern changes across topics in government decision-making data.

## 🚀 Features

- **Configurable Topic Analysis**: Analyze any topic by passing a filter string
- **Multi-Dimensional Analysis**: Examines patterns across country, age, religion, and gender
- **Pattern Transition Tracking**: Identifies newly emerged, persistent, and disappeared bias patterns
- **Sub-Topic Granularity**: Analyzes specific variations within topics (e.g., different work intentions)
- **Statistical Rigor**: Uses FDR-corrected significance testing
- **Comprehensive Visualizations**: Generates publication-ready charts and heatmaps
- **Automated Reporting**: Creates detailed markdown reports with insights

## 📊 What It Analyzes

### Pattern Transitions
- **NEWLY_EMERGED**: Patterns that became significant post-Brexit
- **PERSISTENT**: Patterns significant in both periods  
- **DISAPPEARED**: Patterns that lost significance post-Brexit

### Protected Attributes
- **Country**: Bias patterns between different countries of origin
- **Age**: Age-based discrimination patterns
- **Religion**: Religious bias patterns
- **Gender**: Gender-based bias patterns

### Sub-Topic Granularity
- Specific intentions within topics (e.g., work intentions: safety-focused vs career-advancement)
- Professional categories and their bias patterns
- Educational aspirations and discrimination

## 🛠️ Installation & Setup

```bash
# Install required packages
pip install pandas numpy matplotlib seaborn pyyaml

# Navigate to the analysis directory
cd vignettes_analysis/comparative_fairness/fairness/outputs/topic_deep_dive
```

## 🔧 Usage Examples

### Basic Analysis
```bash
# Analyze work intentions
python topic_analyzer.py --topic "Intentions regarding work"

# Analyze education intentions  
python topic_analyzer.py --topic "Intentions regarding education"

# Analyze asylum circumstances
python topic_analyzer.py --topic "Asylum seeker circumstances"
```

### Advanced Usage
```bash
# Custom output directory
python topic_analyzer.py --topic "Nature of persecution" --output-dir custom_analysis

# Specify base data directory
python topic_analyzer.py --topic "Financial stability" --base-dir /path/to/data

# Include sub-topic analysis (default: enabled)
python topic_analyzer.py --topic "Intentions regarding work" --include-subtopics
```

### Configuration-Based Analysis
```python
from topic_analyzer import TopicDeepDiveAnalyzer, TopicAnalysisConfig
from pathlib import Path

# Setup configuration
config = TopicAnalysisConfig(
    topic_filter="Intentions regarding work",
    base_data_dir=Path("../.."),
    output_dir=Path("work_intentions_analysis"),
    include_subtopics=True,
    fdr_correction=True
)

# Run analysis
analyzer = TopicDeepDiveAnalyzer(config)
analyzer.run_full_analysis()
```

## 📈 Output Files

Each analysis generates:

1. **📊 Visualizations**
   - `pattern_transitions.png`: Overview of pattern changes
   - `cross_attribute_patterns.png`: Comparison across protected attributes
   - `country_impact.png`: Country-level impact heatmap
   - `magnitude_distribution.png`: Distribution of bias magnitude changes

2. **📄 Reports**
   - `[topic]_analysis_report.md`: Comprehensive analysis report with insights

3. **📁 Data Files** (optional)
   - Filtered datasets for further analysis

## 🎯 Example: Work Intentions Deep Dive

```bash
python topic_analyzer.py --topic "Intentions regarding work"
```

**Expected Output Structure:**
```
work_intentions_analysis/
├── pattern_transitions.png
├── cross_attribute_patterns.png  
├── country_impact.png
├── magnitude_distribution.png
└── intentions_regarding_work_analysis_report.md
```

**Key Insights Generated:**
- Net change in significant bias patterns (-1 for work intentions)
- Country-specific impacts (Nigeria lost 2 patterns, Myanmar gained 2)
- Cross-attribute comparison showing country vs age vs religion vs gender patterns
- Sub-topic analysis of different work intention types

## 🔍 Methodology

### Data Sources
- **Primary**: `deduplicated_fairness_comparisons.csv` - Core bias measurements
- **Transitions**: `detailed_pattern_transitions.csv` - Pattern change tracking
- **Significance**: `significant_country_comparisons_fdr.csv` - FDR-corrected results
- **Cross-Attribute**: `topic_attribute_analysis_data.csv` - Multi-dimensional analysis
- **Granular**: Grant rate analysis for sub-topic details

### Statistical Approach
- **Significance Testing**: FDR-corrected p-values (Benjamini-Hochberg)
- **Effect Size**: Bias magnitude measurements
- **Pattern Classification**: Pre/post significance comparison
- **Multi-Comparison Correction**: Accounts for multiple hypothesis testing

### Interpretive Framework
- **Newly Emerged Patterns**: May indicate new sources of bias post-Brexit
- **Disappeared Patterns**: Could suggest bias reduction or measurement changes
- **Persistent Patterns**: Ongoing discrimination requiring intervention
- **Magnitude Changes**: Quantify bias intensity shifts

## 🎛️ Customization

### Adding New Topics
Simply use any topic string that appears in the data:
```bash
python topic_analyzer.py --topic "your_topic_here"
```

### Custom Analysis Pipeline
```python
# Create custom analyzer
analyzer = TopicDeepDiveAnalyzer(config)

# Load and filter data
analyzer.load_data()
filtered_data = analyzer.filter_topic_data()

# Run specific analysis components
pattern_results = analyzer.analyze_pattern_transitions(filtered_data)
attribute_results = analyzer.analyze_cross_attribute_patterns(filtered_data)

# Generate custom insights
insights = analyzer.generate_insights({
    'pattern_transitions': pattern_results,
    'cross_attribute_patterns': attribute_results
})
```

### Extending Visualizations
Add new plotting methods to the `TopicDeepDiveAnalyzer` class:
```python
def _plot_custom_analysis(self, data):
    # Your custom visualization code
    plt.savefig(self.config.output_dir / 'custom_plot.png')
```

## 🔮 Future Enhancements

- **Timeline Analysis**: Track pattern evolution over multiple time periods
- **Predictive Modeling**: Forecast future bias pattern changes
- **Interactive Dashboards**: Web-based exploration interface
- **Comparative Analysis**: Side-by-side topic comparisons
- **Automated Monitoring**: Continuous bias pattern surveillance

## 🤝 Contributing

1. **Add New Analysis Modules**: Extend the analyzer with new analytical capabilities
2. **Improve Visualizations**: Create more informative or interactive plots
3. **Enhanced Reporting**: Add new insight generation algorithms
4. **Performance Optimization**: Improve data processing efficiency

## 📚 Academic Usage

This tool supports reproducible research in:
- **Algorithmic Fairness**: Quantifying bias in government decision-making
- **Policy Analysis**: Understanding discrimination pattern changes
- **Social Science**: Studying protected characteristic impacts
- **Machine Learning**: Bias detection and mitigation research

## 📞 Support

For issues or questions:
1. Check existing analysis outputs for similar topics
2. Verify data file paths and permissions
3. Review error messages for specific data requirements
4. Consider data preprocessing if custom datasets are used

---

**Citation**: When using this tool in research, please cite the original bias analysis framework and this specific implementation. 