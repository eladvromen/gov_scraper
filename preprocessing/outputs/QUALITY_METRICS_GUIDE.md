# LLaMA Preprocessing Quality Metrics & Assessment Guide

## Overview

The LLaMA preprocessing pipeline includes comprehensive quality assessment and monitoring to ensure that processed legal text data meets the highest standards for language model training. This guide explains the quality metrics, thresholds, and interpretation.

## 🎯 Quality Assessment Components

### 1. **Token Distribution Analysis**
- **Target Range**: 512-2048 tokens per chunk (optimal: 1024)
- **Metrics Tracked**:
  - Mean, median, standard deviation
  - Percentiles (25th, 75th)
  - Percentage of chunks within target range
- **Quality Threshold**: ≥80% of chunks should be within target range

### 2. **Legal Domain Specificity**
- **Purpose**: Ensure text maintains legal domain characteristics
- **Metrics**:
  - Legal term density (% of domain-specific vocabulary)
  - Coverage of legal terminology across chunks
- **Key Terms**: tribunal, appellant, asylum, immigration, persecution, etc.
- **Quality Threshold**: ≥80% of chunks should contain legal terms

### 3. **Artifact Detection**
- **Purpose**: Identify preprocessing artifacts that could harm training
- **Detected Artifacts**:
  - Page numbers and isolated numbers
  - Judicial metadata (hearing dates, judge listings)
  - Copyright notices
  - Excessive whitespace
  - Repeated characters
- **Quality Threshold**: <10% of chunks should contain artifacts

### 4. **Semantic Coherence**
- **Purpose**: Ensure logical flow and readability
- **Metrics**:
  - Paragraph transition quality
  - Legal citation coherence
  - Sentence structure analysis
- **Assessment**: Transition words, topic continuity, citation patterns

### 5. **Temporal Consistency**
- **Purpose**: Ensure consistent processing across time periods
- **Metrics**:
  - Cross-dataset coefficient of variation
  - Dataset balance ratios
  - Processing consistency metrics

## 📊 Overall Quality Score

The overall quality score (0-1.0) is calculated using weighted metrics:

- **Token Distribution** (40%): Compliance with target token ranges
- **Legal Domain Coverage** (25%): Domain-specific term presence
- **Artifact Cleanliness** (20%): Absence of preprocessing artifacts
- **Semantic Coherence** (15%): Text flow and logical structure

### Quality Thresholds:
- **✅ PASS** (≥0.70): Ready for training
- **⚠️ REVIEW** (0.50-0.69): Needs attention before training
- **❌ FAIL** (<0.50): Significant issues require addressing

## 🛠️ How to Run Quality Assessment

### Option 1: Integrated with Preprocessing
```bash
# Quality assessment runs automatically
python preprocessing/scripts/main/llama_preprocessing_pipeline.py

# Skip quality assessment if needed
python preprocessing/scripts/main/llama_preprocessing_pipeline.py --skip_quality_assessment
```

### Option 2: Standalone Quality Check
```bash
# Run quality assessment on existing processed data
python preprocessing/scripts/utilities/run_quality_check.py

# Specify custom directories
python preprocessing/scripts/utilities/run_quality_check.py \
    --data_dir preprocessing/outputs/llama_training_ready \
    --output_dir preprocessing/outputs/quality_assessment
```

### Option 3: Detailed Quality Analysis
```bash
# Run comprehensive quality analysis
python preprocessing/scripts/analysis/preprocessing_quality_monitor.py \
    --processed_data_dir preprocessing/outputs/llama_training_ready \
    --output_dir preprocessing/outputs/quality_assessment
```

## 📈 Quality Report Outputs

### 1. **Quality Assessment Report** (`quality_assessment_report.json`)
Comprehensive JSON report containing:
- Overall quality score
- Per-dataset metrics
- Temporal consistency analysis
- Detailed recommendations

### 2. **Quality Dashboard** (`quality_assessment_dashboard.png`)
Visual dashboard showing:
- Token distribution across datasets
- Quality scores by dataset
- Chunk distribution pie chart
- Overall quality summary

### 3. **Processing Statistics**
- Total chunks processed
- Token count distributions
- Legal domain coverage
- Artifact detection results

## 🔍 Interpreting Quality Metrics

### Token Distribution
```json
{
  "mean": 1024.5,
  "median": 1018.0,
  "within_target": 0.85,
  "std": 245.2
}
```
- **Good**: Mean ~1024, within_target >0.8, low std deviation
- **Concerning**: High std deviation, low within_target percentage

### Legal Domain Coverage
```json
{
  "avg_legal_term_density": 0.05,
  "chunks_with_legal_terms": 0.92
}
```
- **Good**: High term density, >90% coverage
- **Concerning**: Low density suggests poor domain specificity

### Artifact Metrics
```json
{
  "avg_artifacts_per_chunk": 0.2,
  "chunks_with_artifacts": 0.08
}
```
- **Good**: Low artifact counts, <10% affected chunks
- **Concerning**: High artifact presence indicates cleaning issues

## ⚠️ Common Quality Issues & Solutions

### Issue: Low Token Range Compliance
**Symptoms**: `within_target < 0.8`
**Solutions**:
- Adjust chunking parameters (`min_chunk_tokens`, `max_chunk_tokens`)
- Review semantic chunking logic
- Check for extremely short/long documents

### Issue: Poor Legal Domain Coverage
**Symptoms**: `chunks_with_legal_terms < 0.8`
**Solutions**:
- Verify source data quality
- Expand legal term vocabulary
- Check text cleaning isn't removing legal terminology

### Issue: High Artifact Count
**Symptoms**: `chunks_with_artifacts > 0.1`
**Solutions**:
- Enhance cleaning patterns in `advanced_legal_text_cleaning()`
- Add new artifact detection patterns
- Review header/footer removal logic

### Issue: Poor Semantic Coherence
**Symptoms**: Low transition scores, fragmented text
**Solutions**:
- Improve paragraph boundary detection
- Adjust chunking to respect document structure
- Review sentence segmentation logic

## 📋 Quality Recommendations System

The system automatically generates actionable recommendations based on quality metrics:

### Automatic Recommendations
- Chunking strategy adjustments
- Cleaning improvement suggestions
- Balance considerations for temporal datasets
- Training readiness assessment

### Manual Review Triggers
- Overall score <0.7
- Token compliance <80%
- Legal coverage <80%
- Artifact presence >10%

## 🔄 Continuous Monitoring

### Best Practices
1. **Run quality assessment after every preprocessing run**
2. **Monitor quality trends across different data batches**
3. **Set up automated alerts for quality degradation**
4. **Review recommendations before training**

### Quality Evolution Tracking
- Compare quality scores across preprocessing runs
- Track improvement after addressing recommendations
- Monitor consistency across temporal datasets

## 🎯 Training Readiness Checklist

Before using processed data for LLaMA training, ensure:

- [ ] Overall quality score ≥0.70
- [ ] Token range compliance ≥80%
- [ ] Legal domain coverage ≥80%
- [ ] Artifact presence <10%
- [ ] All quality recommendations addressed
- [ ] Temporal datasets properly balanced
- [ ] Visual quality dashboard reviewed

## 📞 Troubleshooting

### Quality Assessment Fails to Run
1. Check dependencies: `matplotlib`, `seaborn`, `numpy`, `pandas`
2. Verify processed data directory exists
3. Ensure JSONL files are properly formatted

### Low Quality Scores
1. Review the specific metrics causing issues
2. Check preprocessing parameters
3. Examine sample chunks manually
4. Adjust cleaning/chunking logic as needed

### Visualization Issues
1. Install required visualization libraries
2. Check file permissions for output directory
3. Verify matplotlib backend configuration

---

**For questions or issues with quality assessment, review the preprocessing logs and quality recommendations first. The quality system is designed to provide specific, actionable guidance for improving data quality.** 