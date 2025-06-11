# LLaMA Preprocessing File Locations Guide

## 📁 **Input Data Location**
- **Original Data**: `preprocessing/data/processed_data/processed_legal_cases_with_years.csv`
- **Expected Format**: CSV with columns including `decision_text_cleaned`, `extracted_year`, `reference_number`, etc.

## 📁 **Output Locations**

### **Preprocessed Training Data**
**Location**: `preprocessing/outputs/llama_training_ready/`

**Structure**:
```
preprocessing/outputs/llama_training_ready/
├── pre_brexit_2013_2016/
│   ├── pre_brexit_2013_2016_chunks.jsonl       # JSONL format for training frameworks
│   ├── pre_brexit_2013_2016_training.txt       # Plain text for direct training
│   ├── pre_brexit_2013_2016_metadata.csv       # Chunk metadata (case IDs, tokens, etc.)
│   └── pre_brexit_2013_2016_statistics.json    # Processing statistics
├── transitional_2017/
│   ├── transitional_2017_chunks.jsonl
│   ├── transitional_2017_training.txt
│   ├── transitional_2017_metadata.csv
│   └── transitional_2017_statistics.json
├── post_brexit_2018_2025/
│   ├── post_brexit_2018_2025_chunks.jsonl
│   ├── post_brexit_2018_2025_training.txt
│   ├── post_brexit_2018_2025_metadata.csv
│   └── post_brexit_2018_2025_statistics.json
├── human_inspection/                            # 👁️ NEW: Human inspection samples
│   ├── README.md                                # Instructions for human review
│   ├── inspection_summary.json                 # Overall cleaning statistics
│   ├── pre_brexit_2013_2016_inspection_samples.html    # Interactive HTML for review
│   ├── pre_brexit_2013_2016_inspection_samples.json    # Raw sample data
│   ├── pre_brexit_2013_2016_inspection_summary.csv     # Quick overview
│   ├── transitional_2017_inspection_samples.html
│   ├── transitional_2017_inspection_samples.json
│   ├── transitional_2017_inspection_summary.csv
│   ├── post_brexit_2018_2025_inspection_samples.html
│   ├── post_brexit_2018_2025_inspection_samples.json
│   └── post_brexit_2018_2025_inspection_summary.csv
└── pipeline_summary.json                       # Overall processing summary
```

### **Quality Assessment Reports**
**Location**: `preprocessing/outputs/quality_assessment/`

**Structure**:
```
preprocessing/outputs/quality_assessment/
├── quality_assessment_report.json              # Detailed quality metrics
├── quality_assessment_dashboard.png            # Visual quality dashboard
└── [additional quality analysis files]
```

## 🔧 **Script Locations**

### **Main Processing Scripts**
- **Main Pipeline**: `preprocessing/scripts/main/llama_preprocessing_pipeline.py`
- **Quality Monitor**: `preprocessing/scripts/analysis/preprocessing_quality_monitor.py`

### **Utility Scripts**
- **Quick Quality Check**: `preprocessing/scripts/utilities/run_quality_check.py`
- **Import Test**: `preprocessing/scripts/utilities/test_imports.py`

### **Analysis Scripts**
- **Brexit Analysis**: `preprocessing/scripts/analysis/brexit_temporal_analysis.py`
- **Quality Monitor**: `preprocessing/scripts/analysis/preprocessing_quality_monitor.py`

## 🚀 **How to Run**

### **Complete Pipeline**
```bash
# Run from project root directory
cd /data/shil6369/gov_scraper

# Activate conda environment
conda activate gov_scraper

# Run complete preprocessing with quality assessment and human inspection
python preprocessing/scripts/main/llama_preprocessing_pipeline.py \
    --data_path preprocessing/data/processed_data/processed_legal_cases_with_years.csv \
    --output_dir preprocessing/outputs/llama_training_ready \
    --human_inspection_samples 20

# Run with fewer inspection samples (faster)
python preprocessing/scripts/main/llama_preprocessing_pipeline.py \
    --data_path preprocessing/data/processed_data/processed_legal_cases_with_years.csv \
    --output_dir preprocessing/outputs/llama_training_ready \
    --human_inspection_samples 10

# Skip human inspection entirely (fastest)
python preprocessing/scripts/main/llama_preprocessing_pipeline.py \
    --data_path preprocessing/data/processed_data/processed_legal_cases_with_years.csv \
    --output_dir preprocessing/outputs/llama_training_ready \
    --human_inspection_samples 0
```

### **Quality Assessment Only**
```bash
# Quick quality check on existing processed data
python preprocessing/scripts/utilities/run_quality_check.py \
    --data_dir preprocessing/outputs/llama_training_ready \
    --output_dir preprocessing/outputs/quality_assessment
```

### **Test Imports**
```bash
# Test that all imports work correctly
python preprocessing/scripts/utilities/test_imports.py
```

## 📊 **Output File Formats**

### **JSONL Format** (`*_chunks.jsonl`)
Each line contains a JSON object:
```json
{
  "case_id": "pre_brexit_2013_2016_EA123456", 
  "chunk_id": 0,
  "text": "The appellant appeals against...",
  "token_count": 1024,
  "type": "semantic_chunk",
  "dataset": "pre_brexit_2013_2016",
  "original_case_year": 2015,
  "case_reference": "EA/12345/2015",
  "promulgation_date": "2015-06-15",
  "chunk_length": 4096
}
```

### **Plain Text Format** (`*_training.txt`)
Raw text chunks separated by document separators:
```
[chunk text 1]

==================================================

[chunk text 2]

==================================================
```

### **Metadata CSV** (`*_metadata.csv`)
Spreadsheet with chunk information (without text column for size):
```csv
case_id,chunk_id,token_count,type,dataset,original_case_year,chunk_length
pre_brexit_2013_2016_EA123456,0,1024,semantic_chunk,pre_brexit_2013_2016,2015,4096
```

## 👁️ **Human Inspection Samples** (NEW)

### **Purpose**
Random samples of text before/after cleaning for manual quality oversight.

### **Files**
- **`*_inspection_samples.html`** - Interactive HTML files (BEST for review)
  - Open in web browser
  - Side-by-side before/after comparison
  - Toggle full text view
  - Clear visual formatting

- **`*_inspection_samples.json`** - Raw data with full text
- **`*_inspection_summary.csv`** - Quick overview spreadsheet  
- **`inspection_summary.json`** - Aggregate cleaning statistics
- **`README.md`** - Instructions for human reviewers

### **Review Process**
1. Open HTML files in web browser
2. Review 5-10 samples per dataset minimum
3. Check both preview and full text
4. Look for:
   - ✅ Proper removal of legal headers, page numbers
   - ✅ Preserved legal reasoning and facts
   - ✅ Maintained paragraph structure
   - ❌ Removed important content
   - ❌ Broken sentences/paragraphs

### **Customization**
- Default: 20 samples per dataset
- Use `--human_inspection_samples N` to customize
- Set to 0 to disable entirely

## 🎯 **For LLaMA Training**

### **Recommended Files**
- **For HuggingFace Transformers**: Use `*_chunks.jsonl` files
- **For Custom Training**: Use `*_training.txt` files  
- **For Analysis**: Use `*_metadata.csv` files

### **File Sizes (Estimated)**
- **Pre-Brexit (2013-2016)**: ~152M characters, ~15,000 chunks
- **Post-Brexit (2018-2025)**: ~257M characters, ~20,000 chunks
- **Transitional (2017)**: ~48M characters, ~4,500 chunks

## 🔍 **Quality Metrics Location**
- **Overall Score**: In `quality_assessment_report.json`
- **Visual Dashboard**: `quality_assessment_dashboard.png`
- **Training Readiness**: Check recommendations in quality report
- **Human Oversight**: Review inspection samples in `human_inspection/` directory

## 🆘 **Troubleshooting**

### **If Files Not Found**
1. Check that preprocessing pipeline completed successfully
2. Verify input data path is correct
3. Check file permissions on output directories

### **If Imports Fail**
1. Run: `python preprocessing/scripts/utilities/test_imports.py`
2. Install missing dependencies: `pip install pandas numpy matplotlib seaborn tqdm`
3. Check Python path and conda environment

### **If Quality Assessment Fails**
1. Ensure processed data exists in expected location
2. Check that JSONL files are properly formatted
3. Install visualization dependencies: `pip install matplotlib seaborn`

### **If Human Inspection Files Missing**
1. Check `--human_inspection_samples` parameter (default: 20)
2. Ensure input data has sufficient cases
3. Verify output directory permissions 