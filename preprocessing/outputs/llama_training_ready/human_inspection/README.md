# Human Inspection Samples

Generated: 2025-06-11 12:08:18

## Purpose
These samples allow human oversight of the text preprocessing quality.
Review these to ensure the cleaning process preserves important legal content while removing artifacts.

## Files

### HTML Files (Best for Review)
- `*_inspection_samples.html` - Interactive HTML with before/after text comparison
- Open in web browser for easy side-by-side comparison

### Data Files
- `*_inspection_samples.json` - Structured data with full text
- `*_inspection_summary.csv` - Quick overview spreadsheet
- `inspection_summary.json` - Aggregate statistics

## What to Look For

### ✅ Good Cleaning
- Removed legal headers, page numbers, case citations
- Preserved legal reasoning and factual content
- Maintained paragraph structure
- Clean, readable text

### ❌ Problematic Cleaning
- Removed important legal content
- Broken sentences or paragraphs
- Missing key information
- Over-aggressive cleaning

## Datasets
- **pre_brexit_2013_2016**: 5 samples
- **transitional_2017**: 5 samples
- **post_brexit_2018_2025**: 5 samples

## Review Instructions
1. Open HTML files in web browser
2. Review 5-10 samples per dataset minimum
3. Check both preview and full text
4. Note any systematic issues
5. Provide feedback for pipeline improvements
