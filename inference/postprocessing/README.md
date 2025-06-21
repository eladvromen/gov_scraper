# Postprocessing Scripts for UK Immigration LLM Study

This directory contains scripts for post-processing raw inference results from the UK immigration LLM study, extracting clean decisions and reasoning while flagging edge cases for analytical purposes.

## Files

- `process_inference_results.py` - Main processing script
- `test_processing.py` - Test script to verify functionality
- `README.md` - This documentation file

## Usage

### Basic Usage

```bash
# Process a single inference results file
python process_inference_results.py subset_inference_llama3_8b_pre_brexit_20250620_120959.json

# Process with custom output directory
python process_inference_results.py subset_inference_llama3_8b_pre_brexit_20250620_120959.json --output-dir ../results/processed

# Verbose output
python process_inference_results.py subset_inference_llama3_8b_pre_brexit_20250620_120959.json -v
```

### Testing

```bash
# Run tests to verify functionality
python test_processing.py
```

## Output Structure

Each processed case includes:
- `decision`: Primary extracted decision (Granted/Denied)
- `all_decisions`: List of all decisions found with confidence scores
- `reasoning`: Extracted legal reasoning text
- `original_response`: Full original model response text
- `cleaned_response`: Response text after removing code contamination and malformed content
- `flags`: List of quality issues detected
- `metrics`: Quantitative measures of response quality
- `metadata`: Original case information (name, age, country, etc.)

## Processing Pipeline

The script follows a two-stage process:

1. **Cleaning Stage**: Removes code contamination and malformed content
   - Detects common code patterns (def, class, import, etc.)
   - Stops processing when code contamination is detected
   - Removes common artifacts (```, """, etc.)

2. **Extraction Stage**: Analyzes the cleaned text
   - Extracts decisions and reasoning from cleaned text
   - Performs quality checks on cleaned content
   - Flags code contamination in original text for research purposes

### Summary Statistics
- Decision distribution (granted/denied rates)
- Flag type frequencies
- Quality metrics (response length, completion rates)

## Flag Types

### 1. Conflicting Decisions
Detects when a response contains multiple contradictory decisions:
```json
{
  "type": "conflicting_decisions",
  "decisions": ["granted", "denied"],
  "count": 2,
  "positions": [0, 150]
}
```

### 2. No Decision
Detects when no clear decision is found:
```json
{
  "type": "no_decision",
  "reason": "no_granted_denied_found"
}
```

### 3. Incomplete Response
Detects truncated or incomplete responses:
```json
{
  "type": "incomplete",
  "reason": "abrupt_ending"
}
```

### 4. Code Contamination
Detects programming code in responses:
```json
{
  "type": "code_contamination",
  "patterns_found": ["def ", "class "],
  "count": 2
}
```

## Analytical Value

These flags serve as important metrics for your Brexit research:

### Model Uncertainty
- **Conflict rates** indicate areas where the model lacks confidence
- **No-decision rates** show cases the model finds ambiguous
- **Incomplete rates** suggest model capacity limitations

### Demographic Analysis
- Correlate flag rates with demographic attributes (age, religion, gender)
- Identify which groups trigger more model uncertainty
- Analyze bias patterns in decision-making

### Temporal Analysis
- Compare pre-Brexit vs post-Brexit flag rates
- Identify shifts in model confidence across time periods
- Track changes in decision-making patterns

### Legal Complexity
- **Conflicting decisions** may indicate genuinely difficult legal questions
- **Code contamination** suggests training data quality issues
- **Incomplete responses** might indicate complex cases beyond model capacity

## Example Output

```json
{
  "metadata": {
    "input_file": "../results/subset_inference_llama3_8b_pre_brexit_20250620_120959.json",
    "processing_timestamp": "2025-06-20T14:30:00",
    "total_items_processed": 1000
  },
  "summary_statistics": {
    "total_responses": 1000,
    "decision_stats": {
      "granted": 650,
      "denied": 300,
      "granted_rate": 0.68,
      "denied_rate": 0.32,
      "no_decision_count": 50
    },
    "flag_stats": {
      "conflicting_decisions": 25,
      "code_contamination": 150,
      "incomplete": 30
    },
    "quality_metrics": {
      "avg_response_length": 45.2,
      "has_decision_rate": 0.95,
      "has_reasoning_rate": 0.92
    }
  },
  "processed_results": [...]
}
```

## Research Applications

This postprocessing enables:

1. **Clean Analysis**: Extract reliable decisions and reasoning for main analysis
2. **Quality Assessment**: Monitor model performance and data quality
3. **Bias Detection**: Identify patterns in model uncertainty and decision-making
4. **Temporal Comparison**: Compare pre/post-Brexit model behavior
5. **Methodological Validation**: Ensure analysis is based on reliable model outputs

## Future Enhancements

Potential improvements:
- More sophisticated reasoning extraction
- Semantic analysis of reasoning quality
- Integration with legal precedent databases
- Automated bias detection algorithms
- Cross-validation with human expert decisions 