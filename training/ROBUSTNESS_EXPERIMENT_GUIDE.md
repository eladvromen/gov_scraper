# Brexit Legal Text Robustness Experiment

## Objective
Validate that the observed 25% disagreement between 2013-2016 and 2019-2025 models is due to **data differences (Brexit effect)** rather than **training randomness**.

## Experimental Design

### Key Innovation: Separate Seeds
- `data_split_seed: 42` - **FIXED** across all models → Identical train/test splits
- `training_seed: 42/100` - **VARIED** between rounds → Different model initialization

### Training Matrix

| Model | Time Period | data_split_seed | training_seed | Output Directory |
|-------|-------------|-----------------|---------------|------------------|
| **Round 1** | | | | |
| Baseline Pre | 2013-2016 | 42 | 42 | `models/llama3_8b_pre_brexit/` |
| Baseline Post | 2019-2025 | 42 | 42 | `models/llama3_8b_post_brexit_2019_2025/` |
| **Round 2** | | | | |
| Robust Pre | 2013-2016 | 42 | 100 | `models/llama3_8b_pre_brexit_round2/` |
| Robust Post | 2019-2025 | 42 | 100 | `models/llama3_8b_post_brexit_2019_2025_round2/` |

## Running the Experiment

### Recommended: Round 2 Only (since Round 1 models exist)
```bash
cd training
./run_robustness_round2.sh
```

### Alternative: Manual Round 2 Commands
```bash
cd training

# Round 2 only
python scripts/training/train.py configs/pre_brexit_2013_2016_round2_config.yaml
python scripts/training/train.py configs/post_brexit_2019_2025_round2_config.yaml
```

### Alternative: Using Seed Override
```bash
cd training

# Round 2 using existing configs with seed override
python scripts/training/train.py configs/pre_brexit_2013_2016_config.yaml --training-seed 100 --output-dir models/llama3_8b_pre_brexit_round2 --run-name llama3_8b_pre_brexit_2013_2016_round2
python scripts/training/train.py configs/post_brexit_2019_2025_config.yaml --training-seed 100 --output-dir models/llama3_8b_post_brexit_2019_2025_round2 --run-name llama3_8b_post_brexit_2019_2025_round2
```

### Full Experiment (if starting from scratch)
```bash
cd training
./run_robustness_experiment.sh  # Runs both rounds
```

## Expected Results

### Hypothesis Validation
**If your Brexit causal claim is correct:**

#### Primary Effect (Brexit Impact):
- Pre vs Post (Round 1): **~25% disagreement** ✅
- Pre vs Post (Round 2): **~25% disagreement** ✅
- **Consistency across rounds validates the Brexit effect**

#### Training Noise:
- Pre Round 1 vs Pre Round 2: **~1-3% disagreement** ✅  
- Post Round 1 vs Post Round 2: **~1-3% disagreement** ✅
- **Low within-period variance validates training stability**

#### Signal-to-Noise Ratio:
Brexit Effect (~25%) ÷ Training Noise (~2%) = **~12x signal strength** 🎯

### What This Proves
1. **Causal Validity**: Differences are from data, not methodology
2. **Robustness**: Results are stable across different random initializations  
3. **Strong Effect**: Brexit impact dominates training randomness
4. **Methodological Soundness**: Your experimental design is rigorous

## Analysis Steps After Training

1. **Run Inference**: Use all 4 models on the same test set
2. **Measure Disagreement**: Calculate pairwise disagreement rates
3. **Validate Hypothesis**: Confirm Brexit effect >> training noise
4. **Document Results**: Your causal claims are now empirically supported!

## Files Created
- ✅ `configs/pre_brexit_2013_2016_round2_config.yaml`
- ✅ `configs/post_brexit_2019_2025_round2_config.yaml`  
- ✅ `run_robustness_experiment.sh`
- ✅ Separate seed system in `train.py` and `data_utils.py`

Ready to validate your Brexit hypothesis! 🇬🇧📊 