# Interpretative Drift: Legal Training Data and LLM Decision-Making

Code and data for the paper *Interpretative Drift: Political Transformations in Legal Training Data Systematically Reconfigure AI Reasoning* (under review).

We continued-pretrain two copies of LLaMA-3-8B on decisions of the UK Upper Tribunal (Immigration and Asylum Chamber) from two periods, 2013–2016 and 2019–2025, and compare how the two models decide the same 25,920 controlled asylum vignettes. The comparison is made at three levels: the model, the legal topic, and the individual evidentiary field.

> **Status.** This repository is being cleaned for release. Sections marked **TODO** are unconfirmed and will be updated before the camera-ready version.

---

## Pipeline

The repository follows the paper's pipeline. Each stage reads the previous stage's outputs.

| Stage | Directory | Main entry point | Paper section |
|---|---|---|---|
| 1. Collect decisions | `scraping/` | `main.py`, `run_document_extraction.py` | §3.2 |
| 2. Clean and split corpora | `preprocessing/scripts/` | `main/preprocess_legal_cases.py`, `filter_post_brexit.py`, `main/llama_preprocessing_pipeline.py` | §3.2 |
| 3. Anonymise case text | `annonymization/` | `anonymize_json.py` | §3.2 |
| 4. Continued pre-training | `training/` | `scripts/training/train.py` with `configs/*.yaml` | §3.3 |
| 5. Instruction residuals | `instruction_delta/` | `create_delta.py`, `apply_delta.py` | §3.3 |
| 6. Vignettes | `vignettes/` | `complete_vignettes.json`, `field_definitions.py` | §3.4 |
| 7. Generate decisions | `inference/` | **TODO** confirm script used for the reported run | §3.4 |
| 8. Analysis | `vignettes_analysis/comparative_fairness/` | see below | §3.5, §4 |

### 1–3. Corpora

The scraper downloads all publicly available Upper Tribunal (IAC) decisions (about 45,000, 2001–2025) with rate limiting and resumable progress:

```bash
python scraping/main.py --test          # three pages only
python scraping/main.py --resume        # full run, resuming if interrupted
```

Preprocessing extracts text from PDF and Word files, removes headers and boilerplate, and splits decisions into the two windows used in the paper:

| Corpus | Years | Cases | Tokens |
|---|---|---|---|
| Pre-Brexit | 2013–2016 | 14,609 | 35.28M |
| Post-Brexit | 2019–2025 | 14,343 | 48.35M |

2017–2018 are excluded. A 2,000-case held-out set is kept for each corpus.

### 4. Training

Both models use identical settings: LLaMA-3-8B, continued pre-training (next-token prediction), cosine schedule from 3×10⁻⁵, two epochs, effective batch size 32, bfloat16, seed 42, deterministic operations.

```bash
python training/scripts/training/train.py --config training/configs/pre_brexit_2013_2016_config.yaml
python training/scripts/training/train.py --config training/configs/post_brexit_2019_2025_config.yaml
```

`*_round2_config.yaml` retrain both models with a different training seed (100) for robustness checks. `post_brexit_2020_2025_config.yaml` is the token-matched corpus reported for comparison in Table 1; `post_brexit_2018_2025_config.yaml` is an earlier configuration not used in the paper.

### 5. Instruction residuals

Continued pre-training does not produce models that follow a response format. We add the weight difference between LLaMA-3-8B-Instruct and LLaMA-3-8B to both trained models, without further training:

```bash
python instruction_delta/create_delta.py \
    --base_path meta-llama/Meta-Llama-3-8B \
    --instruct_path meta-llama/Meta-Llama-3-8B-Instruct \
    --delta_out ./models/llama3_8b_instruction_delta --dtype fp16

python instruction_delta/apply_delta.py   # see instruction_delta/README.md for arguments
```

### 6. Vignettes

`vignettes/complete_vignettes.json` contains the 13 topic templates, and `field_definitions.py` defines the legal-context fields and demographic attributes (age, religion, gender, country of origin) that are crossed to produce the 25,920 vignettes.

### 7. Generation

Each vignette is given to both models with the same prompt, asking for reasoning followed by a decision, which is extracted as granted or refused.

**TODO:** confirm the exact script, decoding settings (the paper reports greedy decoding), repetition penalty, and maximum output length used for the reported run, and document them here.

### 8. Analysis

Run the two main calculations first; the analysis scripts read their outputs.

**Doctrinal divergence** (`normative/`): grant rates by topic and field, normalised topic drift, field-level divergence, and paired McNemar tests.

```bash
python vignettes_analysis/comparative_fairness/normative/main_calculation/grant_rate_analysis_by_vignette_fields.py
```

**Disparity drift** (`fairness/`): statistical parity for all group pairs within each topic, two-proportion z-tests with Benjamini–Hochberg correction, and classification of disparities as persisted, disappeared, or emerged.

```bash
python vignettes_analysis/comparative_fairness/fairness/main_calculation/statistical_parity_all_pairs.py
```

Topic-specific analyses reported in §4:

| Result | Script |
|---|---|
| Work Intentions nationality disparity (§4.3.1) | `fairness/analysis/topic_deep_dive/run_work_intentions_analysis.py` |
| Age disparities across topics (§4.3.2) | `fairness/analysis/age/final_age_extraction.py` |
| Persistent gender disparities (§4.4) | `fairness/analysis/gender/gender_bias_analysis.py` |
| SP vector similarity | `fairness/analysis/divergence/sp_vector_extractor.py` |

**TODO:** confirm that each number in the paper is regenerated by the scripts listed, and remove superseded outputs.

---

## Data availability

- **Decisions.** Upper Tribunal decisions are published on GOV.UK under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). The scraper reproduces the collection.
- **Vignettes.** Included in `vignettes/`.
- **Model completions.** **TODO:** the 51,840 completions (2 models × 25,920 vignettes) used in the analysis will be released at [location].
- **Model weights.** **TODO:** state whether weights are released and under what terms (LLaMA 3 licence applies).

## Installation

```bash
pip install -r requirements_full.txt   # full pipeline, including training
pip install -r requirements.txt        # scraping and analysis only
```

Training requires a GPU with at least 80 GB of memory (we used a single A100 80GB).

## Citation

**TODO:** add after review.

## Licence

Code: MIT. **TODO:** add a `LICENSE` file. Tribunal decisions: Open Government Licence v3.0.
