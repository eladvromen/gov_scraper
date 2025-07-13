#!/bin/bash

# Robustness Experiment for Brexit Legal Text Analysis
# Tests whether 25% disagreement is due to data differences (Brexit) vs training randomness

echo "=== ROBUSTNESS EXPERIMENT: Brexit Legal Text Analysis ==="
echo "Testing hypothesis: 25% disagreement is from data, not training randomness"
echo ""

# Set up directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== ROUND 1: Baseline Training (seed=42) ==="
echo ""

echo "Training Pre-Brexit Model (2013-2016) - Round 1..."
python scripts/training/train.py configs/pre_brexit_2013_2016_config.yaml
echo "✅ Pre-Brexit Round 1 Complete"
echo ""

echo "Training Post-Brexit Model (2019-2025) - Round 1..."
python scripts/training/train.py configs/post_brexit_2019_2025_config.yaml
echo "✅ Post-Brexit Round 1 Complete"
echo ""

echo "=== ROUND 2: Robustness Testing (seed=100) ==="
echo ""

echo "Training Pre-Brexit Model (2013-2016) - Round 2..."
python scripts/training/train.py configs/pre_brexit_2013_2016_round2_config.yaml
echo "✅ Pre-Brexit Round 2 Complete"
echo ""

echo "Training Post-Brexit Model (2019-2025) - Round 2..."
python scripts/training/train.py configs/post_brexit_2019_2025_round2_config.yaml
echo "✅ Post-Brexit Round 2 Complete"
echo ""

echo "=== EXPERIMENT COMPLETE ==="
echo ""
echo "Models trained:"
echo "📁 models/llama3_8b_pre_brexit/           (Pre-Brexit, seed=42)"
echo "📁 models/llama3_8b_post_brexit_2019_2025/ (Post-Brexit, seed=42)"
echo "📁 models/llama3_8b_pre_brexit_round2/     (Pre-Brexit, seed=100)"
echo "📁 models/llama3_8b_post_brexit_2019_2025_round2/ (Post-Brexit, seed=100)"
echo ""
echo "Next steps:"
echo "1. Run inference on all 4 models with the same test set"
echo "2. Compare disagreement rates:"
echo "   - Pre vs Post (Round 1): Should be ~25% (Brexit effect)"
echo "   - Pre vs Post (Round 2): Should be ~25% (Brexit effect preserved)"
echo "   - Pre Round 1 vs Pre Round 2: Should be ~1-3% (training noise)"
echo "   - Post Round 1 vs Post Round 2: Should be ~1-3% (training noise)"
echo ""
echo "If Brexit effect (25%) >> training noise (1-3%), your causal claim is validated! 🎯" 