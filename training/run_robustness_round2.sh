#!/bin/bash

# Round 2 Robustness Testing for Brexit Legal Text Analysis
# Round 1 models already exist - just need to train Round 2 with different seeds

echo "=== ROUND 2 ROBUSTNESS TESTING ==="
echo "Using existing Round 1 models, training Round 2 with seed=100"
echo ""

# Set up directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Existing Round 1 models:"
echo "📁 models/llama3_8b_pre_brexit/           (Pre-Brexit, seed=42) ✅"
echo "📁 models/llama3_8b_post_brexit_2019_2025/ (Post-Brexit, seed=42) ✅"
echo ""

echo "Training Round 2 models with seed=100..."
echo ""

echo "Training Pre-Brexit Model (2013-2016) - Round 2..."
python scripts/training/train.py configs/pre_brexit_2013_2016_round2_config.yaml
echo "✅ Pre-Brexit Round 2 Complete"
echo ""

echo "Training Post-Brexit Model (2019-2025) - Round 2..."
python scripts/training/train.py configs/post_brexit_2019_2025_round2_config.yaml
echo "✅ Post-Brexit Round 2 Complete"
echo ""

echo "=== ROBUSTNESS EXPERIMENT COMPLETE ==="
echo ""
echo "All models ready for comparison:"
echo "📁 models/llama3_8b_pre_brexit/           (Pre-Brexit, seed=42)"
echo "📁 models/llama3_8b_post_brexit_2019_2025/ (Post-Brexit, seed=42)"
echo "📁 models/llama3_8b_pre_brexit_round2/     (Pre-Brexit, seed=100) ✅ NEW"
echo "📁 models/llama3_8b_post_brexit_2019_2025_round2/ (Post-Brexit, seed=100) ✅ NEW"
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