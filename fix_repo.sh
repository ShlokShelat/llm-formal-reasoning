#!/usr/bin/env bash
# =============================================================================
# fix_repo.sh  —  run from inside llm-formal-reasoning/ on Mac
# Fixes: adfa README, __init__.py files, dual license, removes update_repo.sh
# =============================================================================
set -euo pipefail

if [ ! -f "README.md" ] || [ ! -f "setup.py" ]; then
    echo "ERROR: Run this from inside llm-formal-reasoning/"
    exit 1
fi


# ── 1. REMOVE update_repo.sh FROM REPO ───────────────────────────────────────
echo "--- Removing update_repo.sh ---"
git rm update_repo.sh 2>/dev/null || rm -f update_repo.sh
echo "✓ update_repo.sh removed"


# ── 2. UPDATE data/adfa/README.md ────────────────────────────────────────────
echo "--- Updating data/adfa/README.md ---"
cat > data/adfa/README.md << 'EOF'
# ADFA Contrastive Dataset

Contrastive examples and variants for mechanistic analysis — Section 4.2.
Generated from Tier 4 test examples using `src/mechanistic/build_adfa_variants.py`.

## Files

### Contrastive split

| File | Description |
|------|-------------|
| `contrastive_positives.jsonl` | 243 Tier 4 examples the CoT-trained 7B model answers **correctly** (label=1) |
| `contrastive_negatives.jsonl` | 42 Tier 4 examples answered **incorrectly** (label=0) |
| `contrastive_combined.jsonl` | Positives + negatives merged — input to ADS neuron scoring |
| `contrastive_summary.json` | Summary statistics: counts, accuracy, tier breakdown |

### Variant types

| File | Type | Description |
|------|------|-------------|
| `variants_v1_alphabet_swap.jsonl` | V1 | Same regex/DFA structure, alphabet symbols renamed |
| `variants_v1_combined.jsonl` | V1 | V1 variants merged with originals for scoring |
| `variants_v2_structural_sibling.jsonl` | V2 | Different regex, isomorphic DFA |
| `variants_v2_combined.jsonl` | V2 | V2 variants merged with originals for scoring |
| `variants_v3_complexity_decoy.jsonl` | V3 | Complex-looking regex → trivial 2-state DFA |
| `variants_v3_combined.jsonl` | V3 | V3 variants merged with originals for scoring |
| `variants_v4_step_specific.jsonl` | V4 | Same example + step-level correctness labels (for VGNS) |
| `variants_summary.json` | — | Coverage stats: V1/V2/V3/V4 counts and percentages |

## ADFA filter rule (Appendix G.11)

A neuron encodes subset construction if all three hold simultaneously:
- ADS_good on V1 (alphabet swap) > threshold
- ADS_good on V2 (structural sibling) > threshold
- V3 bleed ratio < 0.5 — fires at less than half its peak on a complexity decoy

**Paper finding**: No neuron in the 56,000-candidate pool meets this threshold.
Lowest observed V3 bleed ratio: **0.73**.
Computation of subset construction is fully distributed — no small identifiable neuron set.

## Mechanistic analysis subset

285 Tier 4 test examples held out exclusively for mechanistic analysis:
- 243 correct → `contrastive_positives.jsonl`
- 42 incorrect → `contrastive_negatives.jsonl`

Baseline accuracy on this subset: **85.3%** (full Tier 4 test set: 82.9%).

## Regenerate

```bash
python src/mechanistic/build_adfa_variants.py \
    --positives_file data/adfa/contrastive_positives.jsonl \
    --output_dir data/adfa
```
EOF
echo "✓ data/adfa/README.md updated"


# ── 3. src/__init__.py FILES ──────────────────────────────────────────────────
echo "--- Updating __init__.py files ---"

cat > src/__init__.py << 'EOF'
"""
llm-formal-reasoning
====================
Testing the Limits of Large Language Models on Regular Languages.

Subpackages
-----------
dataset     : Regex-to-DFA dataset generation (Appendix G.9)
evaluation  : Tree-of-Thought experiment driver (Appendix G.11)
mechanistic : ADS neuron scoring, steering vector extraction, VGNS (Appendix G.11)
training    : LoRA fine-tuning scripts for Qwen2.5 (Appendix G.10)
"""
EOF

cat > src/dataset/__init__.py << 'EOF'
"""
dataset
=======
Regex-to-DFA dataset generator — Appendix G.9.

Generates CoT training examples by running:
  (1) Thompson's NFA construction
  (2) ε-closure and subset (powerset) construction
  (3) Hopcroft's minimisation
  (4) Programmatic verification on all strings up to length 7
  (5) Qwen ChatML formatting with full intermediate trace

Entry point
-----------
  python src/dataset/generate_dataset.py --n 25000 --seed 42 \
      --out data/finetune/regex_dfa_dataset.jsonl

Tiers
-----
  Tier 1 (~15%): direct symbol mapping, 2-state DFAs
  Tier 2 (~25%): Kleene star, plus, optional
  Tier 3 (~35%): alternation and branching
  Tier 4 (~25%): full NFA-to-DFA subset construction
"""
EOF

cat > src/evaluation/__init__.py << 'EOF'
"""
evaluation
==========
Tree-of-Thought DFA construction evaluation — Appendix G.11.

Evaluates frontier models (GPT-5.2, Grok-4.1, Gemini-2.5) on the
180-problem diagnostic benchmark under two construction strategies:

  Intuitive   : model builds DFA without any procedural template
  Derivative  : model follows Brzozowski derivative procedure

Each strategy is one ToT branch. Up to 3 retries per branch on JSON
parse failure. Results saved independently; no branch scoring.

Entry point
-----------
  python src/evaluation/ToT_Driver.py \
      --model gpt-5.2 \
      --benchmark data/diagnostic/benchmark_180.jsonl \
      --prompt_dir prompts/

See also: tot_experiments/ for the full experiment runner and example outputs.
"""
EOF

cat > src/mechanistic/__init__.py << 'EOF'
"""
mechanistic
===========
Mechanistic analysis pipeline — Section 4.2 and Appendix G.11.

Modules
-------
build_adfa_variants.py
    Generates four contrast variant types (V1-V4) from Tier 4 positives:
      V1: same DFA, symbols renamed (alphabet swap)
      V2: different regex, isomorphic DFA (structural sibling)
      V3: complex-looking regex, trivial 2-state DFA (complexity decoy)
      V4: same example with step-level correctness annotations (VGNS)

compute_ace_scores.py
    Activation Difference Scoring (ADS) across all 28 transformer layers.
    Samples 2,000 neurons per layer (56,000 neuron-layer pairs total).
    Identifies good (facilitating) and bad (inhibiting) neurons.
    Key finding: Layer 27 hosts 22 of the top-25 ADS neurons (execution peak).

extract_steering_vectors.py
    Extracts layer-wise steering vectors via three methods:
      ActAdd       : mean difference vector (fixed global shift)
      Probe dir    : logistic regression coefficient at Layer 10
      SADI         : sparse masking — top-10% dimensions by absolute value
    Key finding: Layer 10 encodes 87.1% of outcome information before
    the model completes half its computation.

run_steering_eval.py
    Evaluates all steering conditions on 285 Tier 4 mechanistic examples:
      Neuron-level : good/bad neuron amplification and suppression
      Representation: ActAdd, probe direction, SADI adaptive steering
      VGNS         : Verification-Guided Neuron Steering (4-round iterative)
    Key finding: VGNS recovers +2.4 pp — 35 examples remain unsolvable.
"""
EOF

cat > src/training/__init__.py << 'EOF'
"""
training
========
LoRA fine-tuning scripts for Qwen2.5 — Appendix G.10.

Modules
-------
train_qwen_cot.py
    Representative CoT fine-tuning script for Qwen2.5-7B-Instruct.
    Uses the full Thompson → subset construction → Hopcroft CoT trace.
    Evaluated every 200 steps on 200 IID + 200 OOD held-out examples.
    Same configuration used for 1.5B and 14B (different model path only).

train_lora.py
    Generic LoRA SFT script with early stopping and optional replay buffer.
    Used for all curriculum learning experiments (phase-wise sequential
    fine-tuning with self-chaining SLURM job queue).

LoRA configuration (identical across all runs)
-----------------------------------------------
  r=32, alpha=64, dropout=0.05
  Target modules: q/k/v/o_proj, gate/up/down_proj
  Optimizer: AdamW, lr=2e-4, cosine schedule, warmup 0.03
  Epochs: 2, effective batch size: 8, precision: bfloat16, seed: 42
"""
EOF

echo "✓ All __init__.py files updated"


# ── 4. DUAL LICENSE (MIT for code, CC BY 4.0 for data) ───────────────────────
echo "--- Setting up dual license ---"
mkdir -p LICENSES

# MIT License for code
cat > LICENSES/LICENSE-CODE << 'EOF'
MIT License

Copyright (c) 2025 The Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

# CC BY 4.0 for data
cat > LICENSES/LICENSE-DATA << 'EOF'
Creative Commons Attribution 4.0 International (CC BY 4.0)

Copyright (c) 2025 The Authors

This license applies to all dataset files in the data/ directory of this
repository, including:
  - data/finetune/     (regex-to-DFA fine-tuning benchmark)
  - data/curriculum/   (phase-wise curriculum datasets)
  - data/adfa/         (ADFA contrastive examples and variants)
  - data/diagnostic/   (diagnostic benchmark — Shelat et al. 2026)

You are free to:
  Share  — copy and redistribute the material in any medium or format
  Adapt  — remix, transform, and build upon the material for any purpose,
            even commercially

Under the following terms:
  Attribution — You must give appropriate credit, provide a link to the
                license, and indicate if changes were made. You may do so
                in any reasonable manner, but not in any way that suggests
                the licensor endorses you or your use.

No additional restrictions — You may not apply legal terms or technological
measures that legally restrict others from doing anything the license permits.

Full license text: https://creativecommons.org/licenses/by/4.0/legalcode
EOF

# Root LICENSE — dual notice
cat > LICENSE << 'EOF'
This repository uses a dual license:

  CODE  (src/, scripts/, configs/, prompts/, tot_experiments/*.py, setup.py)
  Licensed under the MIT License.
  See LICENSES/LICENSE-CODE for the full text.

  DATA  (data/)
  Licensed under the Creative Commons Attribution 4.0 International License
  (CC BY 4.0).
  See LICENSES/LICENSE-DATA for the full text.

Copyright (c) 2025 The Authors
EOF

echo "✓ Dual license created (MIT for code, CC BY 4.0 for data)"


# ── 5. UPDATE README LICENSE SECTION ─────────────────────────────────────────
echo "--- Updating README license badge and section ---"
# Replace the single MIT badge with dual badges
sed -i '' 's|!\[License: MIT\](https://img.shields.io/badge/License-MIT-yellow.svg)(LICENSE)|[![License: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](LICENSES/LICENSE-CODE) [![License: CC BY 4.0](https://img.shields.io/badge/Data-CC--BY--4.0-lightblue.svg)](LICENSES/LICENSE-DATA)|' README.md

# Replace the license section at the bottom
python3 - << 'PYEOF'
with open("README.md", "r") as f:
    content = f.read()

old = """## License

MIT License — see [LICENSE](LICENSE).
All datasets are fully synthetic with no privacy, copyright, or consent concerns."""

new = """## License

This repository uses a dual license:

| Component | License |
|-----------|---------|
| Code — `src/`, `scripts/`, `configs/`, `prompts/`, `tot_experiments/*.py` | [MIT](LICENSES/LICENSE-CODE) |
| Data — `data/` | [CC BY 4.0](LICENSES/LICENSE-DATA) |

All datasets are fully synthetic with no privacy, copyright, or consent concerns.
If you use this data, please cite the paper."""

content = content.replace(old, new)
with open("README.md", "w") as f:
    f.write(content)
print("README license section updated")
PYEOF

echo "✓ README updated"


# ── 6. COMMIT AND PUSH ───────────────────────────────────────────────────────
git add .
git status

echo ""
echo "Committing..."
git commit -m "Fix __init__.py docstrings, dual license, adfa README, remove update_repo.sh

- src/__init__.py: add package docstring
- src/dataset/__init__.py: document generator pipeline and tiers
- src/evaluation/__init__.py: document ToT evaluation setup
- src/mechanistic/__init__.py: document all 4 modules with key findings
- src/training/__init__.py: document both training scripts and LoRA config
- data/adfa/README.md: add new files (contrastive_summary, variants_summary,
  v1/v2/v3_combined variants)
- LICENSE: dual license notice (MIT for code, CC BY 4.0 for data)
- LICENSES/LICENSE-CODE: full MIT text
- LICENSES/LICENSE-DATA: full CC BY 4.0 text
- README.md: dual license badges and table
- remove update_repo.sh (one-time utility, not needed in repo)"

git push origin main

echo ""
echo "=============================================="
echo "  All done."
echo "=============================================="
