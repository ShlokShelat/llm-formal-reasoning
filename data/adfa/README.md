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
