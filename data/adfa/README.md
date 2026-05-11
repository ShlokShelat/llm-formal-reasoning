# ADFA Contrastive Dataset

Contrastive examples and variants for mechanistic analysis — Section 4.2.
Generated from Tier 4 test examples using `src/mechanistic/build_adfa_variants.py`.

## Files

| File | Description |
|------|-------------|
| `contrastive_positives.jsonl` | 243 examples the CoT-trained 7B model answers correctly (label=1) |
| `contrastive_negatives.jsonl` | 42 examples answered incorrectly (label=0) |
| `contrastive_combined.jsonl` | Merged — input to ADS neuron scoring |
| `variants_v1_alphabet_swap.jsonl` | Same DFA structure, symbols renamed |
| `variants_v2_structural_sibling.jsonl` | Different regex, isomorphic DFA |
| `variants_v3_complexity_decoy.jsonl` | Complex-looking regex → trivial 2-state DFA |
| `variants_v4_step_specific.jsonl` | Same example + step-level correctness labels (VGNS) |

## ADFA filter rule

A neuron encodes subset construction only if V3 bleed ratio < 0.5.
Paper finding: lowest observed bleed ratio is **0.73** across all 56,000 candidates.
No neuron specifically encodes the procedure — computation is fully distributed.

## Regenerate

```bash
python src/mechanistic/build_adfa_variants.py \
    --positives_file data/adfa/contrastive_positives.jsonl \
    --output_dir data/adfa
```
