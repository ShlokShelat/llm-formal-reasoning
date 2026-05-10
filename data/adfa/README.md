# ADFA Contrastive Dataset

Used for mechanistic analysis (Section 4.2).
Generated from Tier 4 test examples by src/mechanistic/build_adfa_variants.py.

## Files
| File                                | Description                                    |
|-------------------------------------|------------------------------------------------|
| contrastive_positives.jsonl         | Tier 4 examples model gets correct (label=1)   |
| contrastive_negatives.jsonl         | Tier 4 examples model gets incorrect (label=0) |
| contrastive_combined.jsonl          | Merged — used by ADS scoring                   |
| variants_v1_alphabet_swap.jsonl     | Same DFA, symbols renamed                      |
| variants_v2_structural_sibling.jsonl| Different regex, isomorphic DFA                |
| variants_v3_complexity_decoy.jsonl  | Complex-looking regex → trivial 2-state DFA    |
| variants_v4_step_specific.jsonl     | Same example + step-level annotations (VGNS)   |

## Key finding (Section 4.2)
No neuron in the 56,000-candidate pool meets the ADFA filter threshold.
Lowest observed V3 bleed ratio: 0.73 (threshold is 0.5).
