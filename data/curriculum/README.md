# Curriculum Phase Datasets

Phase-wise curriculum learning datasets — Section 4.1 (Table 3).
Not used in the main CoT / No-CoT SFT experiments.

## Files

| File | Phase | Content | Purpose |
|------|-------|---------|---------|
| `curriculum_phase1.jsonl` | 1 | Tier 1 | Simple concatenation/union |
| `curriculum_phase2.jsonl` | 2 | Tier 2 | Kleene star, plus, optional |
| `curriculum_phase3.jsonl` | 3 | Tier 3 | Combined operators + branching |
| `curriculum_phase4.jsonl` | 4 | All tiers | Prevent catastrophic forgetting |
| `curriculum_phase5.jsonl` | 5 | Tier 4 | Full subset construction |
| `curriculum_train.jsonl` | — | All | Joint training split |
| `curriculum_val.jsonl` | — | All | Validation split |
| `curriculum_test.jsonl` | — | All | Test split |

Each file: 70% construction examples + 30% theory Q&A in Qwen ChatML format.

## Curriculum orderings (Table 3, 7B model)

| Order | Sequence | Overall | Tier 4 |
|-------|----------|---------|--------|
| Natural | 1→2→3→4→5 | 43.5% | 39.0% |
| Mid-out | 3→2→4→1→5 | 43.5% | 39.0% |
| Hard-first | 5→1→2→3→4 | 39.0% | 34.1% |
| Random | 2→5→1→4→3 | 37.0% | 31.7% |
| Reverse | 5→4→3→2→1 | 26.0% | 29.3% |
| **Joint CoT SFT** | — | **96.5%** | **82.9%** |

No ordering closes the gap to joint SFT.

## Usage

```bash
bash scripts/slurm/launch_all_curriculum.sh 7b
# or single run:
bash scripts/slurm/submit_curriculum_phasewise.sh 7b 1,2,3,4,5 natural
```
