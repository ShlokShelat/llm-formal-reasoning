# Curriculum Phase Datasets

Used only for curriculum learning experiments (Section 4.1 / Table 3).

## Phase structure (Appendix C.3)
| Phase | Content             | Purpose                          |
|-------|---------------------|----------------------------------|
| 1     | Tier 1 only         | Simple concatenation/union       |
| 2     | Tier 2 only         | Kleene star, plus, optional      |
| 3     | Tier 3 only         | Combined operators + branching   |
| 4     | All tiers mixed     | Prevent catastrophic forgetting  |
| 5     | Tier 4 only         | Full subset construction         |

Each phase: 70% construction examples + 30% theory Q&A.
Uses 8-step markdown CoT format (distinct from main SFT JSON-trace format).

## Orderings evaluated (Table 3)
| Name       | Order     | T4 Acc |
|------------|-----------|--------|
| Natural    | 1→2→3→4→5 | 39.0%  |
| Reverse    | 5→4→3→2→1 | 29.3%  |
| Hard-first | 5→1→2→3→4 | 34.1%  |
| Mid-out    | 3→2→4→1→5 | 39.0%  |
| Random     | 2→5→1→4→3 | 31.7%  |

Best (Natural) still far below joint CoT SFT: 82.9% T4.
