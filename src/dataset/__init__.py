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
