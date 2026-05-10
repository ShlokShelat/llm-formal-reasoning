# Diagnostic Benchmark

180 regex-to-DFA problems from Shelat et al. (2026).
**Source**: arXiv:2601.13392
**Used in**: Sections 2–3 only. Never used for training.

## How to obtain
Download from the Shelat et al. artifact repository and place as:
  data/diagnostic/benchmark_180.jsonl

## Format (one JSON object per line)
```json
{
  "id":        "prob_001",
  "regex":     "(b|a(a(bb|bbb)*ba)*b)*a(a(bb|bbb)*ba)*a(bb|bbb)*bba",
  "alphabet":  ["a", "b"],
  "gold_dfa":  { "states": [...], "start": "q0", "accept": [...], "transitions": {...} },
  "edge_cases": ["aaababbb", "babbabaab", "ababbbaab"]
}
```
