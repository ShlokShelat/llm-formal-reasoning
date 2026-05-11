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
