# Testing the Limits of Large Language Models on Regular Languages

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![ACL Submission](https://img.shields.io/badge/ACL-2025-blue)]()

Official artifact repository for *"Testing the Limits of Large Language Models on Regular Languages"*.

---

## Overview

We probe LLM symbolic reasoning through **regular languages** — the simplest formal language
class — where correctness is fully verifiable. Using a staged diagnostic framework on
GPT-5.2, Grok-4.1, Gemini-2.5, and Qwen2.5 (1.5B / 7B / 14B), we identify
**11 systematic failure modes** and show that fine-tuning substantially closes the gap
on simpler tiers while Tier 4 failures (full subset construction) resist every
intervention tested, including our proposed VGNS framework.

---

## Setup

```bash
git clone <repo-url>
cd llm-formal-reasoning
pip install -r requirements.txt
```

Hardware: single NVIDIA H100 80 GB per training job.
Frontier model evaluation runs via API only (no local GPU needed).

---

## Repo Structure

```
llm-formal-reasoning/
├── configs/
│   ├── model_configs.yaml          # Model paths + architecture (Table 7)
│   ├── training_config.yaml        # All LoRA + training hyperparameters (Appendix D.2)
│   └── eval_config.yaml            # Evaluation settings + failure mode taxonomy
│
├── prompts/
│   ├── intuitive_construction.txt  # Appendix G.12
│   ├── derivative_construction.txt # Appendix G.12
│   └── cross_consistency.txt       # Appendix G.12
│
├── data/
│   ├── diagnostic/                 # 180-problem benchmark (Shelat et al. 2026)
│   ├── finetune/{cot,nocot}/       # Generated fine-tuning dataset
│   ├── curriculum/phase{1-5}/      # Phase-wise curriculum splits
│   └── adfa/                       # ADFA contrastive examples
│
├── src/
│   ├── dataset/
│   │   └── generate_dataset.py     # Appendix G.9
│   ├── training/
│   │   ├── train_qwen_cot.py       # Appendix G.10
│   │   └── train_lora.py           # Appendix G.10
│   ├── mechanistic/
│   │   ├── build_adfa_variants.py  # Appendix G.11
│   │   ├── compute_ace_scores.py   # Appendix G.11
│   │   ├── extract_steering_vectors.py  # Appendix G.11
│   │   └── run_steering_eval.py    # Appendix G.11
│   └── evaluation/
│       └── evaluate.py             # Referenced in SLURM scripts
│
├── scripts/
│   ├── generate_data.sh
│   ├── run_finetuning.sh
│   ├── run_curriculum.sh
│   ├── run_frontier_eval.sh
│   ├── run_mechanistic.sh
│   └── slurm/
│       ├── submit_curriculum_phasewise.sh  # Appendix G.10
│       └── launch_all_curriculum.sh        # Appendix G.10
│
├── results/                        # Output directory (gitignored)
├── requirements.txt
├── setup.py
└── LICENSE
```

---

## Step-by-Step Reproduction

### 1 — Generate the fine-tuning dataset
```bash
python src/dataset/generate_dataset.py --n 25000 --seed 42 \
    --out data/finetune/cot/regex_dfa_dataset.jsonl
```

### 2 — Diagnostic evaluation (Sections 2–3, frontier models)
```bash
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
python src/evaluation/run_tot_eval.py \
    --model gpt-5.2 \
    --benchmark data/diagnostic/benchmark_180.jsonl \
    --output_dir results/diagnostic/gpt52
```

### 3 — Fine-tuning (Section 4)
```bash
# Example: Qwen2.5-7B CoT
python src/training/train_qwen_cot.py
# All 6 combinations (3 sizes x 2 formats):
bash scripts/run_finetuning.sh
```

### 4 — Evaluate
```bash
python src/evaluation/evaluate.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --adapter_dir checkpoints/qwen7b_cot \
    --test_file data/finetune/cot/regex_dfa_dataset_test.jsonl \
    --output_file results/finetuning/qwen7b_cot.json
```

### 5 — Curriculum learning (Section 4.1)
```bash
# With SLURM:
bash scripts/slurm/launch_all_curriculum.sh 7b
# Without SLURM:
bash scripts/run_curriculum.sh --model_size 7b --order_name natural --ordering 1,2,3,4,5
```

### 6 — Mechanistic analysis (Section 4.2)
```bash
bash scripts/run_mechanistic.sh
```

---

## Key Results

### Fine-Tuning Accuracy by Tier (Table 2)

| Condition   | Size | Overall | T1   | T2   | T3   | T4    |
|-------------|------|---------|------|------|------|-------|
| Zero-shot   | all  | 0%      | 0%   | 0%   | 0%   | 0%    |
| CoT SFT     | 1.5B | 96.0%   | 100% | 100% | 100% | 80.5% |
| CoT SFT     | 7B   | 96.5%   | 100% | 100% | 100% | 82.9% |
| CoT SFT     | 14B  | 96.3%   | 100% | 100% | 100% | 82.1% |
| No-CoT SFT  | 7B   | 98.0%   | 100% | 100% | 100% | 95.4% |
| No-CoT SFT  | 14B  | 98.0%   | 100% | 100% | 97.5%| 97.7% |

### Steering Interventions — Tier 4, CoT-trained 7B (Table 4)

| Condition              | Tier 4 Acc | Δ       |
|------------------------|-----------|---------|
| Baseline               | 85.3%     | —       |
| **VGNS 4-round (ours)**| **87.7%** | +2.4 pp |
| SADI                   | 86.3%     | +1.0 pp |
| Good neurons ×1.5      | 86.0%     | +0.7 pp |
| Random (control)       | 85.6%     | +0.3 pp |
| Probe direction        | 84.9%     | −0.4 pp |

---

## Failure Mode Taxonomy

### Cross-Consistency Protocol (11 modes, Section 2)

| #    | Name                              | Task   |
|------|-----------------------------------|--------|
| i    | Anchor Hallucination              | Task 1 |
| ii   | Nullability Neglect               | Task 1 |
| iii  | Atomic Unit Blindness             | Task 1 |
| iv   | Scope and Nesting Confusion       | Task 1 |
| v    | Pseudo-Structural Hallucination   | Task 2 |
| vi   | Simple-Path Bias                  | Task 2 |
| vii  | Complexity Aversion               | Task 2 |
| viii | Trace Fabrication                 | Task 3 |
| ix   | Greedy Parsing Failures           | Task 3 |
| x    | Indexing and Positional Drift     | Task 3 |
| xi   | Descriptive–Operational Dissonance| Tasks 2–3 |

### DFA Construction (6 modes, Section 3)

| #  | Problem                                        |
|----|------------------------------------------------|
| P1 | Misinterpreting Kleene-Star Structure          |
| P2 | Errors in Brzozowski Derivative                |
| P3 | Incorrect Pre-Minimization Structure           |
| P4 | Over-Acceptance of Non-Language Strings        |
| P5 | Loss of Boundary Conditions Under Concatenation|
| P6 | Creation of Redundant States                   |

---

## Reproducibility

All experiments: `seed=42`, full determinism across Python / NumPy / PyTorch / CUDA.

```bibtex
@article{llm-regular-languages-2025,
  title  = {Testing the Limits of Large Language Models on Regular Languages},
  author = {Anonymous},
  year   = {2025}
}
```
