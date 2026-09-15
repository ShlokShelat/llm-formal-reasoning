# Testing the Limits of Large Language Models on Regular Languages

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![ICLR Submission](https://img.shields.io/badge/ICLR-2027-blue)]()

Official artifact repository for *"Beyond Pattern Matching: Tracing Symbolic Reasoning Failures in LLMs to Their Mechanistic Origin"*.

---

## Overview

We probe LLM symbolic reasoning through **regular languages** — the simplest formal language
class — where correctness is fully verifiable. Using a staged diagnostic framework on
GPT-5.2, Grok-4.1, Gemini-2.5, and Qwen2.5 (1.5B / 7B / 14B), we identify
**11 systematic failure modes** and show that fine-tuning substantially closes the gap
on simpler tiers while Tier 4 failures (full subset construction) resist every
intervention tested. A mechanistic analysis — activation difference scoring, layer
probing, **activation patching**, and steering — shows the failure is not localized to any
neuron or layer and cannot be repaired by correcting the internal representation, indicating
the required computation is absent from the weights rather than merely mislocated.

---

## Setup

```bash
git clone <repo-url>
cd llm-formal-reasoning
pip install -r requirements.txt
```

Hardware: single NVIDIA H100 80 GB per training or patching job.
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
│   │   ├── build_adfa_variants.py       # Appendix G.11
│   │   ├── compute_ads_scores.py        # Appendix G.11
│   │   ├── extract_steering_vectors.py  # Appendix G.11
│   │   ├── run_steering_eval.py         # Appendix G.11
│   │   ├── run_activation_patching.py   # single-layer activation patching (Section 5.2)
│   │   ├── run_multilayer_patching.py   # windowed / all-layer patching (Section 5.2)
│   │   └── merge_patching_results.py    # combine patching runs into the recovery curve
│   └── evaluation/
│       └── evaluate.py             # Referenced in SLURM scripts
│
├── scripts/
│   ├── generate_data.sh
│   ├── run_finetuning.sh
│   ├── run_curriculum.sh
│   ├── run_frontier_eval.sh
│   ├── run_mechanistic.sh
│   ├── run_patching.sh             # launches the activation patching experiments
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

### 6 — Mechanistic analysis (Section 5.2)

Neuron scoring, steering vectors, and steering evaluation (ADS, ADFA variants, VGNS):
```bash
bash scripts/run_mechanistic.sh
```

Activation patching — the causal test of whether correcting the model's internal
representation repairs the failure. For each failing example we replace the hidden
state at a chosen layer (or set of layers) with the mean hidden state of correctly
solved examples, then check whether the output becomes correct:
```bash
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \
ADAPTER_DIR=checkpoints/qwen7b_cot \
TEST_FILE=data/finetune/cot/regex_dfa_dataset_test.jsonl \
OUTPUT_DIR=results/patching \
bash scripts/run_patching.sh
```

Or run a single mode directly:
```bash
# single-layer sweep (patch each of the 28 layers on its own; control at chosen layers)
python src/mechanistic/run_activation_patching.py \
    --base_model  Qwen/Qwen2.5-7B-Instruct --adapter_dir checkpoints/qwen7b_cot \
    --test_file   data/finetune/cot/regex_dfa_dataset_test.jsonl \
    --output_dir  results/patching/single_layer \
    --control_layers 10,20,27 --batch_size 24

# patch a set/range of layers together (e.g. all layers at once, or windows)
python src/mechanistic/run_multilayer_patching.py \
    --base_model  Qwen/Qwen2.5-7B-Instruct --adapter_dir checkpoints/qwen7b_cot \
    --test_file   data/finetune/cot/regex_dfa_dataset_test.jsonl \
    --output_dir  results/patching/all_layers \
    --layer_sets  "0-27" --batch_size 24
```

Each run writes `split.json` (baseline correct/incorrect) and a per-layer or per-set
recovery record. `merge_patching_results.py` combines the outputs of a split (e.g. two
GPUs, one covering early layers and one covering late layers) into a single
recovery-vs-layer curve. A CUDA GPU is required; the scripts refuse to run on CPU.

---

## Key Results

### Fine-Tuning Accuracy by Tier (Table 2)

| Condition   | Size | Overall | T1   | T2   | T3   | T4    |
|-------------|------|---------|------|------|------|-------|
| Zero-shot   | all  | 0%      | 0%   | 0%   | 0%   | 0%    |
| CoT SFT     | 1.5B | 96.0%   | 100% | 100% | 100% | 80.5% |
| CoT SFT     | 7B   | 96.5%   | 100% | 100% | 100% | 82.9% |
| CoT SFT     | 14B  | 96.3%   | 100% | 100% | 100% | 82.1% |
| No-CoT SFT  | 7B   | 97.0%   | 100% | 100% | 100% | 85.4% |
| No-CoT SFT  | 14B  | 96.6%   | 100% | 100% | 97.5%| 87.7% |

### Steering Interventions — Tier 4, CoT-trained 7B (Table 4)

| Condition              | Tier 4 Acc | Δ       |
|------------------------|-----------|---------|
| Baseline               | 85.3%     | —       |
| **VGNS 4-round (ours)**| **87.7%** | +2.4 pp |
| SADI                   | 86.3%     | +1.0 pp |
| Good neurons ×1.5      | 86.0%     | +0.7 pp |
| Random (control)       | 85.6%     | +0.3 pp |
| Probe direction        | 84.9%     | −0.4 pp |

### Activation Patching — Tier 4, CoT-trained 7B (Section 5.2)

Recovery of failing examples when the internal representation is replaced with the
mean correct representation. No layer, range of layers, or all-layers-at-once patch
recovers failures beyond the final-layer noise floor (the incorrect-source control
recovers as many or more), giving causal evidence that the failure is not repairable
by correcting the representation.

| Patch scope                       | Failing examples recovered |
|-----------------------------------|----------------------------|
| Each single layer (0–25)          | 0                          |
| Final layers (26, 27)             | 4–5 (control recovers ≥)   |
| Windowed ranges (except last)     | 0                          |
| All 28 layers at once             | 0                          |

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
@article{llm-regular-languages,
  title  = {Beyond Pattern Matching: Tracing Symbolic Reasoning Failures in LLMs to Their Mechanistic Origin},
  author = {Anonymous},
  year   = {2027}
}
```
