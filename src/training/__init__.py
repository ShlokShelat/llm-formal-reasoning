"""
training
========
LoRA fine-tuning scripts for Qwen2.5 — Appendix G.10.

Modules
-------
train_qwen_cot.py
    Representative CoT fine-tuning script for Qwen2.5-7B-Instruct.
    Uses the full Thompson → subset construction → Hopcroft CoT trace.
    Evaluated every 200 steps on 200 IID + 200 OOD held-out examples.
    Same configuration used for 1.5B and 14B (different model path only).

train_lora.py
    Generic LoRA SFT script with early stopping and optional replay buffer.
    Used for all curriculum learning experiments (phase-wise sequential
    fine-tuning with self-chaining SLURM job queue).

LoRA configuration (identical across all runs)
-----------------------------------------------
  r=32, alpha=64, dropout=0.05
  Target modules: q/k/v/o_proj, gate/up/down_proj
  Optimizer: AdamW, lr=2e-4, cosine schedule, warmup 0.03
  Epochs: 2, effective batch size: 8, precision: bfloat16, seed: 42
"""
