#!/bin/bash
# run_patching.sh -- activation patching on the Tier 4 mechanistic subset.
#
# Set these paths before running. All are local paths / HF model ids; nothing
# is hardcoded.
#   BASE_MODEL   : base HF model id or local dir (e.g. Qwen/Qwen2.5-7B-Instruct)
#   ADAPTER_DIR  : LoRA adapter directory (adapter_config.json + weights)
#   TEST_FILE    : regex-to-DFA test jsonl (examples with metadata.tier == 4)
#   OUTPUT_DIR   : where results are written
#
# Requires: torch, transformers, peft, numpy, tqdm (see requirements.txt).
# A CUDA GPU is required; the scripts refuse to run on CPU.
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
ADAPTER_DIR="${ADAPTER_DIR:-checkpoints/cot_7b}"
TEST_FILE="${TEST_FILE:-data/finetune/regex_dfa_dataset_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/patching}"
BATCH_SIZE="${BATCH_SIZE:-24}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3500}"

# ---- single-layer sweep: patch each of the 28 layers on its own ----
# (correct-source at every layer; leave-one-out control at chosen layers)
python src/mechanistic/run_activation_patching.py \
    --base_model     "$BASE_MODEL" \
    --adapter_dir    "$ADAPTER_DIR" \
    --test_file      "$TEST_FILE" \
    --output_dir     "$OUTPUT_DIR/single_layer" \
    --control_layers 10,20,27 \
    --batch_size     "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS"

# ---- multi-layer: all layers at once (the decisive upper bound) ----
python src/mechanistic/run_multilayer_patching.py \
    --base_model     "$BASE_MODEL" \
    --adapter_dir    "$ADAPTER_DIR" \
    --test_file      "$TEST_FILE" \
    --output_dir     "$OUTPUT_DIR/all_layers" \
    --layer_sets     "0-27" \
    --batch_size     "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS"

# ---- multi-layer: windowed, to localize any recoverable region ----
python src/mechanistic/run_multilayer_patching.py \
    --base_model     "$BASE_MODEL" \
    --adapter_dir    "$ADAPTER_DIR" \
    --test_file      "$TEST_FILE" \
    --output_dir     "$OUTPUT_DIR/windows" \
    --layer_sets     "0-4;5-9;10-14;15-19;20-23;24-27" \
    --batch_size     "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS"

echo "Activation patching complete. Results in $OUTPUT_DIR"
