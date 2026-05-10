#!/bin/bash
# submit_curriculum_phasewise.sh
#
# Usage:
#   sbatch submit_curriculum_phasewise.sh \
#     <model_size> <ordering> <order_name> [queue_file] [queue_index]
#
#SBATCH --job-name=curr_pw
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/curriculum_phasewise_%j.out
#SBATCH --error=logs/curriculum_phasewise_%j.err
#SBATCH --mail-type=END,FAIL

set -eo pipefail
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Started: $(date)"

# -- Arguments ------------------
MODEL_SIZE="${1:-1.5b}"
ORDERING="${2:-1,2,3,4,5}"
ORDER_NAME="${3:-natural}"
QUEUE_FILE="${4:-}"
QUEUE_INDEX="${5:-1}"

echo "Model:    $MODEL_SIZE"
echo "Ordering: $ORDERING"
echo "Name:     $ORDER_NAME"

cd $SLURM_SUBMIT_DIR
source venv/bin/activate
export TOKENIZERS_PARALLELISM=false
mkdir -p logs results

# -- Model path ----------------
if [ "$MODEL_SIZE" = "1.5b" ]; then
    BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
    BATCH_SIZE=4
    GRAD_ACCUM=8
    MAX_MEM="48G"
elif [ "$MODEL_SIZE" = "7b" ]; then
    BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
    BATCH_SIZE=2
    GRAD_ACCUM=16
    MAX_MEM="80G"
else
    echo "Unknown model size: $MODEL_SIZE"; exit 1
fi

OUTPUT_BASE="checkpoints/curriculum_${MODEL_SIZE}_${ORDER_NAME}"
mkdir -p "$OUTPUT_BASE"

TEST_FILE="data/regex_dfa_dataset_test.jsonl"

IFS=',' read -ra PHASES <<< "$ORDERING"

echo ""
echo "Phase ordering: ${PHASES[*]}"
echo ""

# -- Training function -------------
train_phase() {
    local phase=$1
    local phase_num=$2
    local model_input=$3
    local output_dir=$4

    echo ""
    echo "------------------------"
    echo "  Training position $phase_num: Phase $phase data"
    echo "  Input:  $model_input"
    echo "  Output: $output_dir"
    echo "------------------------"

    mkdir -p "$output_dir"

    python train_lora.py \
        --model_path   "$model_input" \
        --train_file   "data/phase${phase}_train.jsonl" \
        --val_file     "data/phase${phase}_val.jsonl" \
        --output_dir   "$output_dir" \
        --lora_r                        32 \
        --lora_alpha                    64 \
        --lora_dropout                  0.05 \
        --num_train_epochs              3 \
        --per_device_train_batch_size   $BATCH_SIZE \
        --gradient_accumulation_steps   $GRAD_ACCUM \
        --learning_rate                 2e-4 \
        --weight_decay                  0.01 \
        --warmup_ratio                  0.05 \
        --lr_scheduler_type             cosine \
        --max_seq_length                4096 \
        --bf16 \
        --gradient_checkpointing \
        --save_steps                    200 \
        --eval_steps                    200 \
        --logging_steps                 10 \
        --save_total_limit              2 \
        --early_stopping_patience       5 \
        --report_to   none \
        --run_name    "curriculum_${MODEL_SIZE}_${ORDER_NAME}_pos${phase_num}_phase${phase}"

    echo "  Training position $phase_num (phase $phase) done: $(date)"
}

# -- Evaluation function -------------------
evaluate_after_phase() {
    local phase=$1
    local phase_num=$2
    local adapter_dir=$3
    local out_file=$4

    echo ""
    echo "  Evaluating after position $phase_num (phase $phase data)..."

    python evaluate.py \
        --base_model     "$BASE_MODEL" \
        --adapter_dir    "$adapter_dir" \
        --test_file      "$TEST_FILE" \
        --output_file    "$out_file" \
        --n_samples      200 \
        --max_new_tokens 3500 \
        --temperature    0.0

    python3 -c "
import json
with open('$out_file') as f:
    d = json.load(f)
a = d['aggregate']
t = a.get('by_tier', {})
print(
    f'  Position $phase_num (Phase $phase): '
    f'overall={a[\"exact_equiv_rate\"]:.3f}  '
    f't1={t.get(\"1\",{}).get(\"exact_equiv\",0):.3f}  '
    f't2={t.get(\"2\",{}).get(\"exact_equiv\",0):.3f}  '
    f't3={t.get(\"3\",{}).get(\"exact_equiv\",0):.3f}  '
    f't4={t.get(\"4\",{}).get(\"exact_equiv\",0):.3f}'
)
"
    echo "  Eval done: $(date)"
}

# =====================
#  MAIN LOOP
# =====================

CURRENT_MODEL="$BASE_MODEL"

for i in "${!PHASES[@]}"; do
    phase_num=$((i + 1))
    phase="${PHASES[$i]}"

    PHASE_OUTPUT="$OUTPUT_BASE/pos${phase_num}_phase${phase}"
    EVAL_OUTPUT="results/curriculum_${MODEL_SIZE}_${ORDER_NAME}_pos${phase_num}_phase${phase}.json"

    train_phase "$phase" "$phase_num" "$CURRENT_MODEL" "$PHASE_OUTPUT"
    evaluate_after_phase "$phase" "$phase_num" "$PHASE_OUTPUT" "$EVAL_OUTPUT"

    CURRENT_MODEL="$PHASE_OUTPUT"
done

# =====================
#  FINAL SUMMARY TABLE
# =====================

echo ""
echo "============================="
echo "  CURRICULUM RESULTS: ${MODEL_SIZE} | Order: ${ORDER_NAME} | ${ORDERING}"
echo "============================="
echo ""

python3 -c "
import json, os

model_size  = '$MODEL_SIZE'
order_name  = '$ORDER_NAME'
ordering    = '$ORDERING'.split(',')

print(f'Order: {ordering}')
print()
print(
    f'{\"Pos\":>4}  {\"Phase\":>5}  {\"Overall\":>8}  '
    f'{\"Tier1\":>6}  {\"Tier2\":>6}  {\"Tier3\":>6}  {\"Tier4\":>6}'
)
print('-' * 55)

for i, phase in enumerate(ordering):
    pos = i + 1
    fname = (
        f'results/curriculum_{model_size}_{order_name}'
        f'_pos{pos}_phase{phase}.json'
    )
    try:
        with open(fname) as f:
            d = json.load(f)
        a = d['aggregate']
        t = a.get('by_tier', {})
        print(
            f'{pos:>4}  {phase:>5}  '
            f'{a[\"exact_equiv_rate\"]:>8.3f}  '
            f'{t.get(\"1\",{}).get(\"exact_equiv\",0):>6.3f}  '
            f'{t.get(\"2\",{}).get(\"exact_equiv\",0):>6.3f}  '
            f'{t.get(\"3\",{}).get(\"exact_equiv\",0):>6.3f}  '
            f'{t.get(\"4\",{}).get(\"exact_equiv\",0):>6.3f}'
        )
    except FileNotFoundError:
        print(f'{pos:>4}  {phase:>5}  (not found)')

print('-' * 55)
"

echo ""
echo "Finished: $(date)"

# =====================
#  SELF-CHAIN: submit next job from queue
# =====================

if [ -n "$QUEUE_FILE" ] && [ -f "$QUEUE_FILE" ]; then
    TOTAL=$(wc -l < "$QUEUE_FILE")
    NEXT_INDEX=$((QUEUE_INDEX + 1))

    if [ "$NEXT_INDEX" -le "$TOTAL" ]; then
        NEXT=$(sed -n "${NEXT_INDEX}p" "$QUEUE_FILE")
        NEXT_MODEL=$(echo "$NEXT" | cut -d'|' -f1)
        NEXT_NAME=$(echo "$NEXT"  | cut -d'|' -f2)
        NEXT_ORDER=$(echo "$NEXT" | cut -d'|' -f3)

        echo ""
        echo "Submitting next job ($NEXT_INDEX/$TOTAL): $NEXT_MODEL $NEXT_NAME ($NEXT_ORDER)"

        sbatch \
            --job-name="curr_${NEXT_MODEL}_${NEXT_NAME}" \
            --output="logs/curriculum_${NEXT_MODEL}_${NEXT_NAME}_%j.out" \
            --error="logs/curriculum_${NEXT_MODEL}_${NEXT_NAME}_%j.err" \
            submit_curriculum_phasewise.sh \
                "$NEXT_MODEL" "$NEXT_ORDER" "$NEXT_NAME" \
                "$QUEUE_FILE" "$NEXT_INDEX"

        echo "Next job submitted. $(( TOTAL - NEXT_INDEX )) remaining."
    else
        echo ""
        echo "All $TOTAL curriculum jobs completed."
    fi
fi
