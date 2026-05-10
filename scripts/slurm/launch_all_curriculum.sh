# PASTE CODE FROM: Appendix G.10 (launch_all_curriculum.sh)
#!/bin/bash
# launch_all_curriculum.sh

MODEL_TARGET="${1:-both}"

declare -A ORDERINGS
ORDERINGS["natural"]="1,2,3,4,5"
ORDERINGS["reverse"]="5,4,3,2,1"
ORDERINGS["hard_first"]="5,1,2,3,4"
ORDERINGS["easy_first"]="1,5,2,4,3"
ORDERINGS["mid_out"]="3,2,4,1,5"
ORDERINGS["random"]="2,5,1,4,3"

ORDER_NAMES=(natural reverse hard_first easy_first mid_out random)

echo "============================="
echo "  Curriculum Phase-wise Experiment Launcher"
echo "  Target models: $MODEL_TARGET"
echo "  Strategy: 2 parallel chains (1.5B chain + 7B chain)"
echo "  Orderings:"
for name in "${ORDER_NAMES[@]}"; do
    echo "    $name: ${ORDERINGS[$name]}"
done
echo "============================="
echo ""

mkdir -p logs results

submit_first_in_chain() {
    local queue_file=$1
    local index=1

    local FIRST=$(head -1 "$queue_file")
    local MODEL=$(echo "$FIRST" | cut -d'|' -f1)
    local NAME=$(echo "$FIRST"  | cut -d'|' -f2)
    local ORDER=$(echo "$FIRST" | cut -d'|' -f3)
    local TOTAL=$(wc -l < "$queue_file")

    local JOB_ID=$(sbatch \
        --job-name="curr_${MODEL}_${NAME}" \
        --output="logs/curriculum_${MODEL}_${NAME}_%j.out" \
        --error="logs/curriculum_${MODEL}_${NAME}_%j.err" \
        submit_curriculum_phasewise.sh \
            "$MODEL" "$ORDER" "$NAME" "$queue_file" "$index" \
        | awk '{print $NF}')

    echo "  Submitted job 1/$TOTAL: $MODEL $NAME ($ORDER) -> Job $JOB_ID"
    echo "  Queue: $queue_file"
}

if [ "$MODEL_TARGET" = "1.5b" ] || [ "$MODEL_TARGET" = "both" ]; then
    QUEUE_15B="logs/curriculum_queue_1.5b.txt"
    > "$QUEUE_15B"
    for name in "${ORDER_NAMES[@]}"; do
        echo "1.5b|$name|${ORDERINGS[$name]}" >> "$QUEUE_15B"
    done
    echo "-- 1.5B queue (6 jobs, sequential chain) ---------------"
    cat "$QUEUE_15B"
    echo ""
    submit_first_in_chain "$QUEUE_15B"
    echo ""
fi

if [ "$MODEL_TARGET" = "7b" ] || [ "$MODEL_TARGET" = "both" ]; then
    QUEUE_7B="logs/curriculum_queue_7b.txt"
    > "$QUEUE_7B"
    for name in "${ORDER_NAMES[@]}"; do
        echo "7b|$name|${ORDERINGS[$name]}" >> "$QUEUE_7B"
    done
    echo "-- 7B queue (6 jobs, sequential chain) ------------------"
    cat "$QUEUE_7B"
    echo ""
    submit_first_in_chain "$QUEUE_7B"
    echo ""
fi

echo ""
echo "Submitted 2 jobs (one per chain)."
echo "Each chain self-submits the next job automatically."
echo "At any time: 2 jobs running (1.5B chain + 7B chain)"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Results in:    results/curriculum_*.json"
