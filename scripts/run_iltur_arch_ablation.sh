#!/usr/bin/env bash

cd ~/TaxFacto

PY=/home/tahiti/Malashin_Projects/.venv_a100/bin/python
MODEL=/home/tahiti/TaxFacto/models/InLegalBERT

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

DATASETS=(
    iltur_cl
    iltur_it
)

METHODS=(
    frozen
    full_ft
    shared_lora
    shared_lora_widehead
    category_lora
    role_adapter
)

mkdir -p logs/arch_ablation
mkdir -p results/stage1

TOTAL=12
N=0

for DATASET in "${DATASETS[@]}"; do
    for METHOD in "${METHODS[@]}"; do

        N=$((N + 1))
        RUN="${DATASET}__${METHOD}__seed42"

        echo
        echo "============================================================"
        echo "[$N/$TOTAL] ${RUN}"
        echo "============================================================"

        # Protect against accidental reruns.
        SUMMARY="results/stage1/${RUN}/smoke_dev_summary.json"

        if [ -s "$SUMMARY" ]; then
            echo "SKIP COMPLETE: ${RUN}"
            continue
        fi

        rm -rf "results/stage1/${RUN}"

        "$PY" -u src/train.py \
            --dataset "$DATASET" \
            --method "$METHOD" \
            --model "$MODEL" \
            --seed 42 \
            --epochs 4 \
            --max_length 256 \
            --batch_size 32 \
            --eval_batch_size 64 \
            --rank 8 \
            --alpha 16 \
            --dropout 0.05 \
            --skip_test \
            2>&1 | tee "logs/arch_ablation/${RUN}.log"

        STATUS=${PIPESTATUS[0]}

        if [ "$STATUS" -ne 0 ]; then
            echo "FAILED: ${RUN}"
            exit "$STATUS"
        fi

        echo "DONE: ${RUN}"
    done
done

echo
echo "============================================================"
echo "IL-TUR ARCHITECTURE ABLATION COMPLETE"
echo "============================================================"
