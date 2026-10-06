#!/usr/bin/env bash

cd ~/TaxFacto

PY=/home/tahiti/Malashin_Projects/.venv_a100/bin/python
MODEL=/home/tahiti/TaxFacto/models/InLegalBERT

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

METHODS=(
    shared_lora_widehead
    role_adapter
    true_category_lora
)

mkdir -p logs/smoke_arch

for METHOD in "${METHODS[@]}"; do

    RUN="legaleval__${METHOD}__seed42"

    rm -rf "results/stage1/${RUN}"

    echo
    echo "============================================================"
    echo "SMOKE: ${RUN}"
    echo "============================================================"

    "$PY" -u src/train.py \
        --dataset legaleval \
        --method "$METHOD" \
        --model "$MODEL" \
        --seed 42 \
        --epochs 1 \
        --max_length 256 \
        --batch_size 32 \
        --eval_batch_size 64 \
        --rank 8 \
        --alpha 16 \
        --dropout 0.05 \
        --skip_test \
        2>&1 | tee "logs/smoke_arch/${RUN}.log"

    STATUS=${PIPESTATUS[0]}

    if [ "$STATUS" -ne 0 ]; then
        echo "FAILED: ${RUN}"
        exit "$STATUS"
    fi

done

echo
echo "ALL ARCHITECTURE SMOKES COMPLETE"
