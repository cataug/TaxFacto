#!/usr/bin/env bash

cd ~/TaxFacto

PY=/home/tahiti/Malashin_Projects/.venv_a100/bin/python

unset HF_HOME

export HF_HUB_CACHE="$PWD/.cache/huggingface/hub"
export HF_DATASETS_CACHE="$PWD/.cache/huggingface/datasets"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p logs/stage1
mkdir -p results/stage1

DATASETS=(
    legaleval
    marro_india
    marro_uk
)

METHODS=(
    frozen
    full_ft
    shared_lora
    category_lora
)

for DATASET in "${DATASETS[@]}"; do
    for METHOD in "${METHODS[@]}"; do

        RUN="${DATASET}__${METHOD}__seed42"

        echo
        echo "============================================================"
        echo "START ${RUN}"
        echo "============================================================"

        "$PY" -u src/train.py \
            --dataset "$DATASET" \
            --method "$METHOD" \
            --model law-ai/InLegalBERT \
            --seed 42 \
            --epochs 4 \
            --max_length 256 \
            --batch_size 32 \
            --eval_batch_size 64 \
            --rank 8 \
            --alpha 16 \
            --dropout 0.05 \
            2>&1 | tee "logs/stage1/${RUN}.log"

        STATUS=${PIPESTATUS[0]}

        if [ "$STATUS" -ne 0 ]; then
            echo
            echo "FAILED: ${RUN}"
            echo "exit code: ${STATUS}"
            exit "$STATUS"
        fi

        echo
        echo "DONE: ${RUN}"
    done
done

echo
echo "============================================================"
echo "STAGE 1 COMPLETE"
echo "============================================================"
