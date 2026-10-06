#!/usr/bin/env bash

cd ~/TaxFacto

PY=/home/tahiti/Malashin_Projects/.venv_a100/bin/python

RESULT_ROOT="results/final_common7"
LOG_ROOT="logs/final_common7"

mkdir -p "$RESULT_ROOT"
mkdir -p "$LOG_ROOT"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

MODELS_KEYS=(
    inlegalbert
    legalbert
    deberta
)

DATASETS=(
    legaleval
    marro_india
    marro_uk
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

SEEDS=(
    42
    43
    44
)

TOTAL=335
CURRENT=0
DONE=0
SKIPPED=0


model_path() {
    case "$1" in
        inlegalbert)
            echo "/home/tahiti/TaxFacto/models/InLegalBERT"
            ;;
        legalbert)
            echo "/home/tahiti/TaxFacto/models/LegalBERT"
            ;;
        deberta)
            echo "/home/tahiti/TaxFacto/models/DeBERTa-v3-base"
            ;;
        *)
            echo "UNKNOWN_MODEL"
            ;;
    esac
}


run_one() {
    PHASE="$1"
    MODEL_KEY="$2"
    DATASET="$3"
    METHOD="$4"
    SEED="$5"
    RANK="$6"
    ALPHA="$7"

    CURRENT=$((CURRENT + 1))

    MODEL=$(model_path "$MODEL_KEY")

    TAG="${PHASE}__${MODEL_KEY}__r${RANK}a${ALPHA}"

    RUN="${TAG}__${DATASET}__${METHOD}__seed${SEED}"

    SUMMARY="${RESULT_ROOT}/${RUN}/run_summary.json"

    LOG="${LOG_ROOT}/${RUN}.log"

    echo
    echo "######################################################################"
    echo "[$CURRENT/$TOTAL]"
    echo "PHASE   : $PHASE"
    echo "MODEL   : $MODEL_KEY"
    echo "DATASET : $DATASET"
    echo "METHOD  : $METHOD"
    echo "SEED    : $SEED"
    echo "RANK    : $RANK"
    echo "ALPHA   : $ALPHA"
    echo "######################################################################"

    if [ -s "$SUMMARY" ]; then
        echo "SKIP COMPLETE: $RUN"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    rm -rf "${RESULT_ROOT:?}/${RUN}"

    "$PY" -u src/train.py \
        --dataset "$DATASET" \
        --method "$METHOD" \
        --model "$MODEL" \
        --seed "$SEED" \
        --epochs 4 \
        --max_length 256 \
        --batch_size 32 \
        --eval_batch_size 64 \
        --rank "$RANK" \
        --alpha "$ALPHA" \
        --dropout 0.05 \
        --run_tag "$TAG" \
        --result_root "$RESULT_ROOT" \
        2>&1 | tee "$LOG"

    STATUS=${PIPESTATUS[0]}

    if [ "$STATUS" -ne 0 ]; then
        echo
        echo "######################################################################"
        echo "FAILED"
        echo "$RUN"
        echo "exit code: $STATUS"
        echo "######################################################################"
        exit "$STATUS"
    fi

    DONE=$((DONE + 1))

    echo
    echo "DONE: $RUN"
}


echo
echo "======================================================================"
echo "TAXFACTO MASSIVE COMMON-7 EXPERIMENT"
echo "TOTAL PLANNED RUNS: $TOTAL"
echo "======================================================================"

# =====================================================================
# PHASE A
#
# Full benchmark:
#
# 5 datasets
# x 3 backbones
# x 6 methods
# x 3 seeds
# = 270 runs
# =====================================================================

for MODEL_KEY in "${MODELS_KEYS[@]}"; do

    for DATASET in "${DATASETS[@]}"; do

        for METHOD in "${METHODS[@]}"; do

            for SEED in "${SEEDS[@]}"; do

                run_one \
                    "main" \
                    "$MODEL_KEY" \
                    "$DATASET" \
                    "$METHOD" \
                    "$SEED" \
                    8 \
                    16

            done
        done
    done
done


# =====================================================================
# PHASE B
#
# Rank sensitivity on representative backbone.
#
# r=8 is already contained in main grid.
#
# 5 datasets
# x 3 PEFT methods
# x 4 additional ranks
# x seed42
# = 60 runs
# =====================================================================

RANK_METHODS=(
    shared_lora
    category_lora
    role_adapter
)

RANKS=(
    2
    4
    16
    32
)

for DATASET in "${DATASETS[@]}"; do

    for METHOD in "${RANK_METHODS[@]}"; do

        for RANK in "${RANKS[@]}"; do

            ALPHA=$((2 * RANK))

            run_one \
                "rank" \
                "inlegalbert" \
                "$DATASET" \
                "$METHOD" \
                42 \
                "$RANK" \
                "$ALPHA"

        done
    done
done


# =====================================================================
# PHASE C
#
# Expensive genuine role-specific LoRA.
#
# 5 datasets x one representative setup = 5 runs
# =====================================================================

for DATASET in "${DATASETS[@]}"; do

    run_one \
        "mechanistic" \
        "inlegalbert" \
        "$DATASET" \
        "true_category_lora" \
        42 \
        8 \
        16

done


echo
echo "======================================================================"
echo "MASSIVE RUN COMPLETE"
echo "planned : $TOTAL"
echo "executed: $DONE"
echo "skipped : $SKIPPED"
echo "======================================================================"
