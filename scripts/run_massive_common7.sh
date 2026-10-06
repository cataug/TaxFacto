#!/usr/bin/env bash

cd ~/TaxFacto || exit 1

PY=/home/tahiti/Malashin_Projects/.venv_a100/bin/python

RESULT_ROOT="results/final_common7"
LOG_ROOT="logs/final_common7"
FAIL_ROOT="${LOG_ROOT}/failed"

mkdir -p "$RESULT_ROOT"
mkdir -p "$LOG_ROOT"
mkdir -p "$FAIL_ROOT"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1


# =====================================================================
# CONCURRENCY
#
# Override at launch if desired, e.g.
#
# BERT_JOBS=2 DEBERTA_JOBS=1 ./scripts/run_massive_common7.sh
# =====================================================================

BERT_JOBS="${BERT_JOBS:-3}"
DEBERTA_JOBS="${DEBERTA_JOBS:-2}"
RANK_JOBS="${RANK_JOBS:-3}"
MECH_JOBS="${MECH_JOBS:-1}"


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

BERT_MODELS=(
    inlegalbert
    legalbert
)

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

TOTAL=335


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
            return 1
            ;;
    esac
}


# =====================================================================
# Cleanup only after a completed run.
# Keeps metrics/reports/predictions, removes model checkpoints.
# =====================================================================

cleanup_checkpoints() {
    RUN_DIR="$1"

    if [ ! -d "$RUN_DIR" ]; then
        return 0
    fi

    find "$RUN_DIR" \
        -type d \
        -name 'checkpoint-*' \
        -prune \
        -exec rm -rf {} + \
        2>/dev/null

    rm -rf \
        "$RUN_DIR/best_model" \
        "$RUN_DIR/checkpoints" \
        2>/dev/null
}


# =====================================================================
# One experiment
# =====================================================================

run_one() {

    PHASE="$1"
    MODEL_KEY="$2"
    DATASET="$3"
    METHOD="$4"
    SEED="$5"
    RANK="$6"
    ALPHA="$7"

    MODEL="$(model_path "$MODEL_KEY")"

    if [ "$MODEL" = "UNKNOWN_MODEL" ]; then
        echo "Unknown model key: $MODEL_KEY"
        return 1
    fi

    TAG="${PHASE}__${MODEL_KEY}__r${RANK}a${ALPHA}"

    RUN="${TAG}__${DATASET}__${METHOD}__seed${SEED}"

    RUN_DIR="${RESULT_ROOT}/${RUN}"
    SUMMARY="${RUN_DIR}/run_summary.json"
    LOG="${LOG_ROOT}/${RUN}.log"
    FAIL_MARK="${FAIL_ROOT}/${RUN}.failed"

    # ---------------------------------------------------------
    # Resume
    # ---------------------------------------------------------

    if [ -s "$SUMMARY" ]; then
        echo "[SKIP] $RUN"
        rm -f "$FAIL_MARK"
        return 0
    fi

    rm -f "$FAIL_MARK"
    rm -rf "$RUN_DIR"

    echo
    echo "================================================================"
    echo "[START] $RUN"
    echo "================================================================"
    echo "phase   = $PHASE"
    echo "model   = $MODEL_KEY"
    echo "dataset = $DATASET"
    echo "method  = $METHOD"
    echo "seed    = $SEED"
    echo "rank    = $RANK"
    echo "alpha   = $ALPHA"
    echo "================================================================"

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
        2>&1 \
        | sed -u "s/^/[${MODEL_KEY}|${DATASET}|${METHOD}|s${SEED}] /" \
        | tee "$LOG"

    STATUS=${PIPESTATUS[0]}

    if [ "$STATUS" -ne 0 ]; then

        echo "$STATUS" > "$FAIL_MARK"

        echo
        echo "================================================================"
        echo "[FAILED] $RUN"
        echo "exit code = $STATUS"
        echo "log       = $LOG"
        echo "================================================================"

        return "$STATUS"
    fi

    # A successful final run MUST contain this.
    if [ ! -s "$SUMMARY" ]; then

        echo "missing run_summary.json" > "$FAIL_MARK"

        echo
        echo "================================================================"
        echo "[FAILED] $RUN"
        echo "training exited 0 but run_summary.json is missing"
        echo "================================================================"

        return 98
    fi

    cleanup_checkpoints "$RUN_DIR"

    rm -f "$FAIL_MARK"

    echo
    echo "================================================================"
    echo "[DONE] $RUN"
    echo "================================================================"

    return 0
}


# =====================================================================
# Small foreground scheduler
# =====================================================================

ACTIVE=0
GROUP_FAILED=0
LIMIT=1


wait_one() {

    wait -n
    STATUS=$?

    ACTIVE=$((ACTIVE - 1))

    if [ "$STATUS" -ne 0 ]; then
        GROUP_FAILED=1
    fi
}


launch() {

    run_one "$@" &

    ACTIVE=$((ACTIVE + 1))

    if [ "$ACTIVE" -ge "$LIMIT" ]; then
        wait_one
    fi
}


drain_group() {

    while [ "$ACTIVE" -gt 0 ]; do
        wait_one
    done

    if [ "$GROUP_FAILED" -ne 0 ]; then

        echo
        echo "################################################################"
        echo "ONE OR MORE RUNS FAILED IN THIS GROUP"
        echo
        echo "Failure markers:"
        find "$FAIL_ROOT" \
            -maxdepth 1 \
            -type f \
            -printf '  %f\n' \
            | sort
        echo
        echo "Fix the problem and rerun the same master command."
        echo "Completed runs will be skipped."
        echo "################################################################"

        exit 1
    fi
}


begin_group() {

    LIMIT="$1"
    ACTIVE=0
    GROUP_FAILED=0

    echo
    echo "################################################################"
    echo "$2"
    echo "parallel jobs = $LIMIT"
    echo "################################################################"
}


# =====================================================================
# HEADER
# =====================================================================

echo
echo "======================================================================"
echo "TAXFACTO MASSIVE COMMON-7"
echo "======================================================================"
echo "planned runs       : $TOTAL"
echo "BERT parallelism   : $BERT_JOBS"
echo "DeBERTa parallelism: $DEBERTA_JOBS"
echo "rank parallelism   : $RANK_JOBS"
echo "mechanistic jobs   : $MECH_JOBS"
echo "======================================================================"


# =====================================================================
# PHASE A1
#
# InLegalBERT + LegalBERT
#
# 2 backbones
# x 5 datasets
# x 6 methods
# x 3 seeds
#
# = 180
# =====================================================================

begin_group \
    "$BERT_JOBS" \
    "PHASE A1 — InLegalBERT + LegalBERT main grid (180 runs)"

for MODEL_KEY in "${BERT_MODELS[@]}"; do

    for DATASET in "${DATASETS[@]}"; do

        for METHOD in "${METHODS[@]}"; do

            for SEED in "${SEEDS[@]}"; do

                launch \
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

drain_group


# =====================================================================
# PHASE A2
#
# DeBERTa
#
# 1 backbone
# x 5 datasets
# x 6 methods
# x 3 seeds
#
# = 90
# =====================================================================

begin_group \
    "$DEBERTA_JOBS" \
    "PHASE A2 — DeBERTa main grid (90 runs)"

for DATASET in "${DATASETS[@]}"; do

    for METHOD in "${METHODS[@]}"; do

        for SEED in "${SEEDS[@]}"; do

            launch \
                "main" \
                "deberta" \
                "$DATASET" \
                "$METHOD" \
                "$SEED" \
                8 \
                16

        done
    done
done

drain_group


# =====================================================================
# PHASE B
#
# Rank sensitivity on InLegalBERT.
#
# r=8 is already in main grid.
#
# 5 datasets
# x 3 PEFT methods
# x 4 additional ranks
#
# = 60
# =====================================================================

begin_group \
    "$RANK_JOBS" \
    "PHASE B — rank sensitivity (60 runs)"

for DATASET in "${DATASETS[@]}"; do

    for METHOD in "${RANK_METHODS[@]}"; do

        for RANK in "${RANKS[@]}"; do

            ALPHA=$((2 * RANK))

            launch \
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

drain_group


# =====================================================================
# PHASE C
#
# Expensive mechanistic ablation.
#
# 5 datasets
# x true_category_lora
#
# = 5
# =====================================================================

begin_group \
    "$MECH_JOBS" \
    "PHASE C — true category LoRA (5 runs)"

for DATASET in "${DATASETS[@]}"; do

    launch \
        "mechanistic" \
        "inlegalbert" \
        "$DATASET" \
        "true_category_lora" \
        42 \
        8 \
        16

done

drain_group


# =====================================================================
# FINAL AUDIT
# =====================================================================

FOUND=$(find "$RESULT_ROOT" \
    -mindepth 2 \
    -maxdepth 2 \
    -name run_summary.json \
    -type f \
    | wc -l)

FAILED=$(find "$FAIL_ROOT" \
    -maxdepth 1 \
    -type f \
    | wc -l)

echo
echo "======================================================================"
echo "MASSIVE RUN FINISHED"
echo "======================================================================"
echo "planned summaries : $TOTAL"
echo "summaries present : $FOUND"
echo "failure markers   : $FAILED"
echo "======================================================================"

if [ "$FAILED" -ne 0 ]; then
    exit 1
fi

if [ "$FOUND" -lt "$TOTAL" ]; then
    echo
    echo "WARNING:"
    echo "Fewer than $TOTAL final summaries exist."
    echo "This can happen if the result directory already contains a different"
    echo "experiment naming scheme or if a run was not scheduled as expected."
    exit 2
fi

echo
echo "ALL 335 EXPERIMENTS COMPLETE."
