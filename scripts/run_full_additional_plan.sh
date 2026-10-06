#!/usr/bin/env bash

cd ~/TaxFacto || exit 1

PY=/home/tahiti/Malashin_Projects/.venv_a100/bin/python

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1


echo
echo "================================================================"
echo "STEP 1/5 — PREPARE TRANSFER + NATIVE TASKS"
echo "================================================================"

"$PY" scripts/prepare_additional_tasks.py

STATUS=$?

if [ "$STATUS" -ne 0 ]; then
    echo "TASK PREPARATION FAILED"
    exit "$STATUS"
fi


echo
echo "================================================================"
echo "STEP 2/5 — DOCUMENT-LEVEL BOOTSTRAP"
echo "================================================================"

mkdir -p results/additional/bootstrap

"$PY" scripts/bootstrap_main_predictions.py \
    | tee results/additional/bootstrap/console.txt

STATUS=${PIPESTATUS[0]}

if [ "$STATUS" -ne 0 ]; then
    echo "BOOTSTRAP FAILED"
    exit "$STATUS"
fi


echo
echo "================================================================"
echo "STEP 3/5 — TECHNICAL PREFLIGHT"
echo "================================================================"

rm -rf results/additional/preflight

mkdir -p results/additional/preflight


# ------------------------------------------------------------
# Transfer override smoke
# ------------------------------------------------------------

TAXFACTO_TASK_FILE="data/tasks/transfer/xfer__iltur_cl__to__iltur_it.parquet" \
"$PY" -u src/run_train_with_task.py \
    --dataset legaleval \
    --method role_adapter \
    --model /home/tahiti/TaxFacto/models/InLegalBERT \
    --seed 999 \
    --epochs 0.05 \
    --max_length 256 \
    --batch_size 32 \
    --eval_batch_size 64 \
    --rank 8 \
    --alpha 16 \
    --dropout 0.05 \
    --skip_test \
    --run_tag preflight-transfer \
    --result_root results/additional/preflight

STATUS=$?

if [ "$STATUS" -ne 0 ]; then
    echo "TRANSFER PREFLIGHT FAILED"
    exit "$STATUS"
fi


# ------------------------------------------------------------
# Native 13-class smoke
# This specifically verifies that the existing train.py
# correctly accepts the dynamically patched native taxonomy.
# ------------------------------------------------------------

TAXFACTO_TASK_FILE="data/tasks/native_eval/native__legaleval.parquet" \
"$PY" -u src/run_train_with_task.py \
    --dataset legaleval \
    --method role_adapter \
    --model /home/tahiti/TaxFacto/models/InLegalBERT \
    --seed 999 \
    --epochs 0.05 \
    --max_length 256 \
    --batch_size 32 \
    --eval_batch_size 64 \
    --rank 8 \
    --alpha 16 \
    --dropout 0.05 \
    --skip_test \
    --run_tag preflight-native13 \
    --result_root results/additional/preflight

STATUS=$?

if [ "$STATUS" -ne 0 ]; then
    echo
    echo "NATIVE-TAXONOMY PREFLIGHT FAILED."
    echo "No full native runs were started."
    exit "$STATUS"
fi


rm -rf results/additional/preflight

echo
echo "PREFLIGHT OK"


echo
echo "================================================================"
echo "STEP 4/5 — 195 ADDITIONAL TRAINING RUNS"
echo "================================================================"

./scripts/run_additional_experiments.sh

STATUS=$?

if [ "$STATUS" -ne 0 ]; then
    echo
    echo "TRAINING PHASE STOPPED."
    echo "Fix the failed run and rerun:"
    echo
    echo "  ./scripts/run_full_additional_plan.sh"
    echo
    echo "Completed runs will be skipped."
    exit "$STATUS"
fi


echo
echo "================================================================"
echo "STEP 5/5 — FINAL ADDITIONAL ANALYSIS"
echo "================================================================"

mkdir -p results/additional/analysis

"$PY" scripts/summarize_additional_plan.py \
    | tee results/additional/analysis/summary_console.txt

STATUS=${PIPESTATUS[0]}

if [ "$STATUS" -ne 0 ]; then
    echo "ADDITIONAL ANALYSIS FAILED"
    exit "$STATUS"
fi


echo
echo "================================================================"
echo "FULL ADDITIONAL PLAN COMPLETE"
echo "================================================================"
echo
echo "Existing main experiment : 335 runs"
echo "New transfer             : 168 runs"
echo "New native taxonomy      :  27 runs"
echo "------------------------------------"
echo "Total trained experiment : 530 runs"
echo
echo "Bootstrap:"
echo "results/additional/bootstrap/"
echo
echo "Transfer/native analysis:"
echo "results/additional/analysis/"
echo "================================================================"
