from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from models_ablation import (
    SharedLoRAWideHeadClassifier,
    RoleAdapterClassifier,
    TrueCategoryLoRAClassifier,
)


ROOT = Path.home() / "TaxFacto"
TASK_ROOT = ROOT / "data/tasks/common7"
RESULT_ROOT = ROOT / "results/stage1"

COMMON7 = [
    "ARG",
    "FAC",
    "PRE",
    "RATIO",
    "RLC",
    "RPC",
    "STA",
]

LABEL2ID = {x: i for i, x in enumerate(COMMON7)}
ID2LABEL = {i: x for x, i in LABEL2ID.items()}


# ============================================================
# Data
# ============================================================

def load_split_dataset(name, split, tokenizer, max_length):
    fn = TASK_ROOT / f"{name}.parquet"
    df = pd.read_parquet(fn)

    df = df[df["split_canonical"] == split].copy()

    bad = sorted(set(df["label"]) - set(COMMON7))
    if bad:
        raise RuntimeError(f"{name}: unexpected labels: {bad}")

    df["labels"] = df["label"].map(LABEL2ID).astype(int)

    keep = df[
        [
            "text",
            "labels",
            "doc_id",
            "sentence_id",
            "label",
        ]
    ].copy()

    ds = Dataset.from_pandas(
        keep,
        preserve_index=False,
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    ds = ds.map(
        tokenize,
        batched=True,
        desc=f"Tokenize {name}/{split}",
    )

    # Metadata is retained separately in `keep` for predictions/reports.
    # The Trainer dataset itself must contain only tensorizable fields.
    remove_cols = [
        c for c in [
            "text",
            "doc_id",
            "sentence_id",
            "label",
        ]
        if c in ds.column_names
    ]

    ds = ds.remove_columns(remove_cols)

    return ds, keep.reset_index(drop=True)


# ============================================================
# Metrics
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]

    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(
            labels,
            preds,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            labels,
            preds,
            average="weighted",
            zero_division=0,
        ),
    }


# ============================================================
# Category-conditioned low-rank classifier
# ============================================================

class CategoryLowRankClassifier(nn.Module):
    """
    Shared LoRA encoder +
    class-specific low-rank residual transformations.

    For class c:

        r_c(h) = B_c(A_c(h))
        logit_c = w_c^T (h + scale * r_c(h)) + b_c

    All class residuals are evaluated simultaneously.
    """

    def __init__(
        self,
        model_name,
        num_labels,
        rank=8,
        alpha=16,
        dropout=0.05,
    ):
        super().__init__()

        config = AutoConfig.from_pretrained(model_name)

        config.num_labels = num_labels
        config.id2label = ID2LABEL
        config.label2id = LABEL2ID

        base = AutoModel.from_pretrained(
            model_name,
            config=config,
        )

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=[
                "query",
                "value",
                "query_proj",
                "value_proj",
                "q_proj",
                "v_proj",
            ],
            bias="none",
        )

        self.encoder = get_peft_model(
            base,
            lora_cfg,
        )

        hidden = config.hidden_size

        self.num_labels = num_labels
        self.rank = rank
        self.scale = alpha / rank

        self.dropout = nn.Dropout(dropout)

        # Per-class low-rank factors.
        self.A = nn.Parameter(
            torch.empty(num_labels, hidden, rank)
        )
        self.B = nn.Parameter(
            torch.empty(num_labels, rank, hidden)
        )

        self.classifier_weight = nn.Parameter(
            torch.empty(num_labels, hidden)
        )
        self.classifier_bias = nn.Parameter(
            torch.zeros(num_labels)
        )

        nn.init.normal_(self.A, std=0.02)
        nn.init.zeros_(self.B)
        nn.init.normal_(self.classifier_weight, std=0.02)

        self.config = config

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }

        if token_type_ids is not None:
            model_kwargs["token_type_ids"] = token_type_ids

        out = self.encoder(**model_kwargs)

        # BERT-like encoders: CLS state.
        h = out.last_hidden_state[:, 0, :]
        h = self.dropout(h)

        # [batch, classes, rank]
        low = torch.einsum(
            "bh,chr->bcr",
            h,
            self.A,
        )

        # [batch, classes, hidden]
        residual = torch.einsum(
            "bcr,crh->bch",
            low,
            self.B,
        )

        class_h = (
            h.unsqueeze(1)
            + self.scale * residual
        )

        logits = torch.einsum(
            "bch,ch->bc",
            class_h,
            self.classifier_weight,
        )

        logits = logits + self.classifier_bias

        loss = None

        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits,
                labels,
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


# ============================================================
# Models
# ============================================================

def build_model(
    model_name,
    method,
    rank,
    alpha,
    dropout,
):
    common_kwargs = dict(
        num_labels=len(COMMON7),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    if method == "frozen":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **common_kwargs,
        )

        # Freeze encoder, train classifier only.
        for p in model.base_model.parameters():
            p.requires_grad = False

        return model

    if method == "full_ft":
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **common_kwargs,
        )

    if method == "shared_lora":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **common_kwargs,
        )

        cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=[
                "query",
                "value",
                "query_proj",
                "value_proj",
                "q_proj",
                "v_proj",
            ],
            bias="none",
        )

        return get_peft_model(
            model,
            cfg,
        )

    if method == "shared_lora_widehead":
        return SharedLoRAWideHeadClassifier(
            model_name=model_name,
            num_labels=len(COMMON7),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            head_width=118,
        )

    if method == "role_adapter":
        return RoleAdapterClassifier(
            model_name=model_name,
            num_labels=len(COMMON7),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    if method == "true_category_lora":
        return TrueCategoryLoRAClassifier(
            model_name=model_name,
            num_labels=len(COMMON7),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    if method == "category_lora":
        return CategoryLowRankClassifier(
            model_name=model_name,
            num_labels=len(COMMON7),
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    raise ValueError(method)


# ============================================================
# Trainable parameter report
# ============================================================

def parameter_stats(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return {
        "parameters_total": int(total),
        "parameters_trainable": int(trainable),
        "trainable_pct": 100.0 * trainable / total,
    }


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--dataset",
        required=True,
        choices=[
            "legaleval",
            "marro_india",
            "marro_uk",
            "iltur_cl",
            "iltur_it",
        ],
    )

    ap.add_argument(
        "--method",
        required=True,
        choices=[
            "frozen",
            "full_ft",
            "shared_lora",
            "shared_lora_widehead",
            "role_adapter",
            "true_category_lora",
            "category_lora",
        ],
    )

    ap.add_argument(
        "--model",
        default="law-ai/InLegalBERT",
    )

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--max_length", type=int, default=256)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_batch_size", type=int, default=64)

    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)

    ap.add_argument(
        "--run_tag",
        default=None,
        help="Unique experiment tag used in output directory names.",
    )

    ap.add_argument(
        "--result_root",
        default="results/final_common7",
        help="Relative to TaxFacto root unless absolute.",
    )

    ap.add_argument(
        "--keep_model",
        action="store_true",
        help="Keep checkpoint/best-model files after evaluation.",
    )

    ap.add_argument(
        "--skip_test",
        action="store_true",
        help="Smoke/debug mode: train and evaluate dev only; never touch test.",
    )

    args = ap.parse_args()

    set_seed(args.seed)

    base_run_name = (
        f"{args.dataset}__{args.method}"
        f"__seed{args.seed}"
    )

    if args.run_tag:
        run_name = f"{args.run_tag}__{base_run_name}"
    else:
        run_name = base_run_name

    result_root = Path(args.result_root)

    if not result_root.is_absolute():
        result_root = ROOT / result_root

    result_root.mkdir(parents=True, exist_ok=True)

    out_dir = result_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RUN:", run_name)
    print("MODEL:", args.model)
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )

    train_ds, train_meta = load_split_dataset(
        args.dataset,
        "train",
        tokenizer,
        args.max_length,
    )

    dev_ds, dev_meta = load_split_dataset(
        args.dataset,
        "dev",
        tokenizer,
        args.max_length,
    )

    if args.skip_test:
        test_ds = None
        test_meta = None
        print(
            "rows:",
            len(train_ds),
            len(dev_ds),
            "test=SKIPPED",
        )
    else:
        test_ds, test_meta = load_split_dataset(
            args.dataset,
            "test",
            tokenizer,
            args.max_length,
        )

        print(
            "rows:",
            len(train_ds),
            len(dev_ds),
            len(test_ds),
        )

    model = build_model(
        model_name=args.model,
        method=args.method,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
    )

    pstats = parameter_stats(model)

    print()
    print("PARAMETERS")
    print(json.dumps(pstats, indent=2))

    # Different methods need different learning rates.
    lr = {
        "frozen": 1e-3,
        "full_ft": 2e-5,
        "shared_lora": 2e-4,
        "shared_lora_widehead": 2e-4,
        "role_adapter": 2e-4,
        "true_category_lora": 2e-4,
        "category_lora": 2e-4,
    }[args.method]

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),

        num_train_epochs=args.epochs,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,

        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.10,

        lr_scheduler_type="linear",

        eval_strategy="epoch",
        save_strategy=("no" if args.skip_test else "epoch"),

        load_best_model_at_end=(not args.skip_test),
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        save_total_limit=1,

        bf16=True,
        fp16=False,
        tf32=True,

        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        logging_steps=50,

        report_to=[],

        seed=args.seed,

        remove_unused_columns=True,

        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()

    train_result = trainer.train()

    train_seconds = time.time() - t0

    if args.skip_test:
        dev_metrics = trainer.evaluate(dev_ds)

        eval_history = [
            x for x in trainer.state.log_history
            if "eval_macro_f1" in x
        ]

        if eval_history:
            best_row = max(
                eval_history,
                key=lambda x: x["eval_macro_f1"],
            )
            best_dev_macro_f1 = float(
                best_row["eval_macro_f1"]
            )
            best_dev_epoch = float(
                best_row.get("epoch", -1)
            )
        else:
            best_dev_macro_f1 = float(
                dev_metrics["eval_macro_f1"]
            )
            best_dev_epoch = float(args.epochs)

        smoke_summary = {
            "run_name": run_name,
            "dataset": args.dataset,
            "model": args.model,
            "method": args.method,
            "seed": args.seed,
            "epochs": args.epochs,
            **pstats,
            "train_seconds": float(train_seconds),
            "best_dev_macro_f1": best_dev_macro_f1,
            "best_dev_epoch": best_dev_epoch,
            **{
                k: float(v)
                for k, v in dev_metrics.items()
                if isinstance(v, (int, float))
            },
        }

        if torch.cuda.is_available():
            smoke_summary["peak_gpu_memory_gb"] = float(
                torch.cuda.max_memory_allocated() / (1024 ** 3)
            )

        with open(
            out_dir / "smoke_dev_summary.json",
            "w",
        ) as f:
            json.dump(
                smoke_summary,
                f,
                indent=2,
            )

        print()
        print("=" * 80)
        print("DEV-ONLY SMOKE RESULT")
        print("=" * 80)
        print(json.dumps(smoke_summary, indent=2))
        print()
        print("TEST SET WAS NOT EVALUATED.")

        return

    print()
    print("=" * 80)
    print("TEST")
    print("=" * 80)

    prediction = trainer.predict(test_ds)

    logits = prediction.predictions

    if isinstance(logits, tuple):
        logits = logits[0]

    true = prediction.label_ids
    pred = np.argmax(logits, axis=-1)

    acc = accuracy_score(true, pred)
    macro = f1_score(
        true,
        pred,
        average="macro",
        zero_division=0,
    )
    weighted = f1_score(
        true,
        pred,
        average="weighted",
        zero_division=0,
    )

    p_macro, r_macro, _, _ = precision_recall_fscore_support(
        true,
        pred,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        true,
        pred,
        labels=list(range(len(COMMON7))),
        target_names=COMMON7,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        true,
        pred,
        labels=list(range(len(COMMON7))),
    )

    test_metrics = {
        "accuracy": float(acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(macro),
        "weighted_f1": float(weighted),
    }

    gpu_peak_gb = None

    if torch.cuda.is_available():
        gpu_peak_gb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

    eval_history = [
        x for x in trainer.state.log_history
        if "eval_macro_f1" in x
    ]

    if eval_history:
        best_eval_row = max(
            eval_history,
            key=lambda x: x["eval_macro_f1"],
        )
        best_dev_macro_f1 = float(
            best_eval_row["eval_macro_f1"]
        )
        best_dev_epoch = float(
            best_eval_row.get("epoch", -1)
        )
    else:
        best_dev_macro_f1 = None
        best_dev_epoch = None

    summary = {
        "run_name": run_name,
        "dataset": args.dataset,
        "model": args.model,
        "method": args.method,
        "seed": args.seed,

        "epochs": args.epochs,
        "max_length": args.max_length,
        "batch_size": args.batch_size,

        "learning_rate": lr,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,

        "train_rows": len(train_ds),
        "dev_rows": len(dev_ds),
        "test_rows": len(test_ds),

        **pstats,

        "train_seconds": float(train_seconds),
        "peak_gpu_memory_gb": (
            float(gpu_peak_gb)
            if gpu_peak_gb is not None
            else None
        ),

        **test_metrics,
    }

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    probs = torch.softmax(
        torch.tensor(logits),
        dim=-1,
    ).numpy()

    pred_df = test_meta.copy()

    pred_df["y_true"] = true
    pred_df["y_pred"] = pred

    pred_df["label_true"] = [
        ID2LABEL[int(x)]
        for x in true
    ]

    pred_df["label_pred"] = [
        ID2LABEL[int(x)]
        for x in pred
    ]

    for i, label in ID2LABEL.items():
        pred_df[f"prob_{label}"] = probs[:, i]

    pred_df.to_csv(
        out_dir / "test_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    pd.DataFrame(
        cm,
        index=COMMON7,
        columns=COMMON7,
    ).to_csv(
        out_dir / "confusion_matrix.csv"
    )

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    with open(
        out_dir / "classification_report.json",
        "w",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    with open(
        out_dir / "run_summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    if args.keep_model:
        trainer.save_model(
            str(out_dir / "best_model")
        )

        tokenizer.save_pretrained(
            str(out_dir / "best_model")
        )
    else:
        shutil.rmtree(
            out_dir / "checkpoints",
            ignore_errors=True,
        )

    print()
    print("=" * 80)
    print("RESULT")
    print("=" * 80)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
