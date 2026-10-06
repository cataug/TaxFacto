from pathlib import Path
from zipfile import ZipFile
import pandas as pd
import re
import unicodedata

ROOT = Path.home() / "TaxFacto"
RAW = ROOT / "data/raw/marro"
STD = ROOT / "data/standardized"
REPORT = ROOT / "reports/dataset_audit"

SRC = STD / "rhetorical_roles_all.parquet"

VALID_LABELS = {
    "FAC",
    "ARG",
    "RATIO",
    "STA",
    "PRE",
    "RPC",
    "RLC",
}


def clean_text(x):
    return re.sub(r"\s+", " ", str(x)).strip()


def norm_text(x):
    x = unicodedata.normalize(
        "NFKC",
        clean_text(x)
    ).casefold()

    return re.sub(r"\s+", " ", x).strip()


def aggressive_norm(x):
    x = unicodedata.normalize(
        "NFKC",
        clean_text(x)
    ).casefold()

    x = re.sub(
        r"[^\w]+",
        " ",
        x,
        flags=re.UNICODE
    )

    return re.sub(r"\s+", " ", x).strip()


archives = [
    ("marro_india", "train", RAW / "IN-train-set.zip"),
    ("marro_india", "test",  RAW / "IN-test-set.zip"),
    ("marro_uk",    "train", RAW / "UK-train-set.zip"),
    ("marro_uk",    "test",  RAW / "UK-test-set.zip"),
]

rows = []
bad_lines = []
archive_stats = []

for dataset, split, zip_path in archives:

    print()
    print("=" * 80)
    print(dataset, split, zip_path.name)
    print("=" * 80)

    if not zip_path.exists():
        print("MISSING:", zip_path)
        continue

    n_docs = 0
    n_rows = 0

    with ZipFile(zip_path) as z:

        names = sorted(
            n for n in z.namelist()
            if not n.endswith("/")
            and n.lower().endswith(".txt")
        )

        print("documents:", len(names))

        for name in names:
            n_docs += 1

            doc_id = Path(name).stem

            raw = z.read(name).decode(
                "utf-8",
                errors="replace"
            )

            for sid, line in enumerate(raw.splitlines()):
                line = line.strip()

                if not line:
                    continue

                # Expected inherited MARRO/DeepRhole format:
                # sentence<TAB>label
                if "\t" in line:
                    text, label = line.rsplit("\t", 1)

                else:
                    # Defensive fallback:
                    # "... sentence ... LABEL"
                    m = re.search(
                        r"\s+(FAC|ARG|RATIO|STA|PRE|RPC|RLC)\s*$",
                        line,
                        flags=re.I,
                    )

                    if not m:
                        bad_lines.append({
                            "archive": zip_path.name,
                            "file": name,
                            "line": sid,
                            "content": line[:500],
                        })
                        continue

                    text = line[:m.start()]
                    label = m.group(1)

                text = clean_text(text)
                label = clean_text(label).upper()

                if not text:
                    continue

                if label not in VALID_LABELS:
                    bad_lines.append({
                        "archive": zip_path.name,
                        "file": name,
                        "line": sid,
                        "content": line[:500],
                    })
                    continue

                rows.append({
                    "dataset": dataset,
                    "split": split,
                    "doc_id": doc_id,
                    "sentence_id": sid,
                    "text": text,
                    "label": label,
                    "norm": norm_text(text),
                    "norm_aggressive": aggressive_norm(text),
                })

                n_rows += 1

    archive_stats.append({
        "dataset": dataset,
        "split": split,
        "documents": n_docs,
        "rows": n_rows,
    })

    print("parsed rows:", f"{n_rows:,}")


marro = pd.DataFrame(rows)

if marro.empty:
    raise RuntimeError("MARRO parser extracted zero rows.")

print()
print("=" * 80)
print("MARRO SUMMARY")
print("=" * 80)

print(
    marro.groupby(
        ["dataset", "split"]
    ).size()
)

print()
print("LABEL COUNTS")

print(
    marro.groupby(
        ["dataset", "label"]
    ).size()
)

print()
print("Documents:")

print(
    marro.groupby("dataset")["doc_id"].nunique()
)

# Save MARRO separately.
marro.to_parquet(
    STD / "marro_all.parquet",
    index=False
)

pd.DataFrame(archive_stats).to_csv(
    REPORT / "marro_archive_stats.csv",
    index=False
)

pd.DataFrame(bad_lines).to_csv(
    REPORT / "marro_bad_lines.csv",
    index=False
)

print()
print("bad/unparsed lines:", len(bad_lines))

# ------------------------------------------------------------
# Merge into unified RR table.
# Remove any previous MARRO records first, making this rerunnable.
# ------------------------------------------------------------

base = pd.read_parquet(SRC)

base = base[
    ~base["dataset"].isin(
        ["marro", "marro_india", "marro_uk"]
    )
].copy()

# Align columns safely.
all_cols = sorted(
    set(base.columns) | set(marro.columns)
)

for c in all_cols:
    if c not in base.columns:
        base[c] = ""
    if c not in marro.columns:
        marro[c] = ""

merged = pd.concat(
    [
        base[all_cols],
        marro[all_cols],
    ],
    ignore_index=True,
)

merged.to_parquet(
    SRC,
    index=False
)

print()
print("=" * 80)
print("UNIFIED TABLE UPDATED")
print("=" * 80)

print(
    merged.groupby("dataset").size()
)

print()
print("TOTAL:", f"{len(merged):,}")
print("saved:", SRC)
