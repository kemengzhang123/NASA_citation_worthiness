"""Run baseline benchmarking across datasets and embedding models."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data_proc"


def batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"text", "label", "source_doi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip().ne("")]
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label", "source_doi"])
    df["label"] = df["label"].astype(int)
    df["source_doi"] = df["source_doi"].astype(str)
    return df.reset_index(drop=True)


def split_train_val(
    df: pd.DataFrame, val_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    indices = np.arange(len(df))
    groups = df["source_doi"].values
    train_idx, val_idx = next(splitter.split(indices, df["label"].values, groups))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def encode_sentence_transformer(
    model_name: str, texts: List[str], batch_size: int, device: str
) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)
    total = math.ceil(len(texts) / batch_size) if texts else 0
    embeddings: List[np.ndarray] = []
    for batch in tqdm(batched(texts, batch_size), total=total, desc=f"Embedding {model_name}"):
        batch_emb = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        embeddings.append(batch_emb)
    if not embeddings:
        return np.empty((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    return np.vstack(embeddings).astype(np.float32, copy=False)


def encode_astrobert(model_name: str, texts: List[str], batch_size: int, device: str) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    total = math.ceil(len(texts) / batch_size) if texts else 0
    embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(batched(texts, batch_size), total=total, desc=f"Embedding {model_name}"):
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            embeddings.append(cls_embeddings)

    if not embeddings:# maybe problem here
        hidden_size = model.config.hidden_size
        return np.empty((0, hidden_size), dtype=np.float32)
    return np.vstack(embeddings).astype(np.float32, copy=False)


def fit_and_eval(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> Dict[str, float]:
    clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def format_markdown_table(rows: List[Dict[str, Any]], headers: List[str]) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    table_rows = [[fmt(row.get(header, "")) for header in headers] for row in rows]
    widths = [max(len(header), *(len(row[i]) for row in table_rows)) for i, header in enumerate(headers)]
    header_line = "| " + " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    data_lines = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in table_rows
    ]
    return "\n".join([header_line, sep_line, *data_lines])


def run_benchmark() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = {
        "dataset_window.csv": DATA_DIR / "dataset_window.csv",
        "dataset_abstract.csv": DATA_DIR / "dataset_abstract.csv",
    }
    models = [
        {"name": "all-MiniLM-L6-v2", "hf_id": "sentence-transformers/all-MiniLM-L6-v2", "type": "st"},
        {"name": "specter2_base", "hf_id": "allenai/specter2_base", "type": "st"},
        {"name": "astroBERT", "hf_id": "adsabs/astroBERT", "type": "astro"},
    ]

    results: List[Dict[str, Any]] = []

    for dataset_name, dataset_path in datasets.items():
        df = load_dataset(dataset_path)
        train_df, val_df = split_train_val(df, val_size=0.2, random_state=42)

        train_texts = train_df["text"].tolist()
        val_texts = val_df["text"].tolist()
        y_train = train_df["label"].values
        y_val = val_df["label"].values

        for model_cfg in models:
            if model_cfg["type"] == "astro":
                X_train = encode_astrobert(model_cfg["hf_id"], train_texts, batch_size=16, device=device)
                X_val = encode_astrobert(model_cfg["hf_id"], val_texts, batch_size=16, device=device)
            else:
                X_train = encode_sentence_transformer(
                    model_cfg["hf_id"], train_texts, batch_size=32, device=device
                )
                X_val = encode_sentence_transformer(
                    model_cfg["hf_id"], val_texts, batch_size=32, device=device
                )

            scores = fit_and_eval(X_train, y_train, X_val, y_val)
            results.append(
                {
                    "dataset": dataset_name,
                    "model": model_cfg["name"],
                    "val_precision": scores["precision"],
                    "val_recall": scores["recall"],
                    "val_f1": scores["f1"],
                }
            )

    results.sort(key=lambda row: row["val_recall"], reverse=True)
    table = format_markdown_table(
        results, headers=["dataset", "model", "val_precision", "val_recall", "val_f1"]
    )
    print(table)

    output_path = DATA_DIR / "benchmark_results.md"
    output_path.write_text(table + "\n", encoding="utf-8")
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    run_benchmark()
