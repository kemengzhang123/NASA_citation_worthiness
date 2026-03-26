from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, set_seed


PROJECT_ROOT = Path(".").resolve()
DATA_DIR = PROJECT_ROOT
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

DATASETS = {
    "dataset_window": DATA_DIR / "dataset_window.csv",
    "dataset_abstract": DATA_DIR / "dataset_abstract.csv",
}

MODEL_NAMES = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "adsabs/astroBERT",
    "allenai/specter2_base",
]

MODEL_ALIASES = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "astrobert": "adsabs/astroBERT",
    "specter2": "allenai/specter2_base",
}

DEFAULT_MLP_HIDDEN_DIMS = [256, 64]
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 2e-5
DEFAULT_EPOCHS = 3
DEFAULT_MAX_LENGTH = 512
DEFAULT_DROPOUT = 0.2
DEFAULT_THRESHOLD = 0.5
DEFAULT_NUM_WORKERS = 0
DEFAULT_VAL_SIZE = 0.2
DEFAULT_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def sample_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def safe_float(value: Any) -> float:
    return float(value) if value is not None else float("nan")


def resolve_model_names(requested_models: Sequence[str]) -> List[str]:
    resolved: List[str] = []
    for requested in requested_models:
        model_name = MODEL_ALIASES.get(requested.lower(), requested)
        if model_name not in MODEL_NAMES:
            raise ValueError(
                f"Unknown model '{requested}'. Use one of: {sorted(MODEL_ALIASES)} "
                f"or a full model id from {MODEL_NAMES}"
            )
        if model_name not in resolved:
            resolved.append(model_name)
    return resolved


def load_dataset(path: Path, limit: int | None = None) -> pd.DataFrame:
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

    if "sent_idx" not in df.columns:
        df["sent_idx"] = np.arange(len(df))
    df["sent_idx"] = pd.to_numeric(df["sent_idx"], errors="coerce").fillna(-1).astype(int)

    if limit is not None:
        df = df.head(limit).copy()

    df = df.reset_index(drop=True)
    df["sample_id"] = [
        f"{doi}::{sent_idx}::{sample_hash(text)}"
        for doi, sent_idx, text in zip(df["source_doi"], df["sent_idx"], df["text"])
    ]
    return df


def split_train_val(
    df: pd.DataFrame, val_size: float = DEFAULT_VAL_SIZE, random_state: int = DEFAULT_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    indices = np.arange(len(df))
    groups = df["source_doi"].values
    train_idx, val_idx = next(splitter.split(indices, df["label"].values, groups))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


class CitationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        encoded = self.tokenizer(
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(row["label"], dtype=torch.float32)
        item["row_idx"] = torch.tensor(idx, dtype=torch.long)
        return item


class CitationMLPClassifier(nn.Module):
    def __init__(self, model_name: str, hidden_dims: Sequence[int], dropout: float):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        layers: List[nn.Module] = []
        input_dim = self.encoder.config.hidden_size
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, **encoder_inputs):
        outputs = self.encoder(**encoder_inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding).squeeze(-1)
        return logits


def build_loader(
    df: pd.DataFrame,
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = CitationDataset(df=df, tokenizer=tokenizer, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        inputs = {
            key: value.to(DEVICE)
            for key, value in batch.items()
            if key not in {"labels", "row_idx"}
        }
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(**inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    loss: float,
    threshold: float,
) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )
    accuracy = accuracy_score(labels, preds)

    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(labels, probs)
    except ValueError:
        pr_auc = float("nan")

    fpr = fp / (fp + tn) if (fp + tn) else float("nan")

    return {
        "loss": safe_float(loss),
        "precision": safe_float(precision),
        "recall": safe_float(recall),
        "f1": safe_float(f1),
        "accuracy": safe_float(accuracy),
        "roc_auc": safe_float(roc_auc),
        "pr_auc": safe_float(pr_auc),
        "threshold": safe_float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fp_count": int(fp),
        "fpr": safe_float(fpr),
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    metadata_df: pd.DataFrame,
    threshold: float,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_row_indices: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            inputs = {
                key: value.to(DEVICE)
                for key, value in batch.items()
                if key not in {"labels", "row_idx"}
            }
            labels = batch["labels"].to(DEVICE)
            row_idx = batch["row_idx"].cpu()

            logits = model(**inputs)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            all_probs.append(torch.sigmoid(logits).detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_row_indices.append(row_idx)

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    row_indices = torch.cat(all_row_indices).numpy().astype(int)
    preds = (probs >= threshold).astype(int)

    metrics = compute_metrics(
        labels=labels,
        probs=probs,
        preds=preds,
        loss=total_loss / max(1, len(loader)),
        threshold=threshold,
    )

    predictions = metadata_df.iloc[row_indices].copy()
    predictions["row_idx"] = row_indices
    predictions["true_label"] = labels
    predictions["pred_prob"] = probs
    predictions["pred_label"] = preds
    predictions["is_false_positive"] = (
        (predictions["true_label"] == 0) & (predictions["pred_label"] == 1)
    ).astype(int)
    predictions = predictions.sort_values("row_idx").reset_index(drop=True)

    return metrics, predictions


def checkpoint_is_better(
    current: Dict[str, float], best: Dict[str, float] | None, eps: float = 1e-12
) -> bool:
    if best is None:
        return True
    if current["recall"] > best["recall"] + eps:
        return True
    if abs(current["recall"] - best["recall"]) <= eps:
        if current["fpr"] < best["fpr"] - eps:
            return True
        if abs(current["fpr"] - best["fpr"]) <= eps:
            if current["precision"] > best["precision"] + eps:
                return True
    return False


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_loss_plot(history_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train loss")
    ax.plot(history_df["epoch"], history_df["val_loss"], marker="o", label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_validation_metrics_plot(history_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for column, label in [
        ("val_recall", "Val recall"),
        ("val_precision", "Val precision"),
        ("val_f1", "Val F1"),
        ("val_fpr", "Val FPR"),
    ]:
        ax.plot(history_df["epoch"], history_df[column], marker="o", label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.set_title("Validation Metrics by Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_confusion_matrix_artifacts(metrics: Dict[str, float], output_dir: Path) -> None:
    matrix = np.array(
        [
            [int(metrics["tn"]), int(metrics["fp"])],
            [int(metrics["fn"]), int(metrics["tp"])],
        ]
    )
    matrix_df = pd.DataFrame(
        matrix,
        index=["true_trivial_0", "true_nontrivial_1"],
        columns=["pred_trivial_0", "pred_nontrivial_1"],
    )
    matrix_df.to_csv(output_dir / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_title("Confusion Matrix")

    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)


def summarize_labels(df: pd.DataFrame) -> Dict[str, int]:
    counts = df["label"].value_counts().to_dict()
    return {"label_0": int(counts.get(0, 0)), "label_1": int(counts.get(1, 0))}


def build_overlap_artifacts(
    dataset_name: str,
    predictions_by_model: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    if not predictions_by_model:
        return

    model_names = list(predictions_by_model.keys())
    base_columns = ["sample_id", "source_doi", "sent_idx", "text", "true_label"]
    base_df = next(iter(predictions_by_model.values()))[base_columns].drop_duplicates("sample_id").copy()

    fp_sets: Dict[str, set[str]] = {}
    for model_name, predictions in predictions_by_model.items():
        flag_column = f"{model_name}_fp"
        flags = predictions[["sample_id", "is_false_positive"]].drop_duplicates("sample_id").rename(
            columns={"is_false_positive": flag_column}
        )
        base_df = base_df.merge(flags, on="sample_id", how="left")
        base_df[flag_column] = base_df[flag_column].fillna(0).astype(int)
        fp_sets[model_name] = set(base_df.loc[base_df[flag_column] == 1, "sample_id"])

    flag_columns = [f"{model_name}_fp" for model_name in model_names]
    base_df["fp_model_count"] = base_df[flag_columns].sum(axis=1)

    overlap_df = base_df[base_df["fp_model_count"] > 0].copy()
    overlap_df = overlap_df.sort_values(
        by=["fp_model_count", "source_doi", "sent_idx"], ascending=[False, True, True]
    ).reset_index(drop=True)
    overlap_df.to_csv(output_dir / f"{dataset_name}_false_positive_overlap_membership.csv", index=False)

    pairwise_summary: Dict[str, Dict[str, float]] = {}
    for idx, left_model in enumerate(model_names):
        for right_model in model_names[idx + 1 :]:
            left_set = fp_sets[left_model]
            right_set = fp_sets[right_model]
            intersection = left_set & right_set
            union = left_set | right_set
            key = f"{left_model}__{right_model}"
            pairwise_summary[key] = {
                "intersection_count": len(intersection),
                "union_count": len(union),
                "jaccard": (len(intersection) / len(union)) if union else 0.0,
                "overlap_over_smaller_set": (
                    len(intersection) / min(len(left_set), len(right_set))
                    if left_set and right_set
                    else 0.0
                ),
            }

    three_way_count = 0
    if len(model_names) >= 3:
        intersection = set.intersection(*(fp_sets[model_name] for model_name in model_names))
        three_way_count = len(intersection)

    summary = {
        "dataset": dataset_name,
        "model_false_positive_counts": {
            model_name: len(fp_sets[model_name]) for model_name in model_names
        },
        "pairwise_overlap": pairwise_summary,
        "three_way_overlap_count": three_way_count,
        "num_samples_with_any_false_positive": int(len(overlap_df)),
    }
    save_json(output_dir / f"{dataset_name}_false_positive_overlap_summary.json", summary)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune transformer encoders with an MLP head for citation-worthiness classification."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Base directory for run outputs.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name. Defaults to a timestamped name.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum token length.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT,
        help="Dropout applied in the MLP head.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Decision threshold for converting probabilities to labels.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=DEFAULT_VAL_SIZE,
        help="Validation split proportion.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS.keys()),
        default=list(DATASETS.keys()),
        help="Datasets to run.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_NAMES),
        help=(
            "Models to run. Accepts full model ids or aliases: "
            + ", ".join(sorted(MODEL_ALIASES.keys()))
        ),
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=None,
        help="Optional row limit for quick smoke tests.",
    )
    parser.add_argument(
        "--mlp-hidden-dims",
        type=int,
        nargs="+",
        default=DEFAULT_MLP_HIDDEN_DIMS,
        help="Hidden dimensions for the MLP head, for example: --mlp-hidden-dims 256 64",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    selected_model_names = resolve_model_names(args.models)
    selected_datasets = {name: DATASETS[name] for name in args.datasets}

    run_name = args.run_name or f"mlp_finetune_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = args.results_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "device": str(DEVICE),
        "datasets": {name: str(path) for name, path in selected_datasets.items()},
        "models": selected_model_names,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "max_length": args.max_length,
        "dropout": args.dropout,
        "threshold": args.threshold,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "val_size": args.val_size,
        "limit_per_dataset": args.limit_per_dataset,
        "mlp_hidden_dims": args.mlp_hidden_dims,
    }
    save_json(run_dir / "run_config.json", config)

    print(f"Using device: {DEVICE}")
    print(f"Writing outputs to: {run_dir}")

    summary_rows: List[Dict[str, Any]] = []

    for dataset_name, dataset_path in selected_datasets.items():
        df = load_dataset(dataset_path, limit=args.limit_per_dataset)
        train_df, val_df = split_train_val(df, val_size=args.val_size, random_state=args.seed)

        dataset_dir = run_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            dataset_dir / "dataset_split_summary.json",
            {
                "dataset_name": dataset_name,
                "dataset_path": str(dataset_path),
                "full_size": int(len(df)),
                "train_size": int(len(train_df)),
                "val_size": int(len(val_df)),
                "full_label_counts": summarize_labels(df),
                "train_label_counts": summarize_labels(train_df),
                "val_label_counts": summarize_labels(val_df),
            },
        )

        print(f"\n=== Dataset: {dataset_name} ===")
        print(
            f"Train size: {len(train_df)} | Val size: {len(val_df)} | "
            f"Train labels: {summarize_labels(train_df)} | Val labels: {summarize_labels(val_df)}"
        )

        predictions_by_model: Dict[str, pd.DataFrame] = {}

        for model_name in selected_model_names:
            model_slug = slugify(model_name)
            model_dir = dataset_dir / model_slug
            model_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n--- Model: {model_name} ---")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = CitationMLPClassifier(
                model_name=model_name,
                hidden_dims=args.mlp_hidden_dims,
                dropout=args.dropout,
            ).to(DEVICE)

            train_loader = build_loader(
                df=train_df,
                tokenizer=tokenizer,
                max_length=args.max_length,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )
            val_loader = build_loader(
                df=val_df,
                tokenizer=tokenizer,
                max_length=args.max_length,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )

            criterion = nn.BCEWithLogitsLoss()
            optimizer = AdamW(model.parameters(), lr=args.lr)

            history_rows: List[Dict[str, Any]] = []
            best_metrics: Dict[str, float] | None = None
            best_epoch = -1
            checkpoint_path = model_dir / "best_checkpoint.pt"

            for epoch in range(1, args.epochs + 1):
                print(f"Epoch {epoch}/{args.epochs}")

                train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
                val_metrics, _ = evaluate_model(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    metadata_df=val_df,
                    threshold=args.threshold,
                )

                history_row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_f1": val_metrics["f1"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_roc_auc": val_metrics["roc_auc"],
                    "val_pr_auc": val_metrics["pr_auc"],
                    "val_fp_count": val_metrics["fp_count"],
                    "val_fpr": val_metrics["fpr"],
                    "val_tn": val_metrics["tn"],
                    "val_fp": val_metrics["fp"],
                    "val_fn": val_metrics["fn"],
                    "val_tp": val_metrics["tp"],
                }
                history_rows.append(history_row)

                print(
                    " | ".join(
                        [
                            f"Train Loss: {train_loss:.4f}",
                            f"Val Loss: {val_metrics['loss']:.4f}",
                            f"Recall: {val_metrics['recall']:.4f}",
                            f"Precision: {val_metrics['precision']:.4f}",
                            f"F1: {val_metrics['f1']:.4f}",
                            f"FPR: {val_metrics['fpr']:.4f}",
                            f"ROC-AUC: {val_metrics['roc_auc']:.4f}",
                            f"PR-AUC: {val_metrics['pr_auc']:.4f}",
                        ]
                    )
                )

                if checkpoint_is_better(val_metrics, best_metrics):
                    best_metrics = dict(val_metrics)
                    best_epoch = epoch
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_name": model_name,
                            "model_state_dict": model.state_dict(),
                            "best_metrics": best_metrics,
                            "mlp_hidden_dims": args.mlp_hidden_dims,
                            "dropout": args.dropout,
                            "threshold": args.threshold,
                        },
                        checkpoint_path,
                    )

            if best_metrics is None:
                raise RuntimeError(f"No checkpoint was saved for {dataset_name} / {model_name}")

            history_df = pd.DataFrame(history_rows)
            history_df.to_csv(model_dir / "history.csv", index=False)
            save_loss_plot(history_df, model_dir / "loss_curve.png")
            save_validation_metrics_plot(history_df, model_dir / "validation_metrics.png")

            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])

            final_metrics, final_predictions = evaluate_model(
                model=model,
                loader=val_loader,
                criterion=criterion,
                metadata_df=val_df,
                threshold=args.threshold,
            )

            final_predictions.to_csv(model_dir / "val_predictions.csv", index=False)
            save_confusion_matrix_artifacts(final_metrics, model_dir)
            save_json(
                model_dir / "metrics.json",
                {
                    "dataset": dataset_name,
                    "model_name": model_name,
                    "model_slug": model_slug,
                    "best_epoch": best_epoch,
                    "selection_metric": "recall",
                    "selection_tie_breakers": ["lower_fpr", "higher_precision"],
                    "metrics": final_metrics,
                },
            )

            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "model_name": model_name,
                    "model_slug": model_slug,
                    "best_epoch": best_epoch,
                    **final_metrics,
                }
            )
            predictions_by_model[model_slug] = final_predictions

        overlap_dir = dataset_dir / "overlap"
        overlap_dir.mkdir(parents=True, exist_ok=True)
        build_overlap_artifacts(
            dataset_name=dataset_name,
            predictions_by_model=predictions_by_model,
            output_dir=overlap_dir,
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["dataset", "recall", "fpr", "precision"],
        ascending=[True, False, True, False],
    )
    summary_df.to_csv(run_dir / "summary_metrics.csv", index=False)

    print(f"\nFinished. Summary saved to: {run_dir / 'summary_metrics.csv'}")


if __name__ == "__main__":
    main()
