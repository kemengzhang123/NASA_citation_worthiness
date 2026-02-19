from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm


# ----------------------------
# Config (same as your notebook)
# ----------------------------
PROJECT_ROOT = Path(".").resolve()
DATA_DIR = PROJECT_ROOT

DATASETS = {
    "dataset_window": DATA_DIR / "dataset_window.csv",
    "dataset_abstract": DATA_DIR / "dataset_abstract.csv",
}

MODEL_NAMES = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "adsabs/astroBERT",
    "allenai/specter2_base",
]

BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 3
MAX_LENGTH = 512
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(SEED)
np.random.seed(SEED)


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


class CitationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(label, dtype=torch.float32)
        return item


class CitationClassifier(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls).squeeze(-1)
        return logits


def train_one_epoch(model, loader, criterion, optimizer) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training", leave=False):
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(**inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def eval_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)

            logits = model(**inputs)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()

    probs = 1 / (1 + np.exp(-all_logits))
    preds = (probs >= 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, preds, average="binary", pos_label=1, zero_division=0
    )
    try:
        roc_auc = roc_auc_score(all_labels, probs)
    except ValueError:
        roc_auc = float("nan")

    return total_loss / max(1, len(loader)), precision, recall, f1, roc_auc


def main():
    for dataset_name, dataset_path in DATASETS.items():
        df = load_dataset(dataset_path)
        train_df, val_df = split_train_val(df, val_size=0.2, random_state=42)

        print(f"\n=== Dataset: {dataset_name} ===")
        print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

        for model_name in MODEL_NAMES:
            print(f"\n--- Model: {model_name} ---")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = CitationClassifier(model_name).to(device)

            train_dataset = CitationDataset(
                train_df["text"].tolist(),
                train_df["label"].tolist(),
                tokenizer,
                max_length=MAX_LENGTH,
            )
            val_dataset = CitationDataset(
                val_df["text"].tolist(),
                val_df["label"].tolist(),
                tokenizer,
                max_length=MAX_LENGTH,
            )

            # Note: if DataLoader workers cause issues on cluster, set num_workers=0
            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=2,
                pin_memory=torch.cuda.is_available(),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=2,
                pin_memory=torch.cuda.is_available(),
            )

            # CHANGE 1: no pos_weight
            criterion = nn.BCEWithLogitsLoss()
            optimizer = AdamW(model.parameters(), lr=LR)

            best_val_recall = -1.0
            best_epoch = -1
            safe_model_name = model_name.replace("/", "_")
            best_model_path = f"best_model_{dataset_name}_{safe_model_name}.pt"

            for epoch in range(1, EPOCHS + 1):
                print(f"\nEpoch {epoch}/{EPOCHS}")

                train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
                val_loss, precision, recall, f1, roc_auc = eval_one_epoch(model, val_loader, criterion)

                print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

                # CHANGE 2: save best by Recall
                if recall > best_val_recall:
                    best_val_recall = recall
                    best_epoch = epoch
                    torch.save(model.state_dict(), best_model_path)

            print(f"\nBest checkpoint by Recall saved to: {best_model_path}")
            print(f"Best epoch: {best_epoch} | Best Recall: {best_val_recall:.4f}")


if __name__ == "__main__":
    main()
