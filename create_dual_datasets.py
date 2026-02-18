"""Create paired window and abstract context datasets from raw jsonl files.

The script reads three inputs under data_raw/:
- reviews.jsonl: article metadata + sentence lists
- nontrivial_checked.jsonl: positive examples (label=1)
- trivial_llm.jsonl: negative examples (label=0)

Outputs two CSVs under data_proc/:
- dataset_window.csv  (local window context)
- dataset_abstract.csv (abstract-level context)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import spacy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data_raw"
PROC_DIR = PROJECT_ROOT / "data_proc"

NLP = spacy.load("en_core_web_sm")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a jsonl file, skipping blank lines."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse line in {path}: {line[:80]}") from exc


def load_reviews(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load reviews into a dictionary keyed by DOI for O(1) access."""
    reviews: Dict[str, Dict[str, Any]] = {}
    for record in iter_jsonl(path):
        doi = record.get("doi")
        if not isinstance(doi, str) or not doi.strip():
            continue
        sentences = record.get("body_sentences")
        if not isinstance(sentences, list):
            sentences = []
        abstract = record.get("abstract") or ""
        reviews[doi] = {
            "sentences": sentences,
            "abstract": abstract if isinstance(abstract, str) else "",
        }
    return reviews


# Precompile citation-cleaning regexes for speed and consistency
SQUARE_REF_PATTERN = re.compile(r"\[\s*(ref|cite)\s*\]", re.IGNORECASE)
SQUARE_NUM_PATTERN = re.compile(r"\[\s*\d+(?:\s*[-,]\s*\d+)*\s*\]")
PAREN_AUTHOR_PATTERN = re.compile(r"\(\s*[A-Za-z][^)]*?\d{4}[^)]*?\)")
EMPTY_PAREN_PATTERN = re.compile(r"\(\s*[^0-9A-Za-z]*\)")
SPACE_BEFORE_PUNCT_PATTERN = re.compile(r"\s+([,.;:])")


def rigorous_clean(text: str) -> str:
    """Remove citation artifacts, fix punctuation spacing, and normalize whitespace."""
    if not isinstance(text, str):
        return ""
    cleaned = text
    cleaned = SQUARE_REF_PATTERN.sub("", cleaned)
    cleaned = SQUARE_NUM_PATTERN.sub("", cleaned)
    cleaned = PAREN_AUTHOR_PATTERN.sub("", cleaned)
    cleaned = EMPTY_PAREN_PATTERN.sub("", cleaned)
    cleaned = SPACE_BEFORE_PUNCT_PATTERN.sub(r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def safe_int(value: Any) -> Optional[int]:
    """Convert value to int when possible; otherwise return None."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def fetch_sentence(review: Dict[str, Any], idx: Optional[int]) -> Optional[str]:
    """Return sentence at idx if valid; otherwise None."""
    sentences = review.get("sentences")
    if not isinstance(sentences, list):
        return None
    if idx is None or idx < 0 or idx >= len(sentences):
        return None
    sentence = sentences[idx]
    return sentence if isinstance(sentence, str) else None


def build_window_text(review: Dict[str, Any], target_text: str, idx: Optional[int]) -> str:
    """Construct ±2 sentence window around idx, replacing the center with target_text."""
    sentences = review.get("sentences")
    if not isinstance(sentences, list) or idx is None or idx < 0 or idx >= len(sentences):
        return rigorous_clean(target_text)

    start = max(0, idx - 2)
    end = min(len(sentences) - 1, idx + 2)
    window: List[str] = []
    for sent in sentences[start : end + 1]:
        if isinstance(sent, str) and sent.strip():
            window.append(rigorous_clean(sent))
    target_pos = idx - start
    if 0 <= target_pos < len(window):
        window[target_pos] = rigorous_clean(target_text)
    if not window:
        return rigorous_clean(target_text)
    return " ".join(window)


def build_abstract_text(review: Dict[str, Any], target_text: str) -> str:
    """Combine abstract with target sentence using [SEP]."""
    abstract = review.get("abstract")
    if isinstance(abstract, str) and abstract.strip():
        return f"{rigorous_clean(abstract)} [SEP] {rigorous_clean(target_text)}"
    return rigorous_clean(target_text)


def process_record(
    record: Dict[str, Any],
    label: int,
    reviews: Dict[str, Dict[str, Any]],
    window_rows: List[Dict[str, Any]],
    abstract_rows: List[Dict[str, Any]],
) -> None:
    doi = record.get("source_doi")
    if not isinstance(doi, str) or not doi.strip():
        return
    review = reviews.get(doi)
    if not review:
        return

    sent_idx = safe_int(record.get("sent_idx"))

    if label == 1:
        target_text = record.get("sent_no_cit") or ""
        if not isinstance(target_text, str):
            target_text = ""
        if not target_text:
            # Fallback to original sentence if cleaned variant missing.
            target_text = fetch_sentence(review, sent_idx) or record.get("sent_original") or ""
    else:
        target_text = fetch_sentence(review, sent_idx) or ""
        if not target_text:
            # Graceful fallback to provided text if index invalid.
            target_text = record.get("sent_no_cit") or record.get("sent_original") or ""

    target_text = rigorous_clean(target_text)

    if not target_text:
        return
    doc = NLP(target_text)
    if not doc:
        return
    if doc[0].pos_ in {"VERB", "AUX"}:
        return

    window_text = build_window_text(review, target_text, sent_idx)
    abstract_text = build_abstract_text(review, target_text)

    row = {"label": label, "source_doi": doi, "sent_idx": sent_idx}
    window_rows.append({"text": window_text, **row})
    abstract_rows.append({"text": abstract_text, **row})


def build_datasets() -> None:
    reviews_path = RAW_DIR / "reviews.jsonl"
    nontrivial_path = RAW_DIR / "nontrivial_checked.jsonl"
    trivial_path = RAW_DIR / "trivial_llm.jsonl"

    reviews = load_reviews(reviews_path)

    window_rows: List[Dict[str, Any]] = []
    abstract_rows: List[Dict[str, Any]] = []

    for record in iter_jsonl(nontrivial_path):
        process_record(record, label=1, reviews=reviews, window_rows=window_rows, abstract_rows=abstract_rows)

    for record in iter_jsonl(trivial_path):
        process_record(record, label=0, reviews=reviews, window_rows=window_rows, abstract_rows=abstract_rows)

    PROC_DIR.mkdir(parents=True, exist_ok=True)

    window_df = pd.DataFrame(window_rows, columns=["text", "label", "source_doi", "sent_idx"])
    abstract_df = pd.DataFrame(abstract_rows, columns=["text", "label", "source_doi", "sent_idx"])

    window_df.to_csv(PROC_DIR / "dataset_window.csv", index=False)
    abstract_df.to_csv(PROC_DIR / "dataset_abstract.csv", index=False)

    print(f"Window dataset rows: {len(window_df)}")
    print(f"Abstract dataset rows: {len(abstract_df)}")


if __name__ == "__main__":
    build_datasets()
