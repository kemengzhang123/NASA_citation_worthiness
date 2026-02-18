# Citation Worthiness Data Pipeline

This folder contains a three-stage pipeline for preparing datasets used in citation worthiness detection. The stages are designed to run sequentially and share data through the `data_raw/` and `data_proc/` directories at the project root.

## Directory Layout (Expected)

```
citation_cls/
├── run_pipeline.py
├── data_pipeline/
│   ├── preprocessing.py
│   ├── dataset_builder.py
│   ├── create_dual_datasets.py
│   └── README.md
├── data_raw/
│   ├── research.jsonl
│   ├── reviews.jsonl
│   ├── trivial_llm.jsonl
│   └── nontrivial_checked.jsonl
└── data_proc/
    ├── dataset_window.csv
    └── dataset_abstract.csv
```

## Stage 1: Preprocessing (`preprocessing.py`)

**Purpose**: Normalize raw records and create sentence-segmented reviews.

**Inputs**
- Default: `data_raw/research.jsonl`
- Override with: `--infiles`

**Outputs**
- Default: `data_raw/reviews.jsonl`
- Override with: `--outfile`

**Key operations**
- Drops records missing required fields.
- Normalizes `doi` and `keywords` fields.
- Segments `body` into sentences and merges very short sentences.

## Stage 2: Dataset Builder (`dataset_builder.py`)

**Purpose**: Extract citation signals from review sentences and produce trivial/nontrivial JSONL files.

**Inputs**
- Research records: `data_raw/research.jsonl` (`--research-path`)
- Review records: `data_raw/reviews.jsonl` (`--reviews-path`)

**Outputs**
- Nontrivial examples: `data_raw/nontrivial_checked.jsonl` (`--nontrivial-out`)
- Trivial examples: `data_raw/trivial_llm.jsonl` (`--trivial-out`)
- Progress log: `data_raw/progress.json` (or `progress_trivial.json` when `--trivial`)

**Key operations**
- Uses `sentence_to_citations` to detect inline citations.
- Resolves citations to bibcodes/DOIs using research references.
- Writes nontrivial examples when citations are found; trivial otherwise.

## Stage 3: Dual Dataset Creation (`create_dual_datasets.py`)

**Purpose**: Build final CSV datasets for modeling with different context strategies.

**Inputs**
- `data_raw/reviews.jsonl`
- `data_raw/nontrivial_checked.jsonl`
- `data_raw/trivial_llm.jsonl`

**Outputs**
- `data_proc/dataset_window.csv`
- `data_proc/dataset_abstract.csv`

**Key operations**
- Cleans citations from sentences.
- **Subject Integrity Filter**: if the cleaned sentence starts with a token tagged as `VERB` or `AUX`, the sample is dropped.
- Builds two views:
  - `window`: ±2 sentence window around the target.
  - `abstract`: abstract + target sentence.

## Running The Full Pipeline

From the project root (`citation_cls/`):

```bash
python run_pipeline.py
```

Common overrides:

```bash
python run_pipeline.py \
  --preprocess-infiles data_raw/research.jsonl \
  --preprocess-outfile data_raw/reviews.jsonl \
  --research-path data_raw/research.jsonl \
  --reviews-path data_raw/reviews.jsonl \
  --nontrivial-out data_raw/nontrivial_checked.jsonl \
  --trivial-out data_raw/trivial_llm.jsonl
```

## Notes
- `create_dual_datasets.py` requires spaCy and the `en_core_web_sm` model.
- The pipeline assumes the data directories live at the root of `citation_cls/`.
