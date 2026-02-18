import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pysbd
from tqdm import tqdm

"""
Originally, data was provided in various JSON files representing papers from various fields or journals.
These records contained the following keys (except for a few missing certain keys):
  'bibcode', 'abstract', 'aff', 'author', 'bibstem', 'doctype', 'doi',
  'id', 'keyword', 'pubdate', 'title', 'read_count', 'reference', 'data',
  'citation_count', 'citation', 'body', 'dois', 'loaded_from'

To make the data uniform, we drop any records missing any of the REQUIRED_KEYS. In addition, this code
  - Extracts the first doi from each record and stores it in a new key 'doi' (the old 'doi' key is renamed to 'dois', and is a list).
  - For research data (the reference dataset): drop any duplicates
  - For review data (the query dataset): segment the body text into sentences and merge any short sentences (less than 60 characters).
The final output is written to two separate JSONL files: 'research.jsonl' and 'reviews.jsonl'.
"""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data_raw"

SEG = pysbd.Segmenter(language="en", clean=False)

# Records missing any of these keys are excluded from the dataset
# Although we may use 'keywords' eventually, it's missing from many records and not critical for the initial processing.
REQUIRED_KEYS = {
    "title",
    "body",
    "abstract",
    "doi",
    "reference",
    "bibcode",
}


def argument_parser():
    parser = argparse.ArgumentParser(description="Preprocess datasets.")

    parser.add_argument(
        "--infiles",
        nargs="+",
        type=str,
        default=[str(RAW_DIR / "research.jsonl")],
        help="Input file paths for the datasets to process. This should be a comma-separated list of JSON files.",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default=str(RAW_DIR / "reviews.jsonl"),
        help="Output file path for the processed dataset. This will be a JSONL file.",
    )
    parser.add_argument(
        "--deduplicate-from",
        type=str,
        default=None,
        help="Path to another JSONL file to deduplicate against. This is useful for removing records that are already present in another dataset.",
    )
    args = parser.parse_args()
    return args


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            file.seek(0)
            data = []
            for line in file:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))

    if isinstance(data, dict):
        data = [data]

    total_records = len(data)
    data = [d for d in data if REQUIRED_KEYS.issubset(d.keys())]
    complete_records = len(data)
    print(f"{path}: {complete_records}/{total_records} have all required keys")

    for record in data:
        record["title"] = record["title"][0]

        # Extract first DOI in list as 'doi'
        assert isinstance(
            record["doi"], list
        ), f"DOI expected to be a list, but it was {type(record['doi'])}, value: {record['doi']}"
        record["dois"] = record["doi"]
        record["doi"] = record["doi"][0]

        # Rename 'keyword' to 'keywords'
        if "keyword" in record:
            record["keywords"] = record.pop("keyword")
        else:
            record["keywords"] = None

        # Additional keys
        record["loaded_from"] = path
        record["body_sentences"] = []
    return data


def merge_short_sentences(sentences, threshold=60):
    """
    Returns a list of sentences where sentences below the threshold length
    are concatenated with the following sentence until they exceed the threshold.
    """
    merged_sentences = []
    buffer = ""

    for sentence in sentences:
        if len(buffer) + len(sentence) < threshold:
            buffer += sentence + " "  # Keep accumulating
        else:
            if buffer:
                merged_sentences.append(buffer.strip())  # Commit the buffer
            buffer = sentence  # Reset buffer to current sentence

    if buffer:  # Add any remaining buffer at the end
        merged_sentences.append(buffer.strip())

    return merged_sentences


def process_record(record):
    """
    This function processes a single record by segmenting the body text into sentences, then merges
    short sentences using the merge_short_sentences function. The final output is a record with a new
    key 'body_sentences' containing the segmented and merged sentences.
    """
    sentences = SEG.segment(record["body"])
    # If there are no sentences in the body, log the error and return the record with blank 'body_sentences' key
    if not sentences:
        print(f"Empty sentences for record: {record['doi']}")
        with open("empty_sentences.csv", "a") as file:
            file.write(f"{record['doi']},{record['title']}\n")
        return record

    # Typical case: we have sentences so merge the short ones
    record["body_sentences"] = merge_short_sentences(sentences)
    return record


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function processes review datasets and writes the output to a single JSON file.
    Each record is processed to segment the body into sentences and merge short sentences.
    """
    # Replace invalid days and months (12-00-00 to 12-01-01)
    df["pubdate"] = df["pubdate"].str.replace(r"-00", "-01", regex=True)
    records = df.to_dict("records")

    # Process records in parallel
    results = []
    max_workers = max(1, os.cpu_count() - 2)
    print(f"Processing records with {max_workers} workers")

    executor = ProcessPoolExecutor(max_workers=max_workers)
    try:
        futures = [executor.submit(process_record, r) for r in records]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing records"
        ):
            results.append(future.result())
        print("All tasks completed, initiating non-blocking shutdown...")
    finally:
        # Non-blocking shutdown
        executor.shutdown(wait=False)

    return pd.DataFrame(results)


def write_data(datasets, output_file: str, deduplicate_from=None):
    # Get all records
    records = [record for dataset in datasets for record in load_dataset(dataset)]

    # Drop duplicates within the current records list based on 'doi'
    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["doi"])

    # Drop any records with 'body' less than 5000 characters
    df = df[df["body"].str.len() >= 5000]

    # Drop any records that already have doi in the outfile, if it exists
    if os.path.exists(output_file):
        existing_df = pd.read_json(output_file, lines=True)
        existing_dois = set(existing_df["doi"])
        df = df[~df["doi"].isin(existing_dois)]

    # Drop any records that are also in the review dataset
    if deduplicate_from:
        other_dataset = pd.read_json(deduplicate_from, lines=True)
        other_dois = set(other_dataset["doi"])
        df = df[~df["doi"].isin(other_dois)]

    df = preprocess_data(df)
    df.to_json(output_file, orient="records", lines=True, mode="a")


def main():
    args = argument_parser()
    write_data(
        datasets=args.infiles,
        output_file=args.outfile,
        deduplicate_from=args.deduplicate_from,
    )


if __name__ == "__main__":
    main()
