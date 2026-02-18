import argparse
import json
import re
from pathlib import Path

import pandas as pd
from citeline.llm.citation_extraction import sentence_to_citations
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data_raw"

def argument_parser():
    """
    example usage: 
    
    # Build trivial & nontrivial datasets
    python dataset_builder.py

    # Build trivial dataset
    python dataset_builder.py --trivial
    """
    parser = argparse.ArgumentParser(description="Build dataset from research and review articles")
    parser.add_argument("--trivial", action="store_true", help="Build trivial dataset")
    parser.add_argument(
        "--research-path",
        type=str,
        default=str(RAW_DIR / "research.jsonl"),
        help="Path to the preprocessed research JSONL file.",
    )
    parser.add_argument(
        "--reviews-path",
        type=str,
        default=str(RAW_DIR / "reviews.jsonl"),
        help="Path to the preprocessed reviews JSONL file.",
    )
    parser.add_argument(
        "--nontrivial-out",
        type=str,
        default=str(RAW_DIR / "nontrivial_checked.jsonl"),
        help="Output path for nontrivial examples JSONL.",
    )
    parser.add_argument(
        "--trivial-out",
        type=str,
        default=str(RAW_DIR / "trivial_llm.jsonl"),
        help="Output path for trivial examples JSONL.",
    )
    parser.add_argument(
        "--progress-log",
        type=str,
        default=None,
        help="Optional path for the progress log JSON file.",
    )
    return parser.parse_args()


REVIEW_JOURNAL_BIBCODES = {
    "RvGeo",
    "SSRv.",
    "LRSP.",
    "NewAR",
    "ESRv.",
    "NRvEE",
    "P&SS.",
    "ARA&A",
    "A&ARv",
}


# Create a function to build a lookup index
def build_bibcode_index(reference_records):
    """Build an index mapping bibcodes to records for fast lookup"""
    # Pre-filter non-review journals
    filtered_records = {}
    for ref in reference_records:
        bibcode = ref.get("bibcode")
        if bibcode and bibcode[4:9] not in REVIEW_JOURNAL_BIBCODES:
            if bibcode not in filtered_records:
                filtered_records[bibcode] = []
            filtered_records[bibcode].append(ref)
    return filtered_records


# Cache regex patterns
# @lru_cache(maxsize=2048)
def bibcode_regex(author: str, year: str):
    """
    Given first author and year, return a regex pattern for the
    corresponding bibcode. Results are cached for performance.
    """
    initial = author[0]
    year = year[:4]  # cut off any letters at the end
    pattern = rf"^{year}.*{initial}$"
    return re.compile(pattern)


def bibcode_matches(inline_citation: tuple[str, str], references: list[str]) -> int:
    """
    Given an inline citation and a list of references, return the references
    h the inline citation's bibcode regex pattern
    """
    author, year = inline_citation
    if not author or not year:
        return []
    pattern = bibcode_regex(*inline_citation)
    return [s for s in references if pattern.match(s)]


def clean_query(sent_no_cit: str) -> str:
    """
    Remove [REF] tokens and empty parentheses from sentence.
    Also cleans up multiple consecutive whitespace.
    """
    # Remove [REF] tokens
    text = sent_no_cit.replace("[REF]", "")
    # Remove parentheses containing only whitespace
    text = re.sub(r'\(\s*\)', '', text)
    # Clean up multiple consecutive whitespace to single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    return text.strip()


# Modify examples_from_record to accept the index
def examples_from_record_with_index(record, bibcode_index):
    num_sentences = len(record["body_sentences"])
    return [
        example
        for i, sentence in enumerate(
            tqdm(
                record["body_sentences"],
                leave=False,
                desc=f"Processing {record['doi']} (# sentences: {num_sentences})",
            )
        )
        if (example := sentence_to_example_with_index(record, sentence, i, bibcode_index)) is not None
    ]


# Modified sentence_to_example function using the index
def sentence_to_example_with_index(record, sentence, index, bibcode_index):
    def citation_to_doi_and_bibcode(citation):
        """
        Takes a citation tuple (author, year) and returns a tuple of
        (doi, bibcode). It first grabs all the matching bibcodes from the 'reference' field
        of the record, checks which of those bibcodes correspond to records in the bibcode_index,
        then checks the remaining candidates for author name
        """
        matching_ref_bibcodes = bibcode_matches(citation, record["reference"])
        if len(matching_ref_bibcodes) == 0:
            return None

        remaining_candidates = []
        author_prefix = citation[0][:4].lower()  # Use the first 4 characters of the author name
        for bibcode in matching_ref_bibcodes:
            if not bibcode in bibcode_index:
                continue
            reference_authors = [name.lower() for name in bibcode_index[bibcode][0]["author"]]
            matching_authors = [name for name in reference_authors if name.startswith(author_prefix)]
            if matching_authors:
                print(matching_authors, end=", ")
                remaining_candidates.append(bibcode)

        if len(remaining_candidates) != 1:
            return None

        bib = remaining_candidates[0]
        doi = bibcode_index[bib][0]["doi"]

        if doi:
            return doi, bib
        return None

    # Remove inline citations from the sentence, skip if result is too short (chose 63 after some inspection)
    print(f"\033[2K\rWorking on sentence {index}: {sentence[:70]}...", end="")
    result = sentence_to_citations(sentence)

    # If the sentence was invalid or errors in processing, skip it
    if not result:
        return None
    citations, sent_no_cit = result

    # NOTE: originally we checked sentence length as a signal whether the sentence was usable or not;
    # however using the LLM to check sentence validity should sufficiently pass through meaningful sentences regardless of length.

    citation_dois, bibcodes = [], []
    print(f"Citations: {citations}", end=", ")

    # If ANY inline citation is not found, return None
    for citation in citations:
        citation_extraction = citation_to_doi_and_bibcode(citation)
        if not citation_extraction:
            print(f"Did not find a valid unique bibcode for citation: {citation}")
            print("\033[2K\r", end="")
            return None
        doi, bib = citation_extraction
        citation_dois.append(doi)
        bibcodes.append(bib)

    # Convert pubdate from "YYYY-MM-DD" string to int YYYYMMDD format
    pubdate_int = int(record["pubdate"].replace("-", ""))

    return {
        "source_doi": record["doi"],
        "sent_original": sentence,
        "sent_no_cit": sent_no_cit,
        "query": clean_query(sent_no_cit),
        "sent_idx": index,
        "citation_dois": citation_dois,
        "pubdate": pubdate_int,
        "resolved_bibcodes": bibcodes,
    }


def main():
    args = argument_parser()
    research_path = Path(args.research_path)
    reviews_path = Path(args.reviews_path)
    nontrivial_out = Path(args.nontrivial_out)
    trivial_out = Path(args.trivial_out)

    nontrivial_out.parent.mkdir(parents=True, exist_ok=True)
    trivial_out.parent.mkdir(parents=True, exist_ok=True)

    if args.trivial:
        print("Building trivial dataset only...")
    progress_log_path = (
        Path(args.progress_log)
        if args.progress_log
        else RAW_DIR / ("progress_trivial.json" if args.trivial else "progress.json")
    )
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)

    print("Starting dataset builder...loading data...")
    # Load data
    research = pd.read_json(research_path, lines=True)
    reviews = pd.read_json(reviews_path, lines=True)
    print(f"Loaded {len(research)} research records and {len(reviews)} review records.")

    # Convert DataFrames to lists of dictionaries
    research_dicts = research.to_dict("records")
    reviews_dicts = reviews.to_dict("records")

    # Build the index for fast lookup by bibcode
    print("Building bibcode index for faster lookups...", end="")
    bibcode_index = build_bibcode_index(research_dicts)
    print("done.")

    del research  # Free memory
    del reviews

    """
    0. Check log file to see where we left off
    1. Create a log file logging the index of the review records processed so far
    2. Any errors, write out to an error log
    """

    if not progress_log_path.exists():
        print("No progress log found.")
        progress = {"record_idx": 0, "sent_idx": 0}
        with progress_log_path.open("w", encoding="utf-8") as f:
            json.dump(progress, f)

    # Check log file to see where we left off
    with progress_log_path.open("r", encoding="utf-8") as f:
        progress = json.load(f)
    last_record_index = progress["record_idx"]
    last_sent_index = progress["sent_idx"]
    print(f"Starting from record: {last_record_index}, sentence: {last_sent_index}")

    reviews_dicts = reviews_dicts[last_record_index:]
    for record in tqdm(reviews_dicts, total=len(reviews_dicts), desc="Processing records"):
        # Skip papers with body text > 250,000 characters
        if len(record.get("body", "")) > 250000:
            print(f"\nSkipping {record['doi']} - body text too long ({len(record['body'])} chars)")
            progress["record_idx"] += 1
            progress["sent_idx"] = 0
            with progress_log_path.open("w", encoding="utf-8") as f:
                json.dump(progress, f)
            continue

        for i, sentence in enumerate(
            tqdm(
                record["body_sentences"][last_sent_index:],
                leave=False,
                desc=f"Processing {record['doi']} (# sentences: {len(record['body_sentences'])})",
            ),
            start=last_sent_index,
        ):

            example = sentence_to_example_with_index(record, sentence, i, bibcode_index)

            # Write results
            if example is None:
                pass
            elif len(example["citation_dois"]) > 0:
                with nontrivial_out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(example) + "\n")
            else:
                with trivial_out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(example) + "\n")

            # Update progress log (sentence level)
            progress["sent_idx"] += 1
            with progress_log_path.open("w", encoding="utf-8") as f:
                json.dump(progress, f)

        # Update progress log (record level)
        progress["record_idx"] += 1
        progress["sent_idx"] = 0  # Reset sentence index for the next record
        with progress_log_path.open("w", encoding="utf-8") as f:
            json.dump(progress, f)


if __name__ == "__main__":
    main()
    # import cProfile, pstats

    # profiler = cProfile.Profile()
    # profiler.enable()
    # main()  # Run your main function
    # profiler.disable()

    # stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
    # stats.print_stats(40)
