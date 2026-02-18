from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PIPELINE_DIR = PROJECT_ROOT / "data_pipeline"
RAW_DIR = PROJECT_ROOT / "data_raw"


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the citation-worthiness data pipeline.")
    parser.add_argument(
        "--preprocess-infiles",
        nargs="+",
        default=[str(RAW_DIR / "research.jsonl")],
        help="Input files for preprocessing (Stage 1).",
    )
    parser.add_argument(
        "--preprocess-outfile",
        type=str,
        default=str(RAW_DIR / "reviews.jsonl"),
        help="Output JSONL from preprocessing (Stage 1).",
    )
    parser.add_argument(
        "--preprocess-deduplicate-from",
        type=str,
        default=None,
        help="Optional JSONL to deduplicate against in Stage 1.",
    )
    parser.add_argument(
        "--research-path",
        type=str,
        default=str(RAW_DIR / "research.jsonl"),
        help="Research JSONL for dataset builder (Stage 2).",
    )
    parser.add_argument(
        "--reviews-path",
        type=str,
        default=str(RAW_DIR / "reviews.jsonl"),
        help="Reviews JSONL for dataset builder (Stage 2).",
    )
    parser.add_argument(
        "--nontrivial-out",
        type=str,
        default=str(RAW_DIR / "nontrivial_checked.jsonl"),
        help="Nontrivial output JSONL (Stage 2).",
    )
    parser.add_argument(
        "--trivial-out",
        type=str,
        default=str(RAW_DIR / "trivial_llm.jsonl"),
        help="Trivial output JSONL (Stage 2).",
    )
    parser.add_argument(
        "--progress-log",
        type=str,
        default=None,
        help="Optional progress log path for dataset builder (Stage 2).",
    )
    parser.add_argument(
        "--trivial",
        action="store_true",
        help="Pass through to dataset_builder.py to build trivial dataset only.",
    )
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = argument_parser()
    python = sys.executable

    preprocess_cmd = [
        python,
        str(PIPELINE_DIR / "preprocessing.py"),
        "--infiles",
        *args.preprocess_infiles,
        "--outfile",
        args.preprocess_outfile,
    ]
    if args.preprocess_deduplicate_from:
        preprocess_cmd.extend(["--deduplicate-from", args.preprocess_deduplicate_from])

    dataset_builder_cmd = [
        python,
        str(PIPELINE_DIR / "dataset_builder.py"),
        "--research-path",
        args.research_path,
        "--reviews-path",
        args.reviews_path,
        "--nontrivial-out",
        args.nontrivial_out,
        "--trivial-out",
        args.trivial_out,
    ]
    if args.progress_log:
        dataset_builder_cmd.extend(["--progress-log", args.progress_log])
    if args.trivial:
        dataset_builder_cmd.append("--trivial")

    create_dual_cmd = [python, str(PIPELINE_DIR / "create_dual_datasets.py")]

    run_cmd(preprocess_cmd)
    run_cmd(dataset_builder_cmd)
    run_cmd(create_dual_cmd)


if __name__ == "__main__":
    main()
