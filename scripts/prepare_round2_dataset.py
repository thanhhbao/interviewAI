from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from interview_ai.dataset_adapters import (
    load_dataturks_resume_records,
    load_interview_behavior_vi_records,
    load_interview_question_records,
)
from interview_ai.io import write_jsonl


def discover_resume_file(dataset_dir: Path) -> Path:
    if dataset_dir.is_file():
        return dataset_dir
    matches = sorted(path for path in dataset_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".zip"})
    if not matches:
        raise FileNotFoundError(f"No resume dataset file found under {dataset_dir}")
    return matches[0]


def discover_question_file(dataset_dir: Path) -> Path:
    if dataset_dir.is_file():
        return dataset_dir
    matches = sorted(path for path in dataset_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".csv", ".json", ".jsonl", ".zip"})
    if not matches:
        raise FileNotFoundError(f"No interview question dataset file found under {dataset_dir}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare round-2 dataset with Vietnamese interview behavior data plus resume data.")
    parser.add_argument("--resume-dataset-dir", required=True)
    parser.add_argument("--question-dataset-dir", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--num-augments", type=int, default=2)
    parser.add_argument("--behavior-limit", type=int)
    args = parser.parse_args()

    resume_file = discover_resume_file(Path(args.resume_dataset_dir))
    question_file = discover_question_file(Path(args.question_dataset_dir))

    resume_records = load_dataturks_resume_records(
        resume_file,
        augment=args.augment,
        num_augments=args.num_augments,
    )
    question_records = load_interview_question_records(
        question_file,
        augment=args.augment,
        num_augments=args.num_augments,
    )
    behavior_records = load_interview_behavior_vi_records(
        question_file,
        limit=args.behavior_limit,
    )

    all_records = []
    all_records.extend(behavior_records)
    all_records.extend(resume_records)
    all_records.extend(question_records)

    write_jsonl(all_records, args.output_file)
    print(f"Wrote {len(all_records)} round-2 records to {args.output_file}")
    print(f"Resume records: {len(resume_records)}")
    print(f"Question records: {len(question_records)}")
    print(f"Vietnamese behavior records: {len(behavior_records)}")
    print(f"Resume source: {resume_file}")
    print(f"Question source: {question_file}")


if __name__ == "__main__":
    main()
