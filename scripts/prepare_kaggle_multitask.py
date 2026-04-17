from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from interview_ai.dataset_adapters import load_dataturks_resume_records, load_interview_question_records
from interview_ai.io import write_jsonl


def discover_resume_file(dataset_dir: Path) -> Path:
    if dataset_dir.is_file():
        return dataset_dir
    matches = sorted(path for path in dataset_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".json", ".jsonl"})
    if not matches:
        raise FileNotFoundError(f"No resume dataset JSON file found under {dataset_dir}")
    return matches[0]


def discover_question_file(dataset_dir: Path) -> Path:
    if dataset_dir.is_file():
        return dataset_dir
    matches = sorted(path for path in dataset_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".csv", ".json", ".jsonl"})
    if not matches:
        raise FileNotFoundError(f"No interview question dataset file found under {dataset_dir}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare multitask SFT dataset from Kaggle resume and interview datasets.")
    parser.add_argument("--resume-dataset-dir", required=True)
    parser.add_argument("--question-dataset-dir", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--num-augments", type=int, default=2)
    args = parser.parse_args()

    resume_file = discover_resume_file(Path(args.resume_dataset_dir))
    question_file = discover_question_file(Path(args.question_dataset_dir))

    records = []
    records.extend(
        load_dataturks_resume_records(
            resume_file,
            augment=args.augment,
            num_augments=args.num_augments,
        )
    )
    records.extend(
        load_interview_question_records(
            question_file,
            augment=args.augment,
            num_augments=args.num_augments,
        )
    )

    write_jsonl(records, args.output_file)
    print(f"Wrote {len(records)} multitask records to {args.output_file}")
    print(f"Resume source: {resume_file}")
    print(f"Question source: {question_file}")
    print(f"Augmentation enabled: {args.augment}")


if __name__ == "__main__":
    main()
