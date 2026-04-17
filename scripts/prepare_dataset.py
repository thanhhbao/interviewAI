from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from interview_ai.io import read_any_text, write_jsonl
from interview_ai.parsers import (
    generate_questions_weak,
    load_answer_training_examples,
    match_resume_to_jd,
    parse_resume_weak,
)
from interview_ai.prompts import (
    SYSTEM_PROMPT,
    build_answer_evaluation_prompt,
    build_question_generation_prompt,
    build_resume_extract_prompt,
    build_resume_optimize_prompt,
)
from interview_ai.schemas import ChatMessage, SFTRecord


def build_chat_record(task: str, user_prompt: str, assistant_content: str, meta: dict | None = None) -> dict:
    record = SFTRecord(
        task=task,
        messages=[
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
            ChatMessage(role="assistant", content=assistant_content),
        ],
        meta=meta or {},
    )
    return record.model_dump()


def pair_resume_and_jd(resume_files: list[Path], jd_files: list[Path]) -> list[tuple[Path, Path]]:
    if not jd_files:
        return []
    pairs = []
    for index, resume_file in enumerate(sorted(resume_files)):
        pairs.append((resume_file, jd_files[index % len(jd_files)]))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT dataset for interview AI tasks.")
    parser.add_argument("--resume-dir", required=True)
    parser.add_argument("--jd-dir", required=True)
    parser.add_argument("--answer-file")
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    resume_dir = Path(args.resume_dir)
    jd_dir = Path(args.jd_dir)

    resume_files = [path for path in resume_dir.iterdir() if path.suffix.lower() in {".pdf", ".docx", ".txt", ".md"}]
    jd_files = [path for path in jd_dir.iterdir() if path.suffix.lower() in {".txt", ".md", ".json"}]

    records: list[dict] = []
    for resume_file, jd_file in pair_resume_and_jd(resume_files, jd_files):
        resume = parse_resume_weak(resume_file)
        jd_text = read_any_text(jd_file)

        extract_prompt = build_resume_extract_prompt(read_any_text(resume_file))
        records.append(
            build_chat_record(
                task="resume_extract",
                user_prompt=extract_prompt,
                assistant_content=json.dumps(resume.model_dump(), ensure_ascii=False, indent=2),
                meta={"resume_file": resume_file.name},
            )
        )

        optimization = match_resume_to_jd(resume, jd_text)
        optimize_prompt = build_resume_optimize_prompt(
            json.dumps(resume.model_dump(), ensure_ascii=False, indent=2),
            jd_text,
        )
        records.append(
            build_chat_record(
                task="resume_optimize",
                user_prompt=optimize_prompt,
                assistant_content=json.dumps(optimization.model_dump(), ensure_ascii=False, indent=2),
                meta={"resume_file": resume_file.name, "jd_file": jd_file.name},
            )
        )

        questions = generate_questions_weak(resume, jd_text)
        question_prompt = build_question_generation_prompt(
            json.dumps(resume.model_dump(), ensure_ascii=False, indent=2),
            jd_text,
        )
        records.append(
            build_chat_record(
                task="question_generation",
                user_prompt=question_prompt,
                assistant_content=json.dumps([item.model_dump() for item in questions], ensure_ascii=False, indent=2),
                meta={"resume_file": resume_file.name, "jd_file": jd_file.name},
            )
        )

    for answer_example in load_answer_training_examples(args.answer_file):
        eval_prompt = build_answer_evaluation_prompt(
            answer_example["question"],
            answer_example["answer"],
            answer_example["rubric"],
        )
        records.append(
            build_chat_record(
                task="answer_evaluation",
                user_prompt=eval_prompt,
                assistant_content=json.dumps(answer_example["target"], ensure_ascii=False, indent=2),
                meta={"source": answer_example.get("source", "manual_annotation")},
            )
        )

    write_jsonl(records, args.output_file)
    print(f"Wrote {len(records)} records to {args.output_file}")


if __name__ == "__main__":
    main()
