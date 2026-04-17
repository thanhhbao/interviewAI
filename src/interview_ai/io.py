from __future__ import annotations

import json
from pathlib import Path


def read_text_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def read_pdf(path: str | Path) -> str:
    import fitz

    doc = fitz.open(path)
    try:
        return "\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()


def read_docx(path: str | Path) -> str:
    from docx import Document

    document = Document(path)
    return "\n".join(p.text for p in document.paragraphs if p.text.strip())


def read_any_text(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix == ".docx":
        return read_docx(path)
    if suffix in {".txt", ".md"}:
        return read_text_file(path)
    if suffix == ".json":
        return json.dumps(json.loads(read_text_file(path)), ensure_ascii=False, indent=2)
    raise ValueError(f"Unsupported file type: {path}")


def write_jsonl(records: list[dict], output_file: str | Path) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
