#!/usr/bin/env python3
"""Start the InterviewAI FastAPI backend.

Usage:
    PYTHONPATH=src python3 scripts/run_api.py
    PYTHONPATH=src python3 scripts/run_api.py --port 8001 --reload
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run InterviewAI API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", default=True)
    parser.add_argument("--no-reload", dest="reload", action="store_false")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(
        "interview_ai.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(SRC)],
    )


if __name__ == "__main__":
    main()
