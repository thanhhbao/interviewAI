"""Vercel entrypoint for the FastAPI backend (the package lives under src/)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from interview_ai.api import app  # noqa: E402,F401
