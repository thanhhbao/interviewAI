"""FastAPI backend for InterviewAI.

Endpoints are protected with Supabase JWTs — the frontend attaches
`Authorization: Bearer <token>` on every request via `apiFetch`.

Run:
    PYTHONPATH=src uvicorn interview_ai.api:app --reload --port 8000

Environment variables:
    OPENROUTER_API_KEY  — required for LLM/TTS/STT calls
    SUPABASE_URL        — required for JWT verification
    SUPABASE_JWT_SECRET — optional, for older HS256 Supabase projects
"""
from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
load_dotenv(Path(__file__).parents[2] / ".env")

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from interview_ai.auth import AuthError, get_user_id
from interview_ai.config import load_config
from interview_ai.conversation import TurnBasedConversationManager
from interview_ai.pipeline import InterviewPipeline
from interview_ai.providers import (
    OpenRouterLLMProvider,
    make_tts_provider,
)
from interview_ai.schemas import ConversationState

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="InterviewAI API", version="0.1.0")

FRONTEND_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # Add your production domain here, e.g. "https://interviewai.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config / providers ────────────────────────────────────────────────────────

_cfg = load_config(Path(__file__).parents[2] / "configs" / "models.yaml")

def _make_pipeline() -> InterviewPipeline:
    llm = OpenRouterLLMProvider(
        model=_cfg.models.planning.model,
    )
    return InterviewPipeline(provider=llm)

def _make_manager() -> TurnBasedConversationManager:
    pipeline = _make_pipeline()
    tts_cfg = _cfg.speech.tts
    extra = dict(tts_cfg.extra or {})
    tts = make_tts_provider(provider=tts_cfg.provider, model=tts_cfg.model, **extra)
    return TurnBasedConversationManager(llm_pipeline=pipeline, tts_provider=tts)

# ── In-memory session store (replace with Redis/DB for production) ─────────────
# Keyed by (user_id, session_id) so sessions are user-scoped.
_sessions: dict[tuple[str, str], ConversationState] = {}

# ── Auth dependency ────────────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)

async def current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> str:
    """Extract and verify the Supabase JWT; return the user_id (sub claim)."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return get_user_id(credentials.credentials)
    except AuthError as exc:
        raise HTTPException(
            status_code=exc.status,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )

UserID = Annotated[str, Depends(current_user)]

# ── Response models ────────────────────────────────────────────────────────────

class SessionStartResponse(BaseModel):
    session_id: str
    current_question: str
    candidate_name: str
    target_role: str
    total_planned_questions: int

class TurnResponse(BaseModel):
    session_id: str
    transcript: str
    next_action: str          # "follow_up" | "next_question" | "end_interview"
    current_question: str     # next question to ask (empty when ended)
    completed_turns: int

class SessionSummaryResponse(BaseModel):
    session_id: str
    candidate_name: str
    target_role: str
    completed_turns: int
    current_question: str
    remaining_planned_questions: int

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    """Unprotected liveness check."""
    return {"status": "ok"}


@app.get("/api/me")
async def me(user_id: UserID) -> dict:
    """Return the authenticated user's Supabase ID. Quick auth smoke test."""
    return {"user_id": user_id}


@app.post("/api/session/start", response_model=SessionStartResponse)
async def start_session(
    user_id: UserID,
    resume: UploadFile = File(..., description="Resume file (PDF, DOCX, or TXT)"),
    jd: UploadFile = File(..., description="Job description file (TXT or DOCX)"),
    max_turns: int = Form(5),
):
    """Start a new interview session.

    Extracts CandidateProfile + JobProfile via LLM, generates an InterviewPlan,
    returns the first question.
    """
    # Save uploads to temp files so the pipeline can read them by path
    suffix_resume = Path(resume.filename or "resume.pdf").suffix or ".pdf"
    suffix_jd = Path(jd.filename or "jd.txt").suffix or ".txt"

    with (
        tempfile.NamedTemporaryFile(delete=False, suffix=suffix_resume) as rf,
        tempfile.NamedTemporaryFile(delete=False, suffix=suffix_jd) as jf,
    ):
        rf.write(await resume.read())
        jf.write(await jd.read())
        resume_path = rf.name
        jd_path = jf.name

    try:
        manager = _make_manager()
        state = manager.start_session(resume_path, jd_path, max_turns=max_turns)
    finally:
        Path(resume_path).unlink(missing_ok=True)
        Path(jd_path).unlink(missing_ok=True)

    session_id = state.session_id
    _sessions[(user_id, session_id)] = state

    return SessionStartResponse(
        session_id=session_id,
        current_question=state.current_question,
        candidate_name=state.candidate_profile.name if state.candidate_profile else "",
        target_role=state.job_profile.role if state.job_profile else "",
        total_planned_questions=len(state.planned_questions) + 1,
    )


@app.post("/api/session/{session_id}/turn", response_model=TurnResponse)
async def process_turn(
    session_id: str,
    user_id: UserID,
    audio: UploadFile = File(..., description="Answer audio (WAV/MP3) or text transcript (.txt)"),
):
    """Process one answer turn and advance the session."""
    state = _sessions.get((user_id, session_id))
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    if not state.current_question:
        raise HTTPException(status_code=400, detail="Interview has already ended.")

    suffix = Path(audio.filename or "answer.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as af:
        af.write(await audio.read())
        audio_path = af.name

    try:
        manager = _make_manager()
        # Re-attach existing state instead of calling start_session
        state = manager.process_answer_turn(state, audio_path)
    finally:
        Path(audio_path).unlink(missing_ok=True)

    _sessions[(user_id, session_id)] = state
    last_turn = state.completed_turns[-1] if state.completed_turns else None

    return TurnResponse(
        session_id=session_id,
        transcript=last_turn.transcript if last_turn else "",
        next_action=last_turn.next_action if last_turn else "end_interview",
        current_question=state.current_question or "",
        completed_turns=len(state.completed_turns),
    )


@app.get("/api/session/{session_id}", response_model=SessionSummaryResponse)
async def get_session(session_id: str, user_id: UserID) -> SessionSummaryResponse:
    """Return the current state summary for a session."""
    state = _sessions.get((user_id, session_id))
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    manager = _make_manager()
    summary = manager.summarize_state(state)
    return SessionSummaryResponse(**summary)


@app.get("/api/sessions")
async def list_sessions(user_id: UserID) -> dict:
    """List all active session IDs for the current user."""
    ids = [sid for (uid, sid) in _sessions if uid == user_id]
    return {"session_ids": ids}
