from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from interview_ai.audio import AudioAnalyzer
from interview_ai.io import read_any_text
from interview_ai.pipeline import InterviewPipeline
from interview_ai.schemas import ConversationState, InterviewTurnRecord
from interview_ai.tts import LocalTTSService


def _safe_json_loads(text: str) -> dict:
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


class TurnBasedConversationManager:
    def __init__(
        self,
        llm_pipeline: InterviewPipeline,
        audio_analyzer: AudioAnalyzer | None = None,
        tts_service: LocalTTSService | None = None,
    ) -> None:
        self.llm_pipeline = llm_pipeline
        self.audio_analyzer = audio_analyzer or AudioAnalyzer()
        self.tts_service = tts_service or LocalTTSService()

    def start_session(
        self,
        resume_file: str,
        jd_file: str,
        max_turns: int = 5,
        follow_up_budget: int = 2,
    ) -> ConversationState:
        questions_result = self.llm_pipeline.generate_questions(resume_file, jd_file)
        planned_questions = [item["question"] for item in questions_result.get("weak_label", [])]
        current_question = planned_questions[0] if planned_questions else "Introduce yourself and summarize your fit for this role."
        return ConversationState(
            session_id=datetime.utcnow().strftime("conversation-%Y%m%d-%H%M%S"),
            resume_file=resume_file,
            jd_file=jd_file,
            current_question=current_question,
            planned_questions=planned_questions[1:] if planned_questions else [],
            follow_up_budget=follow_up_budget,
            max_turns=max_turns,
        )

    def speak_current_question(self, state: ConversationState, output_file: str) -> str:
        return self.tts_service.synthesize(state.current_question, output_file)

    def process_answer_turn(
        self,
        state: ConversationState,
        answer_audio_file: str,
        rubric: str | None = None,
    ) -> ConversationState:
        audio = self.audio_analyzer.analyze(answer_audio_file)
        llm_eval = self.llm_pipeline.evaluate_answer(
            question=state.current_question,
            answer=audio.transcript,
            rubric=rubric or "Score based on relevance, communication, confidence, and technical clarity.",
            duration_seconds=audio.duration_seconds,
        )
        llm_scores = _safe_json_loads(llm_eval["model_output"])

        transcript_history = "\n".join(
            f"Q: {turn.question}\nA: {turn.transcript}" for turn in state.completed_turns
        )
        follow_up_result = self.llm_pipeline.generate_follow_up_question(
            current_question=state.current_question,
            answer=audio.transcript,
            transcript_history=transcript_history,
        )
        follow_up_payload = _safe_json_loads(follow_up_result["model_output"])

        next_action = "next_question"
        next_question = ""

        if (
            follow_up_payload.get("next_action") == "follow_up"
            and state.follow_up_budget > 0
            and follow_up_payload.get("follow_up_question")
        ):
            next_action = "follow_up"
            next_question = str(follow_up_payload["follow_up_question"]).strip()
            state.follow_up_budget -= 1
        elif state.planned_questions and len(state.completed_turns) + 1 < state.max_turns:
            next_question = state.planned_questions.pop(0)
            next_action = "next_question"
        else:
            next_question = ""
            next_action = "end_interview"

        turn_record = InterviewTurnRecord(
            turn_index=len(state.completed_turns) + 1,
            question=state.current_question,
            question_type="technical",
            answer_audio_file=answer_audio_file,
            transcript=audio.transcript,
            audio_metrics=audio.model_dump(),
            llm_evaluation={
                "raw_output": llm_eval["model_output"],
                "parsed": llm_scores,
                "follow_up_raw_output": follow_up_result["model_output"],
                "follow_up_parsed": follow_up_payload,
            },
            next_question=next_question,
            next_action=next_action,
        )
        state.completed_turns.append(turn_record)
        state.current_question = next_question
        return state

    def save_state(self, state: ConversationState, output_file: str) -> str:
        target = Path(output_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(state.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        return str(target)

    def summarize_state(self, state: ConversationState) -> dict:
        return {
            "session_id": state.session_id,
            "completed_turns": len(state.completed_turns),
            "current_question": state.current_question,
            "remaining_planned_questions": len(state.planned_questions),
            "jd_preview": read_any_text(state.jd_file)[:200],
        }
