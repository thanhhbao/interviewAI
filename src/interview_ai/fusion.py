from __future__ import annotations

from interview_ai.schemas import AudioAnalysis, FusionScore, VisionAnalysis


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, round(value, 2)))


class FusionScorer:
    def score(
        self,
        llm_scores: dict,
        audio: AudioAnalysis,
        vision: VisionAnalysis,
    ) -> FusionScore:
        content_score = clamp01(float(llm_scores.get("overall_score", 0.0)))
        behavior_score = clamp01(
            (vision.attention_score * 0.35)
            + (vision.eye_contact_score * 0.35)
            + ((1.0 - vision.head_down_ratio) * 0.2)
            + (vision.face_presence_ratio * 0.1)
        )
        speaking_score = clamp01(
            (audio.fluency_score * 0.6)
            + ((1.0 - min(audio.filler_count / 10.0, 1.0)) * 0.2)
            + ((1.0 - audio.pause_ratio) * 0.2)
        )
        final_score = clamp01((content_score * 0.5) + (behavior_score * 0.25) + (speaking_score * 0.25))

        strengths = list(llm_scores.get("strengths", []))
        improvements = list(llm_scores.get("improvements", []))
        if vision.eye_contact_score >= 0.7:
            strengths.append("Maintains steady eye contact.")
        if audio.filler_count <= 2:
            strengths.append("Uses few filler words.")
        if vision.head_down_ratio > 0.4:
            improvements.append("Reduce looking down during answers.")
        if audio.pause_ratio > 0.3:
            improvements.append("Reduce long pauses between ideas.")

        return FusionScore(
            content_score=content_score,
            behavior_score=behavior_score,
            speaking_score=speaking_score,
            final_score=final_score,
            breakdown={
                "content_weight": 0.5,
                "behavior_weight": 0.25,
                "speaking_weight": 0.25,
            },
            strengths=strengths,
            improvements=improvements,
        )
