from __future__ import annotations

import json
from datetime import datetime

from interview_ai.audio import AudioAnalyzer
from interview_ai.fusion import FusionScorer
from interview_ai.io import read_any_text
from interview_ai.parsers import parse_resume_weak
from interview_ai.pipeline import InterviewPipeline
from interview_ai.report import ReportGenerator
from interview_ai.schemas import InterviewSessionReport
from interview_ai.vision import VisionAnalyzer


class InterviewSessionRunner:
    def __init__(
        self,
        llm_pipeline: InterviewPipeline,
        audio_analyzer: AudioAnalyzer | None = None,
        vision_analyzer: VisionAnalyzer | None = None,
        fusion_scorer: FusionScorer | None = None,
        report_generator: ReportGenerator | None = None,
    ) -> None:
        self.llm_pipeline = llm_pipeline
        self.audio_analyzer = audio_analyzer or AudioAnalyzer()
        self.vision_analyzer = vision_analyzer or VisionAnalyzer()
        self.fusion_scorer = fusion_scorer or FusionScorer()
        self.report_generator = report_generator or ReportGenerator()

    def run(
        self,
        resume_file: str,
        jd_file: str,
        answer_audio_file: str,
        vision_file: str,
        report_dir: str,
        question: str | None = None,
        rubric: str | None = None,
    ) -> dict[str, str]:
        resume = parse_resume_weak(resume_file)
        questions_result = self.llm_pipeline.generate_questions(resume_file, jd_file)
        first_question = question or "Introduce yourself and summarize why you fit this role."
        if questions_result.get("weak_label"):
            first_question = questions_result["weak_label"][0]["question"]

        audio_analysis = self.audio_analyzer.analyze(answer_audio_file)
        if vision_file.lower().endswith(".json"):
            vision_analysis = self.vision_analyzer.analyze_precomputed(vision_file)
        else:
            vision_analysis = self.vision_analyzer.analyze_images([vision_file])

        llm_eval = self.llm_pipeline.evaluate_answer(
            question=first_question,
            answer=audio_analysis.transcript,
            rubric=rubric or "Score based on relevance, communication, confidence, and technical clarity.",
            duration_seconds=audio_analysis.duration_seconds,
        )

        try:
            llm_scores = json.loads(llm_eval["model_output"])
        except json.JSONDecodeError:
            llm_scores = {
                "overall_score": audio_analysis.fluency_score,
                "strengths": ["Fallback score from fluency metrics."],
                "improvements": ["Provide a stricter JSON evaluation output from the tuned model."],
            }

        fusion_score = self.fusion_scorer.score(llm_scores, audio_analysis, vision_analysis)
        session_id = datetime.utcnow().strftime("session-%Y%m%d-%H%M%S")
        report = InterviewSessionReport(
            session_id=session_id,
            candidate_profile=resume.model_dump(),
            questions=questions_result.get("weak_label", []),
            audio_analysis=audio_analysis,
            vision_analysis=vision_analysis,
            llm_evaluation=llm_eval,
            fusion_score=fusion_score,
            report_summary={
                "jd_preview": read_any_text(jd_file)[:300],
                "question_used": first_question,
            },
        )
        return {
            "json_report": self.report_generator.write_json(report, report_dir),
            "pdf_report": self.report_generator.write_pdf(report, report_dir),
        }
