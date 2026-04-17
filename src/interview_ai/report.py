from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from interview_ai.schemas import InterviewSessionReport


class ReportGenerator:
    def write_json(self, report: InterviewSessionReport, output_dir: str) -> str:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        target = output_path / f"{report.session_id}.json"
        target.write_text(json.dumps(report.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        return str(target)

    def write_pdf(self, report: InterviewSessionReport, output_dir: str) -> str:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        target = output_path / f"{report.session_id}.pdf"
        pdf = canvas.Canvas(str(target), pagesize=A4)
        width, height = A4

        lines = [
            f"Interview Report: {report.session_id}",
            f"Candidate: {report.candidate_profile.get('name', 'Unknown')}",
            f"Final Score: {report.fusion_score.final_score}",
            f"Content Score: {report.fusion_score.content_score}",
            f"Behavior Score: {report.fusion_score.behavior_score}",
            f"Speaking Score: {report.fusion_score.speaking_score}",
            f"WPM: {report.audio_analysis.wpm}",
            f"Filler Count: {report.audio_analysis.filler_count}",
            f"Eye Contact: {report.vision_analysis.eye_contact_score}",
            f"Attention: {report.vision_analysis.attention_score}",
            "Strengths:",
        ]
        lines.extend(f"- {item}" for item in report.fusion_score.strengths[:5])
        lines.append("Improvements:")
        lines.extend(f"- {item}" for item in report.fusion_score.improvements[:5])

        y = height - 50
        for line in lines:
            pdf.drawString(40, y, line[:110])
            y -= 18
            if y < 60:
                pdf.showPage()
                y = height - 50

        pdf.save()
        return str(target)
