"""Prompt builders for every LLM task in the interview pipeline.

All Vietnamese text uses proper Unicode diacritics.
Keys in JSON output stay in English for schema compatibility.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "Bạn là trợ lý AI chuyên về phỏng vấn cho người dùng Việt Nam. "
    "Khi được yêu cầu trả về JSON, chỉ trả về JSON hợp lệ, súc tích, không có markdown, không giải thích thêm. "
    "Khi tạo câu hỏi phỏng vấn, câu hỏi tiếp theo, hoặc đánh giá, hãy viết nội dung bằng tiếng Việt tự nhiên, chuyên nghiệp."
)


# ── Resume extraction ─────────────────────────────────────────────────────────

def build_resume_extract_prompt(resume_text: str) -> str:
    return (
        "Trích xuất thông tin từ CV sau đây thành JSON với cấu trúc chính xác:\n\n"
        "{\n"
        '  "name": "",\n'
        '  "email": "",\n'
        '  "phone": "",\n'
        '  "summary": "",\n'
        '  "total_years_experience": null,\n'
        '  "skills": [],\n'
        '  "skills_with_evidence": [\n'
        '    {"name": "FastAPI", "evidence": ["Xây dựng API phục vụ 20k request/ngày"], "inferred": false}\n'
        "  ],\n"
        '  "experience": [\n'
        '    {\n'
        '      "company": "",\n'
        '      "role": "",\n'
        '      "duration": "",\n'
        '      "highlights": [],\n'
        '      "technologies": [],\n'
        '      "responsibilities": [],\n'
        '      "achievements": [],\n'
        '      "metrics": []\n'
        "    }\n"
        "  ],\n"
        '  "education": [{"institution": "", "degree": "", "year": ""}],\n'
        '  "projects": []\n'
        "}\n\n"
        "Chỉ bao gồm thông tin CÓ THỰC trong CV. Đánh dấu inferred=true nếu thông tin được suy luận, không ghi rõ.\n"
        "Không bịa đặt thông tin không có trong CV.\n\n"
        f"CV:\n{resume_text.strip()}"
    )


# ── JD extraction ─────────────────────────────────────────────────────────────

def build_jd_extract_prompt(jd_text: str) -> str:
    return (
        "Trích xuất thông tin từ Job Description sau thành JSON với cấu trúc:\n\n"
        "{\n"
        '  "role": "",\n'
        '  "seniority": "",\n'
        '  "required_skills": [],\n'
        '  "preferred_skills": [],\n'
        '  "responsibilities": [],\n'
        '  "competencies": [],\n'
        '  "domain": "",\n'
        '  "important_requirements": []\n'
        "}\n\n"
        "Chỉ bao gồm thông tin THỰC SỰ có trong JD. Không suy luận thêm.\n\n"
        f"Job Description:\n{jd_text.strip()}"
    )


# ── Interview planning ────────────────────────────────────────────────────────

def build_interview_plan_prompt(candidate_json: str, job_json: str) -> str:
    return (
        "Dựa trên hồ sơ ứng viên và yêu cầu vị trí, hãy tạo kế hoạch phỏng vấn gồm đúng 5 câu hỏi bằng tiếng Việt.\n\n"
        "Yêu cầu:\n"
        "- Mỗi câu hỏi phải gắn với bằng chứng cụ thể trong CV, yêu cầu quan trọng của JD, hoặc điểm không khớp giữa CV và JD.\n"
        "- Ưu tiên câu hỏi tình huống, kiểm tra chiều sâu kỹ thuật, trade-off, bottleneck, quyết định thiết kế.\n"
        "- Tránh câu hỏi chung chung khi đã có bằng chứng cụ thể.\n\n"
        "Trả về JSON array, mỗi phần tử có dạng:\n"
        "[\n"
        "  {\n"
        '    "question_id": "q1",\n'
        '    "type": "technical",\n'
        '    "skill_target": "PostgreSQL",\n'
        '    "difficulty": "hard",\n'
        '    "question": "Câu hỏi cụ thể bằng tiếng Việt...",\n'
        '    "expected_keywords": ["index", "query plan", "EXPLAIN ANALYZE", "connection pool"]\n'
        "  }\n"
        "]\n\n"
        "Các trường type hợp lệ: intro, technical, behavioral, follow_up.\n"
        "Các trường difficulty hợp lệ: easy, medium, hard.\n"
        "Chỉ trả về JSON array, không markdown, không giải thích.\n\n"
        f"Hồ sơ ứng viên:\n{candidate_json}\n\n"
        f"Vị trí tuyển dụng:\n{job_json}"
    )


# ── Question generation (legacy, without separate JD/candidate objects) ───────

def build_question_generation_prompt(resume_json: str, jd_text: str) -> str:
    return (
        "Hãy đóng vai một người phỏng vấn kỹ thuật khó tính. "
        "Dựa vào Resume JSON và Job Description, hãy tạo chính xác 5 câu hỏi phỏng vấn bằng tiếng Việt, "
        "sát với kinh nghiệm thực tế trong CV và sát với yêu cầu trong JD. "
        "Ưu tiên câu hỏi tình huống, câu hỏi đào sâu, câu hỏi kiểm tra tính trung thực, trade-off, bottleneck, tối ưu, và quyết định kỹ thuật. "
        "Không đặt câu hỏi quá chung chung, không đặt câu hỏi kiểu giao bài tập, không đặt câu hỏi không liên quan đến kỹ năng và kinh nghiệm trong CV. "
        "Mỗi expected_keywords phải gồm ít nhất 5 từ khóa kỹ thuật cụ thể, chuyên sâu, có thể dùng để chấm câu trả lời. "
        "Chỉ trả về một JSON array hợp lệ, không trả về JSON object đơn, không markdown, không giải thích thêm. "
        "Mỗi phần tử trong array phải có các khóa: question_id, type, skill_target, difficulty, question, expected_keywords.\n\n"
        "Ví dụ định dạng đầu ra:\n"
        "[\n"
        "  {\n"
        '    "question_id": "q1",\n'
        '    "type": "technical",\n'
        '    "skill_target": "FastAPI/PostgreSQL",\n'
        '    "difficulty": "hard",\n'
        '    "question": "Hệ thống analytics dùng FastAPI và PostgreSQL bị chậm khi số lượng request tăng mạnh. Bạn sẽ điều tra bottleneck và tối ưu truy vấn như thế nào?",\n'
        '    "expected_keywords": ["index", "query plan", "connection pooling", "pagination", "join optimization"]\n'
        "  }\n"
        "]\n\n"
        f"Resume JSON:\n{resume_json.strip()}\n\n"
        f"Job Description:\n{jd_text.strip()}"
    )


# ── Answer evaluation ─────────────────────────────────────────────────────────

def build_answer_evaluation_prompt(
    question: str,
    answer: str,
    rubric: str,
    cv_evidence: str = "",
    jd_requirement: str = "",
) -> str:
    context_block = ""
    if cv_evidence:
        context_block += f"\nBằng chứng CV liên quan:\n{cv_evidence.strip()}"
    if jd_requirement:
        context_block += f"\nYêu cầu JD liên quan:\n{jd_requirement.strip()}"

    return (
        "Hãy đánh giá câu trả lời phỏng vấn và trả về JSON hợp lệ với các khóa: "
        "relevance_score, communication_score, confidence_score, overall_score, strengths, improvements. "
        "Giữ nguyên tên khóa bằng tiếng Anh, nhưng toàn bộ nội dung trong strengths và improvements phải viết bằng tiếng Việt. "
        "Điểm số là số thực trong khoảng [0.0, 1.0]. "
        "Chỉ trả về JSON, không markdown, không giải thích thêm.\n\n"
        f"Câu hỏi:\n{question.strip()}\n\n"
        f"Câu trả lời:\n{answer.strip()}\n\n"
        f"Tiêu chí đánh giá:\n{rubric.strip()}"
        f"{context_block}"
    )


# ── Follow-up decision ────────────────────────────────────────────────────────

def build_follow_up_prompt(
    question: str,
    answer: str,
    transcript_history: str,
    cv_evidence: str = "",
    jd_requirement: str = "",
    expected_signals: list[str] | None = None,
) -> str:
    signals_block = ""
    if expected_signals:
        signals_block = "\nTín hiệu kỳ vọng: " + ", ".join(expected_signals)

    context_block = ""
    if cv_evidence:
        context_block += f"\nBằng chứng CV liên quan:\n{cv_evidence.strip()}"
    if jd_requirement:
        context_block += f"\nYêu cầu JD liên quan:\n{jd_requirement.strip()}"

    return (
        "Dựa vào câu hỏi hiện tại, câu trả lời của ứng viên, và lịch sử phỏng vấn trước đó, "
        "hãy trả về JSON hợp lệ với các khóa: next_action, follow_up_question, rationale. "
        "Giá trị của next_action phải là một trong ba: follow_up, next_question, end_interview. "
        "Nên chọn follow_up khi: câu trả lời mơ hồ, thiếu số liệu cụ thể, ứng viên đề cập quyết định kỹ thuật thú vị cần đào sâu, "
        "hoặc có tuyên bố trong CV chưa được kiểm chứng. "
        "Tránh hỏi follow_up liên tục nếu câu trả lời đã đầy đủ. "
        "Nội dung follow_up_question và rationale phải viết bằng tiếng Việt. "
        "Chỉ trả về JSON, không markdown, không giải thích thêm.\n\n"
        f"Câu hỏi hiện tại:\n{question.strip()}\n\n"
        f"Câu trả lời ứng viên:\n{answer.strip()}\n\n"
        f"Lịch sử phỏng vấn:\n{transcript_history.strip()}"
        f"{signals_block}"
        f"{context_block}"
    )


# ── Resume optimisation ───────────────────────────────────────────────────────

def build_resume_optimize_prompt(resume_json: str, jd_text: str) -> str:
    return (
        "Dựa trên Resume JSON và Job Description, hãy trả về JSON với các khóa: "
        "matched_skills, missing_skills, suggested_resume_improvements. "
        "Chỉ trả về JSON, không markdown, không giải thích.\n\n"
        f"Resume JSON:\n{resume_json.strip()}\n\n"
        f"Job Description:\n{jd_text.strip()}"
    )
