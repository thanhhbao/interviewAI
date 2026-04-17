SYSTEM_PROMPT = (
    "You are an interview AI assistant for Vietnamese interview practice. "
    "When the task asks for JSON, return structured, concise, valid JSON only. "
    "When generating interview questions, follow-up questions, or interview reviews, write the natural-language content in Vietnamese."
)


def build_resume_extract_prompt(resume_text: str) -> str:
    return (
        "Extract the following resume into JSON with keys: "
        "candidate_profile.name, email, phone, summary, skills, experience, "
        "education, projects.\n\n"
        f"Resume:\n{resume_text.strip()}"
    )


def build_resume_optimize_prompt(resume_json: str, jd_text: str) -> str:
    return (
        "Given the resume JSON and job description, return JSON with keys: "
        "matched_skills, missing_skills, suggested_resume_improvements.\n\n"
        f"Resume JSON:\n{resume_json.strip()}\n\n"
        f"Job Description:\n{jd_text.strip()}"
    )


def build_question_generation_prompt(resume_json: str, jd_text: str) -> str:
    return (
        "Hay tao chinh xac 5 cau hoi phong van duoi dang mot JSON array hop le. "
        "Chi tra ve JSON array, khong tra ve JSON object don, khong markdown, khong giai thich them. "
        "Noi dung truong question phai viet bang tieng Viet tu nhien. "
        "Moi phan tu trong array phai co cac khoa: question_id, type, skill_target, difficulty, question, expected_keywords.\n\n"
        "Vi du dinh dang dau ra:\n"
        "[\n"
        "  {\n"
        '    "question_id": "q1",\n'
        '    "type": "technical",\n'
        '    "skill_target": "python",\n'
        '    "difficulty": "medium",\n'
        '    "question": "Hay mo ta mot du an backend ban da thuc hien bang Python.",\n'
        '    "expected_keywords": ["python", "api"]\n'
        "  }\n"
        "]\n\n"
        f"Resume JSON:\n{resume_json.strip()}\n\n"
        f"Job Description:\n{jd_text.strip()}"
    )


def build_answer_evaluation_prompt(question: str, answer: str, rubric: str) -> str:
    return (
        "Hay danh gia cau tra loi phong van va tra ve JSON hop le voi cac khoa: "
        "relevance_score, communication_score, confidence_score, overall_score, strengths, improvements. "
        "Hay giu nguyen ten khoa bang tieng Anh, nhung toan bo noi dung trong strengths va improvements phai viet bang tieng Viet. "
        "Chi tra ve JSON, khong markdown, khong giai thich them.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Answer:\n{answer.strip()}\n\n"
        f"Rubric:\n{rubric.strip()}"
    )


def build_follow_up_prompt(question: str, answer: str, transcript_history: str) -> str:
    return (
        "Duoi vao cau hoi hien tai, cau tra loi cua ung vien, va lich su phong van truoc do, "
        "hay tra ve JSON hop le voi cac khoa: next_action, follow_up_question, rationale. "
        "Gia tri cua next_action phai la mot trong ba gia tri: follow_up, next_question, end_interview. "
        "Neu cau tra loi mo ho, thieu thong tin, hoac mo ra chi tiet du an dang de dao sau, uu tien follow_up. "
        "Noi dung follow_up_question va rationale phai viet bang tieng Viet. "
        "Chi tra ve JSON, khong markdown, khong giai thich them.\n\n"
        f"Current Question:\n{question.strip()}\n\n"
        f"Candidate Answer:\n{answer.strip()}\n\n"
        f"Transcript History:\n{transcript_history.strip()}"
    )
