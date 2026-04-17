SYSTEM_PROMPT = (
    "You are an interview AI assistant. "
    "Return structured, concise, valid JSON when the task asks for JSON."
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
        "Generate 5 interview questions in JSON. "
        "Each item must include question_id, type, skill_target, difficulty, "
        "question, expected_keywords.\n\n"
        f"Resume JSON:\n{resume_json.strip()}\n\n"
        f"Job Description:\n{jd_text.strip()}"
    )


def build_answer_evaluation_prompt(question: str, answer: str, rubric: str) -> str:
    return (
        "Evaluate the interview answer and return JSON with keys: "
        "relevance_score, communication_score, confidence_score, overall_score, "
        "strengths, improvements.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Answer:\n{answer.strip()}\n\n"
        f"Rubric:\n{rubric.strip()}"
    )
