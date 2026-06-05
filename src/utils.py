"""
Utility helpers.
"""
import re
from datetime import datetime


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


def validate_inputs(cv_text: str, job_description: str) -> tuple:
    if not cv_text or not cv_text.strip():
        return False, "Please paste your CV / resume text."
    if not job_description or not job_description.strip():
        return False, "Please paste the job description."
    if word_count(cv_text) < 50:
        return False, "CV seems too short (under 50 words). Please paste your full CV."
    if word_count(job_description) < 30:
        return False, "Job description seems too short. Please paste the full JD."
    return True, ""


def score_to_emoji(score: int) -> str:
    if score >= 75:
        return "🟢"
    elif score >= 50:
        return "🟡"
    return "🔴"


def format_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clean_cover_letter(text: str) -> str:
    preambles = [
        r"^here is (your |a |the )?(tailored |professional |cover letter[:\s]*)",
        r"^cover letter[:\s]*",
        r"^sure[,!]?\s+",
        r"^of course[,!]?\s+",
        r"^certainly[,!]?\s+",
    ]
    result = text.strip()
    for pattern in preambles:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE).strip()
    return result
