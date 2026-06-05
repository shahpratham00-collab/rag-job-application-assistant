"""
Output parsers for structured LLM responses.
"""
import json
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def parse_skills_gap_json(raw_output: str) -> Optional[dict]:
    if not raw_output:
        return None
    try:
        return json.loads(raw_output.strip())
    except json.JSONDecodeError:
        pass
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass
    brace_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    logger.warning("Could not parse JSON from LLM output.")
    return None


def build_fallback_analysis(raw_output: str) -> dict:
    return {
        "match_score": 50,
        "matched_skills": ["Unable to parse — see summary below"],
        "missing_skills": ["Unable to parse — see summary below"],
        "partial_skills": [],
        "recommendations": ["Please review the summary below for details"],
        "summary": raw_output[:500] if raw_output else "No analysis generated."
    }
