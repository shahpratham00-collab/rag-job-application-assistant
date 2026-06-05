"""
Unit tests for the RAG Job Application Assistant.
Run with: pytest tests/ -v
"""
import pytest
from unittest.mock import patch, MagicMock

from src.utils import word_count, validate_inputs, score_to_emoji, clean_cover_letter
from src.parser import parse_skills_gap_json, build_fallback_analysis


# ── Utils tests ───────────────────────────────────────────────────────────────

class TestWordCount:
    def test_normal_sentence(self):
        assert word_count("hello world foo") == 3

    def test_empty_string(self):
        assert word_count("") == 0

    def test_none_equivalent(self):
        assert word_count("") == 0

    def test_single_word(self):
        assert word_count("Python") == 1


class TestValidateInputs:
    GOOD_CV = "Python developer with 5 years experience in machine learning and NLP projects. " * 5
    GOOD_JD = "We are looking for a Python developer with ML experience. " * 3

    def test_valid_inputs(self):
        ok, msg = validate_inputs(self.GOOD_CV, self.GOOD_JD)
        assert ok is True
        assert msg == ""

    def test_empty_cv(self):
        ok, msg = validate_inputs("", self.GOOD_JD)
        assert ok is False
        assert "CV" in msg

    def test_empty_jd(self):
        ok, msg = validate_inputs(self.GOOD_CV, "")
        assert ok is False
        assert "job description" in msg.lower()

    def test_short_cv(self):
        ok, msg = validate_inputs("Short CV.", self.GOOD_JD)
        assert ok is False
        assert "50 words" in msg

    def test_short_jd(self):
        ok, msg = validate_inputs(self.GOOD_CV, "Short JD.")
        assert ok is False
        assert "30" in msg


class TestScoreToEmoji:
    def test_high_score(self):
        assert score_to_emoji(80) == "🟢"

    def test_mid_score(self):
        assert score_to_emoji(60) == "🟡"

    def test_low_score(self):
        assert score_to_emoji(30) == "🔴"

    def test_boundary_75(self):
        assert score_to_emoji(75) == "🟢"

    def test_boundary_50(self):
        assert score_to_emoji(50) == "🟡"

    def test_boundary_49(self):
        assert score_to_emoji(49) == "🔴"


class TestCleanCoverLetter:
    def test_removes_here_is_preamble(self):
        raw = "Here is your tailored cover letter:\n\nDear Hiring Manager,"
        result = clean_cover_letter(raw)
        assert result.startswith("Dear Hiring Manager,")

    def test_removes_sure_preamble(self):
        raw = "Sure! Here is the cover letter.\n\nDear Sir,"
        result = clean_cover_letter(raw)
        assert not result.lower().startswith("sure")

    def test_no_preamble_unchanged(self):
        raw = "Dear Hiring Manager, I am writing to express..."
        result = clean_cover_letter(raw)
        assert result == raw

    def test_strips_whitespace(self):
        raw = "   \n\nDear Manager,   "
        result = clean_cover_letter(raw)
        assert result == "Dear Manager,"


# ── Parser tests ──────────────────────────────────────────────────────────────

class TestParseSkillsGapJson:
    VALID_JSON = '''{
        "match_score": 72,
        "matched_skills": ["Python", "SQL"],
        "missing_skills": ["Kubernetes"],
        "partial_skills": ["Docker"],
        "recommendations": ["Learn Kubernetes basics"],
        "summary": "Strong candidate with some gaps."
    }'''

    def test_valid_json(self):
        result = parse_skills_gap_json(self.VALID_JSON)
        assert result is not None
        assert result["match_score"] == 72
        assert "Python" in result["matched_skills"]

    def test_json_in_fences(self):
        fenced = f"```json\n{self.VALID_JSON}\n```"
        result = parse_skills_gap_json(fenced)
        assert result is not None
        assert result["match_score"] == 72

    def test_json_with_surrounding_text(self):
        messy = f"Here is the analysis:\n{self.VALID_JSON}\nThat's all."
        result = parse_skills_gap_json(messy)
        assert result is not None

    def test_empty_string(self):
        result = parse_skills_gap_json("")
        assert result is None

    def test_none_input(self):
        result = parse_skills_gap_json(None)
        assert result is None

    def test_invalid_json(self):
        result = parse_skills_gap_json("not json at all")
        assert result is None


class TestBuildFallbackAnalysis:
    def test_structure(self):
        result = build_fallback_analysis("Some raw text output.")
        assert "match_score" in result
        assert "matched_skills" in result
        assert "missing_skills" in result
        assert "partial_skills" in result
        assert "recommendations" in result
        assert "summary" in result

    def test_default_score(self):
        result = build_fallback_analysis("Some output.")
        assert result["match_score"] == 50

    def test_summary_truncation(self):
        long_text = "x" * 1000
        result = build_fallback_analysis(long_text)
        assert len(result["summary"]) <= 500

    def test_empty_input(self):
        result = build_fallback_analysis("")
        assert result["summary"] == "No analysis generated."


# ── LLM selector tests ────────────────────────────────────────────────────────

class TestIsOllamaAvailable:
    def test_returns_false_when_offline(self):
        from src.llm_selector import is_ollama_available
        result = is_ollama_available("http://localhost:9999")
        assert result is False

    @patch("src.llm_selector.requests")
    def test_returns_true_when_online(self, mock_requests):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests.get.return_value = mock_response
        from src.llm_selector import is_ollama_available
        result = is_ollama_available("http://localhost:11434")
        assert result is True
