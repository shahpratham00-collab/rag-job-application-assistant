"""
Prompt templates for the RAG Job Application Assistant.
"""

COVER_LETTER_PROMPT = """You are an expert UK career coach specialising in tech and AI roles.

Write a tailored professional cover letter for a UK job application.

JOB DESCRIPTION:
{job_description}

CANDIDATE CV / RESUME:
{cv_text}

RELEVANT CONTEXT FROM CV (retrieved):
{context}

STRICT INSTRUCTIONS:
1. Opening: Reference the specific company and role — avoid cliches like "I am thrilled" or "I am excited"
2. Paragraph 2: Pick the 3 most relevant projects/skills from the CV that directly match the job requirements — be specific with numbers and outcomes
3. Paragraph 3: Address the nice-to-have requirements the candidate meets — do NOT say they lack skills that ARE listed in their CV
4. Closing: Confident, professional UK-style closing — no "I would be thrilled"
5. Tone: Professional, confident, concise — UK hiring manager standard
6. Length: Exactly 3-4 paragraphs, 300-350 words
7. Sign off: "Yours sincerely, Pratham Shah"
8. IMPORTANT: Read the CV carefully — only say a skill is missing if it is genuinely absent from the CV

Write the cover letter now:"""


SKILLS_GAP_PROMPT = """You are a senior UK technical recruiter analysing candidate fit.

JOB DESCRIPTION:
{job_description}

CANDIDATE CV / RESUME:
{cv_text}

Analyse the skills gap carefully. Return ONLY this JSON with no extra text, no markdown, no explanation:

{{"match_score": <integer 0-100>, "matched_skills": [<list of strings>], "missing_skills": [<list of strings>], "partial_skills": [<list of strings>], "recommendations": [<list of 3-5 specific actionable strings>], "summary": "<2-3 sentence assessment string>"}}"""


QUICK_SUMMARY_PROMPT = """Summarise this job description in exactly 3 bullet points under 15 words each.

JOB DESCRIPTION:
{job_description}

Return only the 3 bullet points."""
