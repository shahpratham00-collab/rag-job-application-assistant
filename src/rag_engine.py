"""
Core RAG engine — document ingestion, vector store, chain execution.
Modern LangChain (0.2+) compatible — no deprecated LLMChain.
"""
import logging
from typing import Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from src.llm_selector import get_llm, get_embeddings
from src.prompts import COVER_LETTER_PROMPT, SKILLS_GAP_PROMPT
from src.parser import parse_skills_gap_json, build_fallback_analysis

logger = logging.getLogger(__name__)


def build_vector_store(cv_text: str, job_description: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    combined_text = f"JOB DESCRIPTION:\n{job_description}\n\nCANDIDATE CV:\n{cv_text}"
    chunks = splitter.create_documents([combined_text])
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)


def retrieve_context(vector_store, query: str, k: int = 4) -> str:
    docs = vector_store.similarity_search(query, k=k)
    return "\n\n".join(doc.page_content for doc in docs)


def run_chain(llm, prompt_template: str, variables: dict) -> str:
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    result = chain.invoke(variables)
    if hasattr(result, "content"):
        return result.content
    return str(result)


def generate_cover_letter(cv_text: str, job_description: str, vector_store, temperature: float = 0.7) -> Tuple[str, str]:
    llm, backend = get_llm(temperature=temperature)
    context = retrieve_context(vector_store, query="relevant experience skills achievements for this job", k=4)
    result = run_chain(llm, COVER_LETTER_PROMPT, {
        "job_description": job_description,
        "cv_text": cv_text,
        "context": context
    })
    return result.strip(), backend


def analyse_skills_gap(cv_text: str, job_description: str, temperature: float = 0.1) -> Tuple[dict, str]:
    llm, backend = get_llm(temperature=temperature)
    result = run_chain(llm, SKILLS_GAP_PROMPT, {
        "job_description": job_description,
        "cv_text": cv_text
    })
    parsed = parse_skills_gap_json(result)
    if parsed is None:
        parsed = build_fallback_analysis(result)
    return parsed, backend
