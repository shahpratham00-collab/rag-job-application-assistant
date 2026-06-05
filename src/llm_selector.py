"""
LLM selector — automatically picks Ollama (local) or HuggingFace (cloud).
"""
import os
import logging

logger = logging.getLogger(__name__)


def is_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    try:
        import requests
        response = requests.get(f"{base_url}/api/tags", timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def get_llm(temperature: float = 0.7):
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3")

    if is_ollama_available(ollama_url):
        logger.info("Using Ollama: %s", ollama_model)
        try:
            from langchain_community.llms import Ollama
            return Ollama(model=ollama_model, base_url=ollama_url, temperature=temperature), "ollama"
        except Exception as e:
            logger.warning("Ollama import failed: %s", e)

    hf_token = os.getenv("HUGGINGFACE_API_TOKEN", "")
    hf_model = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

    if hf_token:
        logger.info("Using HuggingFace: %s", hf_model)
        try:
            from langchain_huggingface import HuggingFaceEndpoint
            return HuggingFaceEndpoint(
                repo_id=hf_model,
                huggingfacehub_api_token=hf_token,
                temperature=temperature,
                max_new_tokens=1024,
            ), "huggingface"
        except Exception as e:
            logger.warning("HuggingFace import failed: %s", e)

    raise RuntimeError(
        "No LLM available. Either start Ollama locally or set HUGGINGFACE_API_TOKEN in your .env file."
    )


def get_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
