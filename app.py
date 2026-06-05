"""
RAG Job Application Assistant
Pratham Shah — MSc AI & Data Science, Nottingham Trent University
Project 7 of 7
"""
import os
import sys
import logging
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_engine import build_vector_store, generate_cover_letter, analyse_skills_gap
from src.utils import validate_inputs, score_to_emoji, clean_cover_letter, format_timestamp
from src.llm_selector import is_ollama_available

st.set_page_config(
    page_title="RAG Job Application Assistant | Pratham Shah",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.7, 0.1,
        help="Higher = more creative. Lower = more consistent.")
    st.markdown("---")
    st.markdown("### 🔌 LLM Backend")
    ollama_running = is_ollama_available()
    if ollama_running:
        st.success("✅ Ollama running locally")
        st.caption(f"Model: {os.getenv('OLLAMA_MODEL', 'llama3')}")
    else:
        hf_token = os.getenv("HUGGINGFACE_API_TOKEN", "")
        if hf_token:
            st.info("☁️ Using HuggingFace API")
            st.caption(os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"))
        else:
            st.error("⚠️ No LLM backend found")
            st.caption("Start Ollama or add HF token in .env")
    st.markdown("---")
    st.markdown("### 📋 How to Use")
    st.markdown("""
1. Paste the **job description**
2. Paste your **full CV**
3. Click **Analyse & Generate**
4. Review your **cover letter**
5. Check the **skills gap**
6. Follow the **action plan**
    """)
    st.markdown("---")
    st.markdown("### 👨‍💻 About")
    st.markdown("""
**Pratham Shah**
MSc AI & Data Science
Nottingham Trent University

*Portfolio Project 7 of 7*
[GitHub](https://github.com/shahpratham00-collab)
    """)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🎯 RAG Job Application Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Paste a job description and your CV — get a tailored cover letter, '
    'match score, and skills gap analysis powered by LangChain + Llama3.</p>',
    unsafe_allow_html=True
)

# ── Input area ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 📄 Job Description")
    job_description = st.text_area(
        label="jd",
        height=320,
        placeholder=(
            "Paste the full job description here...\n\n"
            "Example:\nWe are looking for a Machine Learning Engineer...\n\n"
            "Requirements:\n- Python, PyTorch, TensorFlow\n"
            "- NLP and transformers experience\n"
            "- Cloud platform experience (AWS/GCP)"
        ),
        key="jd_input",
        label_visibility="collapsed"
    )
with col2:
    st.markdown("### 📋 Your CV / Resume")
    cv_text = st.text_area(
        label="cv",
        height=320,
        placeholder=(
            "Paste your full CV here...\n\n"
            "Example:\nPratham Shah\nMSc AI & Data Science — NTU 2024\n\n"
            "EXPERIENCE:\nML Engineer Intern | Company | 2023\n"
            "- Built NLP pipeline with DistilBERT\n\n"
            "SKILLS:\nPython, TensorFlow, LangChain, Streamlit..."
        ),
        key="cv_input",
        label_visibility="collapsed"
    )

c1, c2 = st.columns(2)
with c1:
    if job_description:
        st.caption(f"📝 {len(job_description.split())} words")
with c2:
    if cv_text:
        st.caption(f"📝 {len(cv_text.split())} words")

st.markdown("---")

_, col_btn, _ = st.columns([1, 2, 1])
with col_btn:
    run_button = st.button(
        "🚀 Analyse & Generate Cover Letter",
        type="primary",
        use_container_width=True
    )

# ── Main logic ────────────────────────────────────────────────────────────────
if run_button:
    is_valid, error_msg = validate_inputs(cv_text, job_description)
    if not is_valid:
        st.error(f"❌ {error_msg}")
    else:
        progress = st.progress(0, text="Initialising RAG pipeline...")
        try:
            progress.progress(15, text="📦 Chunking and embedding documents...")
            vector_store = build_vector_store(cv_text, job_description)

            progress.progress(40, text="🔍 Analysing skills gap...")
            analysis, gap_backend = analyse_skills_gap(cv_text, job_description, temperature=0.1)

            progress.progress(70, text="✍️ Generating tailored cover letter...")
            cover_letter_raw, cl_backend = generate_cover_letter(
                cv_text, job_description, vector_store, temperature=temperature
            )
            cover_letter = clean_cover_letter(cover_letter_raw)

            progress.progress(100, text="✅ Done!")
            progress.empty()

            # ── Results ──────────────────────────────────────────────────────
            st.success(f"✅ Generated at {format_timestamp()}")

            tab1, tab2, tab3 = st.tabs(["✉️ Cover Letter", "📊 Skills Gap Analysis", "📋 Action Plan"])

            with tab1:
                backend_label = "🦙 Ollama" if cl_backend == "ollama" else "🤗 HuggingFace"
                st.markdown(
                    f'<span class="backend-badge">{backend_label}</span>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="cover-letter-box">{cover_letter}</div>',
                    unsafe_allow_html=True
                )
                st.download_button(
                    label="⬇️ Download Cover Letter (.txt)",
                    data=cover_letter,
                    file_name="cover_letter.txt",
                    mime="text/plain"
                )

            with tab2:
                score = analysis.get("match_score", 0)
                emoji = score_to_emoji(score)
                st.markdown(
                    f'<div class="score-card">'
                    f'<div class="score-number">{emoji} {score}</div>'
                    f'<div class="score-label">Match Score / 100</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown(f"**Summary:** {analysis.get('summary', 'N/A')}")
                st.markdown("---")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown("#### ✅ Matched Skills")
                    badges = "".join(
                        f'<span class="skill-badge-green">{s}</span>'
                        for s in analysis.get("matched_skills", [])
                    )
                    st.markdown(badges or "_None identified_", unsafe_allow_html=True)
                with col_b:
                    st.markdown("#### ❌ Missing Skills")
                    badges = "".join(
                        f'<span class="skill-badge-red">{s}</span>'
                        for s in analysis.get("missing_skills", [])
                    )
                    st.markdown(badges or "_None identified_", unsafe_allow_html=True)
                with col_c:
                    st.markdown("#### 🟡 Partial Skills")
                    badges = "".join(
                        f'<span class="skill-badge-yellow">{s}</span>'
                        for s in analysis.get("partial_skills", [])
                    )
                    st.markdown(badges or "_None identified_", unsafe_allow_html=True)

            with tab3:
                st.markdown("### 🗺️ Your Personalised Action Plan")
                recommendations = analysis.get("recommendations", [])
                if recommendations:
                    for i, rec in enumerate(recommendations, 1):
                        st.markdown(f"**{i}.** {rec}")
                else:
                    st.info("No specific recommendations generated.")
                st.markdown("---")
                st.markdown("#### 💡 General Tips")
                st.markdown("""
- **Tailor for every application** — use this tool for each role, not just once.
- **Quantify your achievements** — add numbers wherever possible in your CV.
- **Close skills gaps fast** — one Coursera/Udemy course can change a ❌ to ✅.
- **Network on LinkedIn** — 70% of roles are filled before they're advertised.
                """)

        except RuntimeError as e:
            progress.empty()
            st.error(f"🔴 LLM Error: {e}")
            st.info("💡 Tip: Start Ollama with `ollama serve` and pull a model with `ollama pull llama3`")
        except Exception as e:
            progress.empty()
            logger.exception("Unexpected error during generation")
            st.error(f"❌ Unexpected error: {e}")
            st.exception(e)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>RAG Job Application Assistant · "
    "Pratham Shah · MSc AI & Data Science · NTU · "
    "Portfolio Project 7 of 7</small></center>",
    unsafe_allow_html=True
)
