"""
================================================================================
Figure of Speech Analyzer & Beautifier Toolkit (Academic NLP Project)
================================================================================
A streamlined, academic-grade Natural Language Processing web application
built with Streamlit, NLTK, and the Google GenAI SDK (Gemini 2.5 Flash).

Components:
1. Classical Rule-Based NLP Pipeline:
   - Similes (Syntactic Regex & POS Tagging)
   - Alliteration (CMU Pronouncing Dictionary Phonemic Onset Clustering)
   - Oxymorons (WordNet Antonymy Graph & Contradiction Lexicon)
   - Onomatopoeia (Sound-Symbolic Lexicon & WordNet Lemmatization)
2. Generative LLM Rewriter (Gemini 2.5 Flash):
   - Literary stylistic elevation and figures of speech synthesis
"""

import os
from dotenv import load_dotenv
load_dotenv()
# try:
#     from dotenv import load_dotenv
#     load_dotenv()
# except ImportError:
#     pass
import streamlit as st
from typing import List, Dict, Optional
import figure_detector as fd


st.set_page_config(
    page_title="Figure of Speech Analyzer & Beautifier",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom minimalist academic styling with modern color scheme & micro-interactions
CUSTOM_CSS = """
<style>
/* Global aesthetic adjustments */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header Banner */
.main-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    color: #f8fafc;
    padding: 24px 30px;
    border-radius: 14px;
    margin-bottom: 24px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
}
.main-header h1 {
    color: #ffffff !important;
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    margin-bottom: 6px !important;
    display: flex;
    align-items: center;
    gap: 12px;
}
.main-header p {
    color: #94a3b8 !important;
    font-size: 0.95rem;
    margin: 0 !important;
}

/* Metric KPI Cards */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 12px;
    margin: 18px 0;
}
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px 14px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.06);
}
.kpi-num {
    font-size: 1.5rem;
    font-weight: 700;
    line-height: 1.2;
}
.kpi-label {
    font-size: 0.76rem;
    color: #64748b;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-top: 4px;
}

/* Legend badges */
.legend-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 8px;
    margin-bottom: 8px;
    border: 1px solid transparent;
}
.legend-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}

/* Highlight Output Container */
.highlight-container {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 22px 24px;
    margin-top: 10px;
    margin-bottom: 24px;
    font-size: 1.05rem;
    line-height: 1.85;
    color: #1e293b;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.02);
}

/* Data summary card */
.summary-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #64748b;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.summary-card.Simile { border-left-color: #10b981; }
.summary-card.Alliteration { border-left-color: #2563eb; }
.summary-card.Oxymoron { border-left-color: #d97706; }
.summary-card.Onomatopoeia { border-left-color: #7c3aed; }

/* Beautified Container */
.beautified-container,
div[class*="st-key-beautified_box"] [data-testid="stVerticalBlockBorderWrapper"],
div[class*="st-key-beautified_box"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 22px 24px !important;
    margin-top: 10px !important;
    color: #1e293b !important;
    font-size: 1.05rem !important;
    line-height: 1.85 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
}
div[class*="st-key-beautified_box"] p,
div[class*="st-key-beautified_box"] span,
div[class*="st-key-beautified_box"] li,
div[class*="st-key-beautified_box"] strong,
div[class*="st-key-beautified_box"] em,
div[class*="st-key-beautified_box"] h1,
div[class*="st-key-beautified_box"] h2,
div[class*="st-key-beautified_box"] h3,
div[class*="st-key-beautified_box"] h4,
.beautified-container p,
.beautified-container span,
.beautified-container li,
.beautified-container strong,
.beautified-container em,
.beautified-container h1,
.beautified-container h2,
.beautified-container h3,
.beautified-container h4 {
    color: #1e293b !important;
}

/* Primary buttons styling */
div[data-testid="stButton"] button {
    font-weight: 600;
    border-radius: 8px;
    height: 46px;
    transition: all 0.2s ease;
}
div[data-testid="stButton"] button:hover {
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


PRESET_SAMPLES = {
    "Select a pre-built sample or write your own...": "",
    "Sample 1: Classical Literature & Poetic Prose": (
        "The brave warrior stood as firm as a fortress against the encroaching storm. "
        "Suddenly, a deafening silence blanketed the moonlit moor as the whispering winds wandered "
        "wild and free through the tall pines. In the quiet distance, sparks sizzled and crackled "
        "over the dying campfire, and an owl flew as swift as an arrow through the dark."
    ),
    "Sample 2: Battle & Dramatic Sensory Description": (
        "The knight fought like a lion in the midst of blood and iron, yet he found only sweet sorrow "
        "in victory. The heavy hammer struck the anvil with a resonant clang, while lightning flashed "
        "like fiery serpents across the sky. Peter paused pensively as the cold rain went patter against "
        "his armor, maintaining a calm chaos amid the battlefield."
    ),
    "Sample 3: Simple Everyday Narrative (Best for Beautifying)": (
        "The old man walked slowly down the dark empty street at night. "
        "The cold wind blew against his thin jacket. The leaves made noise on the sidewalk. "
        "He stopped outside the bakery window because he was very hungry. "
        "The city was quiet and lonely."
    )
}


with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    st.caption("Figure of Speech NLP Analysis & LLM Rewriter Toolkit")

    st.markdown("---")
    st.markdown("#### 📚 Classical NLP Pipeline Architecture")
    with st.expander("🔬 Processing Pipeline & Methodology", expanded=True):
        st.markdown("""
        **NLTK Preprocessing Pipeline:**
        1. **Sentence Tokenization**: Paragraph is segmented into individual sentences (`sent_tokenize`).
        2. **Word Tokenization**: Each sentence is parsed into lexical tokens (`word_tokenize`).
        3. **POS Tagging**: Syntactic role assignment using Penn Treebank tags (`pos_tag`).
        4. **Stopwords Filtering**: Inbuilt NLTK stopwords (`nltk.corpus.stopwords`) applied during alliteration & oxymoron resolution.

        **Figure of Speech Detectors:**
        - **Simile**: Syntactic pattern detection (`as ... as`, `like ...`) validated via POS-tags.
        - **Alliteration**: Phonetic consonant onset clustering via **CMU Pronouncing Dictionary** (`cmudict`).
        - **Oxymoron**: Antonymous word-pair clash verified via **WordNet** antonymy relations & curated lexicon.
        - **Onomatopoeia**: Acoustic sound-symbolism matching with **WordNet Lemmatization**.
        """)

    st.markdown("---")
    st.caption("Academic NLP Project | Powered by NLTK & Gemini")


st.markdown("""
<div class="main-header">
    <h1><span>✨</span> Figure of Speech Toolkit</h1>
    <p>Rule-Based Classical NLP Analysis & LLM-Powered Literary Elevation</p>
</div>
""", unsafe_allow_html=True)


col_sel, _ = st.columns([2, 1])
with col_sel:
    selected_sample = st.selectbox(
        "Load an Academic Test Case:",
        options=list(PRESET_SAMPLES.keys()),
        index=1
    )

# Text Area for User Input
default_text = PRESET_SAMPLES.get(selected_sample, "")
input_text = st.text_area(
    "Input Paragraph for Linguistic Analysis & Stylistic Enhancement:",
    value=default_text,
    height=160,
    placeholder="Type or paste a paragraph here to analyze or elevate with figures of speech...",
    help="Enter standard English text. The rule-based engine will parse similes, alliteration, oxymorons, and onomatopoeia."
)

# Initialize Session State
if "detections" not in st.session_state:
    st.session_state.detections = None
if "detected_text" not in st.session_state:
    st.session_state.detected_text = ""
if "beautified_output" not in st.session_state:
    st.session_state.beautified_output = None
if "beautified_detections" not in st.session_state:
    st.session_state.beautified_detections = None
if "used_model" not in st.session_state:
    st.session_state.used_model = None

# Two distinct primary action buttons placed side-by-side
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    detect_clicked = st.button("🔍 Detect Figures of Speech", use_container_width=True, type="primary")

with col_btn2:
    beautify_clicked = st.button("✨ Beautify Paragraph with LLM", use_container_width=True)


# -----------------------------------------------------------------------------
# 5. Execution Handler: Button 1 — Detect (Rule-Based Pipeline)
# -----------------------------------------------------------------------------
if detect_clicked:
    if not input_text or not input_text.strip():
        st.warning("Please enter or paste a paragraph into the text area before running detection.", icon="⚠️")
    else:
        with st.spinner("Executing rule-based classical NLP analysis pipeline..."):
            detections = fd.analyze_all_figures(input_text.strip())
            st.session_state.detections = detections
            st.session_state.detected_text = input_text.strip()


# -----------------------------------------------------------------------------
# 6. Execution Handler: Button 2 — Beautify (LLM-Powered Rewriter)
# -----------------------------------------------------------------------------
models = {
    "gemini3.8": "gemini-3.8-flash",
    "gemini3.5lite": "gemini-3.5-flash-lite",
    "gemini3.5": "gemini-3.5-flash",
}

if beautify_clicked:
    if not input_text or not input_text.strip():
        st.warning("Please enter or paste a paragraph into the text area before beautifying.", icon="⚠️")
    else:
        gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not gemini_api_key:
            st.error(
                "🔑 **GEMINI_API_KEY Not Found**: No API key detected in your `.env` file. "
                "Please add `GEMINI_API_KEY=your_key_here` to your `.env` file to use the Beautify feature.",
                icon="🚨"
            )
        else:
            with st.spinner("Getting your rich literary figures of speech..."):
                try:
                    from google import genai
                    from google.genai import types

                    client = genai.Client(api_key=gemini_api_key)
                    prompt = (
                        "You are an expert literary stylist and creative linguistics scholar. "
                        "Take the following user paragraph and rewrite it into an elevated, evocative, "
                        "and literary masterpiece. "
                        "Naturally weave in rich and prominent figures of speech, ensuring you include:\n"
                        "1. Vivid Similes (using 'like' or 'as ... as')\n"
                        "2. Musical Alliteration (consonant repetition)\n"
                        "3. Expressive Oxymorons (juxtaposition of contradictory words)\n"
                        "4. Sensory Onomatopoeia (sound-imitation words)\n\n"
                        "Output Requirements:\n"
                        "- Begin directly with the elevated rewritten paragraph.\n"
                        "- Then include a markdown divider (---) followed by an academic breakdown: "
                        "'### 📖 Literary Figures of Speech Introduced' with bullet points explaining each added figure.\n\n"
                        f"Original Paragraph:\n\"\"\"{input_text.strip()}\"\"\""
                    )

                    beautified_text = None
                    used_model_name = None
                    used_model_id = None
                    model_errors = []

                    # Iterate through the models in priority order; failover if one fails to respond
                    for alias, model_id in models.items():
                        try:
                            response = client.models.generate_content(
                                model=model_id,
                                contents=prompt
                            )
                            if response and response.text:
                                beautified_text = response.text
                                used_model_name = alias
                                used_model_id = model_id
                                break
                        except Exception as model_err:
                            model_errors.append(f"`{model_id}` ({alias}): {str(model_err)}")
                            continue

                    if beautified_text:
                        st.session_state.beautified_output = beautified_text
                    else:
                        error_bullets = "\n".join([f"- {err}" for err in model_errors])
                        st.error(
                            f"All Gemini models failed to respond:\n\n{error_bullets}",
                            icon="❌"
                        )

                except Exception as e:
                    st.error(f"Failed to initialize Gemini client: {str(e)}", icon="❌")


if st.session_state.detections is not None:
    detections: List[fd.DetectedFigure] = st.session_state.detections
    detected_text = st.session_state.detected_text

    st.markdown("---")
    st.markdown("### 📊 Detection Analysis Results")

    simile_cnt = sum(1 for d in detections if d.figure_type == "Simile")
    allit_cnt = sum(1 for d in detections if d.figure_type == "Alliteration")
    oxy_cnt = sum(1 for d in detections if d.figure_type == "Oxymoron")
    onoma_cnt = sum(1 for d in detections if d.figure_type == "Onomatopoeia")
    total_cnt = len(detections)

    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card" style="border-top: 3px solid #0f172a;">
            <div class="kpi-num" style="color: #0f172a;">{total_cnt}</div>
            <div class="kpi-label">Total Found</div>
        </div>
        <div class="kpi-card" style="border-top: 3px solid {fd.FIGURE_CONFIG['Simile']['color']};">
            <div class="kpi-num" style="color: {fd.FIGURE_CONFIG['Simile']['color']};">{simile_cnt}</div>
            <div class="kpi-label">Similes</div>
        </div>
        <div class="kpi-card" style="border-top: 3px solid {fd.FIGURE_CONFIG['Alliteration']['color']};">
            <div class="kpi-num" style="color: {fd.FIGURE_CONFIG['Alliteration']['color']};">{allit_cnt}</div>
            <div class="kpi-label">Alliterations</div>
        </div>
        <div class="kpi-card" style="border-top: 3px solid {fd.FIGURE_CONFIG['Oxymoron']['color']};">
            <div class="kpi-num" style="color: {fd.FIGURE_CONFIG['Oxymoron']['color']};">{oxy_cnt}</div>
            <div class="kpi-label">Oxymorons</div>
        </div>
        <div class="kpi-card" style="border-top: 3px solid {fd.FIGURE_CONFIG['Onomatopoeia']['color']};">
            <div class="kpi-num" style="color: {fd.FIGURE_CONFIG['Onomatopoeia']['color']};">{onoma_cnt}</div>
            <div class="kpi-label">Onomatopoeia</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Color Legend & Methodology:**")
    legend_cols = st.columns(4)
    for idx, (fig_name, config) in enumerate(fd.FIGURE_CONFIG.items()):
        with legend_cols[idx]:
            st.markdown(f"""
            <div class="legend-pill" style="background: {config['bg_color']}; color: {config['text_color']}; border-color: {config['border_color']};">
                <span class="legend-dot" style="background: {config['color']};"></span>
                <span>{fig_name}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("#### 📝 Annotated Text with Inline Badges")
    highlighted_html = fd.render_highlighted_text(detected_text, detections)
    st.markdown(f'<div class="highlight-container">{highlighted_html}</div>', unsafe_allow_html=True)

    st.markdown("#### 📋 Detailed Evidence & Rule Validation")
    if not detections:
        st.info("No figures of speech were detected in the input paragraph using the rule-based heuristics.", icon="ℹ️")
    else:

        filter_option = st.radio(
            "Filter by Figure of Speech:",
            options=["All", "Simile", "Alliteration", "Oxymoron", "Onomatopoeia"],
            horizontal=True
        )

        filtered_detections = [
            d for d in detections
            if filter_option == "All" or d.figure_type == filter_option
        ]

        st.caption(f"Showing {len(filtered_detections)} matching item(s):")

        for d in filtered_detections:
            config = fd.FIGURE_CONFIG[d.figure_type]
            st.markdown(f"""
            <div class="summary-card {d.figure_type}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-weight: 700; font-size: 1.05rem; color: #0f172a;">
                        &ldquo;{d.phrase}&rdquo;
                    </span>
                    <span style="background: {config['badge_bg']}; color: #ffffff; font-size: 0.75rem; padding: 2px 8px; border-radius: 9999px; font-weight: 600;">
                        {d.figure_type.upper()}
                    </span>
                </div>
                <div style="font-size: 0.88rem; color: #475569; margin-bottom: 4px;">
                    <strong>Context:</strong> <em>&ldquo;{d.sentence}&rdquo;</em>
                </div>
                <div style="font-size: 0.84rem; color: #64748b; margin-bottom: 2px;">
                    <strong>NLP Method:</strong> {d.method}
                </div>
                <div style="font-size: 0.84rem; color: {config['text_color']};">
                    <strong>Linguistic Rationale:</strong> {d.explanation}
                </div>
            </div>
            """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 8. Render Output Section: Beautify Results
# -----------------------------------------------------------------------------
if st.session_state.beautified_output:
    st.markdown("---")
    st.markdown("### ✨ LLM Literary Beautification")

    with st.container(border=True, key="beautified_box"):
        st.markdown(st.session_state.beautified_output)

    # Secondary action: Allow user to run the classical detector on the beautified paragraph!
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔬 Run Rule-Based Detector on Beautified Text"):
        # Extract the rewritten paragraph (before the divider if present)
        raw_beautified = st.session_state.beautified_output
        if "---" in raw_beautified:
            target_paragraph = raw_beautified.split("---")[0].strip()
        else:
            target_paragraph = raw_beautified.strip()

        st.session_state.detections = fd.analyze_all_figures(target_paragraph)
        st.session_state.detected_text = target_paragraph
        st.rerun()
