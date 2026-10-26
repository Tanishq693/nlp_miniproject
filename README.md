# Figure of Speech Analyzer & Beautifier Toolkit 🎭✨
### Academic NLP Mini-Project

A clean, minimalist web application built with **Streamlit**, classical **NLTK rule-based NLP algorithms**, and **Gemini 2.5 Flash** (via the official `google-genai` SDK) to analyze, detect, and elevate figures of speech in text.

---

## 🌟 Core Features

### 1. 🔍 Rule-Based Classical NLP Pipeline ("Detect")
Detects 4 classical figures of speech with transparent computational linguistic methodologies:
* **Simile** (*Emerald Green*): Equative (`as ... as`) and comparative prepositional (`like ...`) syntactic structures validated against Penn Treebank POS tags (`JJ*`, `RB*`, `IN`).
* **Alliteration** (*Sapphire Blue*): Phonemic onset consonant matching using the **CMU Pronouncing Dictionary** (`cmudict`) with orthographic consonant fallback and stop-word skipping.
* **Oxymoron** (*Amber Orange*): Juxtaposition of semantically contradictory terms resolved via **WordNet** antonymy graph traversal and a curated classical literary contradiction lexicon.
* **Onomatopoeia** (*Amethyst Purple*): Acoustic sound-symbolism matching coupled with **WordNet Lemmatization** for inflected verb/noun forms (`sizzled`, `crackled`, `buzzing`).

### 2. ✨ LLM-Powered Literary Rewriter ("Beautify")
* Utilizes **Gemini 2.5 Flash** (`gemini-2.5-flash`) via the official `google-genai` SDK.
* Rewrites ordinary input text into an evocative, literary passage weaving in rich figures of speech.
* Provides an academic breakdown explaining each figurative enhancement introduced.
* Includes a secondary action to immediately run the rule-based detector on the beautified text to measure figurative density.

---

## 📁 Project Architecture

```
nlp_miniproject/
├── app.py                 # Streamlit web application & UI layout
├── figure_detector.py     # Classical rule-based NLP detection engine & HTML highlighter
├── test.ipynb             # Jupyter Notebook unit tests & academic evaluation suite
├── README.md              # Project documentation
└── .venv/                 # Python virtual environment
```

---

## 🚀 How to Run

### 1. Activate Environment & Run Web App
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
Then navigate to `http://localhost:8501` in your browser.

### 2. (Optional) Provide Gemini API Key for "Beautify"
You can set your Gemini API key in your terminal before launching:
```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```
Or simply paste your API key directly into the sidebar text field inside the Streamlit web interface.
*(Note: The "Detect" engine runs 100% offline using local NLTK rules and does not require an API key).*

### 3. Run Academic Notebook Tests
Open and run `test.ipynb` in VS Code or Jupyter Notebook to execute the standalone tests and demonstrations for each figure of speech.
