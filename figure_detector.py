"""
Figure of Speech Detector Module (Rule-Based Classical NLP Pipeline)
=====================================================================
Developed for Academic NLP Project: Figure of Speech Analyzer & Beautifier.
Uses the pipeline architecture from test.ipynb:
  1. Paragraph tokenized to sentences (sent_tokenize)
  2. Sentences tokenized into word tokens (word_tokenize)
  3. Part-of-speech tagging (pos_tag)
  4. Inbuilt NLTK stopwords filtering (nltk.corpus.stopwords)
  5. Figure detectors evaluated per sentence:
     - Simile: Syntactic pattern & POS verification
     - Alliteration: CMU Pronouncing Dictionary phoneme onset clustering
     - Oxymoron: WordNet antonymy graph & curated contradiction lexicon
     - Onomatopoeia: Acoustic sound-symbolism matching with WordNet lemmatization
"""

import string
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Set, Any
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import cmudict, wordnet as wn, stopwords
from nltk.stem import WordNetLemmatizer

def ensure_nltk_corpora():
    checks = {
        'punkt': 'tokenizers/punkt',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
        'cmudict': 'corpora/cmudict',
        'wordnet': 'corpora/wordnet',
        'omw-1.4': 'corpora/omw-1.4',
        'stopwords': 'corpora/stopwords'
    }
    for res, path in checks.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(res, quiet=True)
            except Exception:
                pass

ensure_nltk_corpora()

try:
    CMU_DICT = cmudict.dict()
except Exception:
    CMU_DICT = {}

try:
    STOPWORDS: Set[str] = set(stopwords.words('english'))
except Exception:
    STOPWORDS = set()

LEMMATIZER = WordNetLemmatizer()


FIGURE_CONFIG = {
    "Simile": {
        "color": "#10b981",       
        "bg_color": "#d1fae5",
        "border_color": "#6ee7b7",
        "text_color": "#065f46",
        "badge_bg": "#059669",
        "method": "Syntactic Pattern & POS-Tagging Analysis",
        "description": "An explicit comparison between two distinct entities using 'like' or 'as'."
    },
    "Alliteration": {
        "color": "#2563eb",      
        "bg_color": "#dbeafe",
        "border_color": "#93c5fd",
        "text_color": "#1e40af",
        "badge_bg": "#1d4ed8",
        "method": "Phonemic Analysis via CMU Pronouncing Dictionary",
        "description": "Repetition of identical initial consonant sounds in adjacent or closely spaced words."
    },
    "Oxymoron": {
        "color": "#d97706",       
        "bg_color": "#fef3c7",
        "border_color": "#fcd34d",
        "text_color": "#92400e",
        "badge_bg": "#b45309",
        "method": "WordNet Antonymy Graph & Curated Contradiction Lexicon",
        "description": "Juxtaposition of two apparently contradictory or incongruous terms."
    },
    "Onomatopoeia": {
        "color": "#7c3aed",       
        "bg_color": "#ede9fe",
        "border_color": "#c4b5fd",
        "text_color": "#5b21b6",
        "badge_bg": "#6d28d9",
        "method": "Sound-Symbolic Lexical Matching & Morphological Lemmatization",
        "description": "A word whose phonetic structure imitates or suggests the sound it depicts."
    }
}



ONOMATOPOEIA_LEXICON: Set[str] = {
    
    "bang", "boom", "buzz", "click", "clash", "crash", "hiss", "hum", 
    "mew", "moo", "pop", "quack", "roar", "rustle", "screech", "slap", 
    "splash", "squeak", "thud", "thump", "tick", "whack", "whisper", 
    "zap", "zoom", "zip", "giggle", "murmur", "snort", "splat", "bam",
    
    "clang", "clank", "clatter", "clink", "crunch", "ding", "dong", "knock",
    "plink", "plop", "rattle", "slam", "smash", "snap", "tap", "zing", "pow",
    "splatter", "drip", "drop", "gush", "trickle", "fizz", "sizzle", "crackle",
    "bubble", "slurp", "gargle", "swish", "swoosh", "whoosh", "whistle", "vroom",
    "bark", "woof", "meow", "purr", "baa", "bleat", "neigh", "oink", "grunt",
    "caw", "chirp", "tweet", "croak", "ribbit", "howl", "growl", "snarl",
    "gasp", "sigh", "chuckle", "sob", "shriek", "patter"
}

KNOWN_OXYMORONS: Set[Tuple[str, str]] = {
    
    ("deafening", "silence"), ("jumbo", "shrimp"), ("open", "secret"),
    ("bittersweet", ""), ("living", "dead"), ("original", "copy"),
    ("act", "naturally"), ("awfully", "good"), ("virtual", "reality"),
    
    ("alone", "together"), ("cruel", "kindness"), ("clearly", "confused"),
    ("passive", "aggressive"), ("cold", "fire"), ("dark", "light"),
    ("controlled", "chaos"), ("pretty", "ugly"), ("organized", "mess"),
    ("sweet", "sorrow"), ("calm", "chaos"), ("loud", "silence"),
    ("peaceful", "war"), ("blind", "sight"), ("found", "missing"),
    ("definite", "maybe"), ("foolish", "wisdom"), ("exact", "estimate"),
    ("silent", "scream"), ("growing", "smaller"), ("random", "order"),
    ("same", "difference"), ("unbiased", "opinion"), ("true", "fiction"),
    ("friendly", "fire"), ("honest", "thief"), ("heavy", "lightness")
}

VOWEL_PHONEMES = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}

@dataclass
class DetectedFigure:
    figure_type: str
    phrase: str
    start: int
    end: int
    sentence: str
    method: str
    explanation: str

    def to_dict(self) -> Dict:
        return asdict(self)


def preprocess_text(text: str) -> List[Dict[str, Any]]:
   
    sentences = sent_tokenize(text)
    processed_corpus = []

    for sent in sentences:
        tokens = word_tokenize(sent)
        pos_tags = pos_tag(tokens)
        processed_corpus.append({
            "sentence": sent,
            "tokens": tokens,
            "pos_tags": pos_tags
        })

    return processed_corpus


def detect_similes(pos_tags_or_text) -> List[Dict[str, Any]]:
    
    if isinstance(pos_tags_or_text, str):
        corpus = preprocess_text(pos_tags_or_text)
        results = []
        for item in corpus:
            results.extend(detect_similes(item["pos_tags"]))
        return results

    pos_tags = pos_tags_or_text
    similes = []
    tokens = [t[0] for t in pos_tags]
    tags = [t[1] for t in pos_tags]

    for i in range(len(pos_tags)):
        word = tokens[i].lower()

        
        if word == 'like' and i + 1 < len(pos_tags):
            
            if i > 0 and tokens[i - 1].lower() in {"i", "you", "we", "they", "who", "which", "would", "could", "should", "do", "does", "did"}:
                continue

            determiners = {"a", "an", "the", "any", "some", "his", "her", "their", "our", "its"}
            if tokens[i + 1].lower() in determiners and i + 2 < len(pos_tags):
                matched = f"like {tokens[i + 1]} {tokens[i + 2]}"
            else:
                matched = f"like {tokens[i + 1]}"

            similes.append({
                "figure": "Simile",
                "matched_pattern": matched,
                "index": i,
                "method": "POS Rule",
                "explanation": f"Prepositional comparative simile introducing a figurative vehicle via '{matched}'."
            })

        elif word == 'as' and i + 2 < len(pos_tags):
            if tokens[i + 2].lower() == 'as' and (tags[i + 1].startswith('JJ') or tags[i + 1].startswith('RB')):
                if i + 4 < len(pos_tags) and tokens[i + 3].lower() in {"a", "an", "the"}:
                    matched = f"as {tokens[i + 1]} as {tokens[i + 3]} {tokens[i + 4]}"
                elif i + 3 < len(pos_tags) and tokens[i + 3].isalpha():
                    matched = f"as {tokens[i + 1]} as {tokens[i + 3]}"
                else:
                    matched = f"as {tokens[i + 1]} as"

                similes.append({
                    "figure": "Simile",
                    "matched_pattern": matched,
                    "index": i,
                    "method": "POS Rule",
                    "explanation": f"Correlative equative simile comparing an attribute via '{matched}'."
                })

        elif word == 'as' and i + 1 < len(pos_tags) and tokens[i + 1].lower() in {'if', 'though'}:
            matched = f"as {tokens[i + 1]}"
            similes.append({
                "figure": "Simile",
                "matched_pattern": matched,
                "index": i,
                "method": "POS Rule",
                "explanation": f"Hypothetical comparative simile established via subordinating conjunction '{matched}'."
            })

    return similes


def get_initial_phoneme(word: str) -> Optional[str]:
   
    clean_w = word.lower().strip(string.punctuation)
    if not clean_w:
        return None

    if clean_w in CMU_DICT:
        first_phone = CMU_DICT[clean_w][0][0]
        base_phone = first_phone.rstrip('0123456789')
        if base_phone not in VOWEL_PHONEMES:
            return base_phone
        return None

   
    for digraph, sound in [("ph", "F"), ("th", "TH"), ("sh", "SH"), ("ch", "CH"), ("wh", "W"), ("kn", "N"), ("wr", "R")]:
        if clean_w.startswith(digraph):
            return sound

    first_char = clean_w[0]
    if first_char.isalpha() and first_char not in "aeiou":
        return first_char.upper()
    return None


def detect_alliteration(tokens_or_text, min_consecutive: int = 2) -> List[Dict[str, Any]]:

    if isinstance(tokens_or_text, str):
        corpus = preprocess_text(tokens_or_text)
        results = []
        for item in corpus:
            results.extend(detect_alliteration(item["tokens"], min_consecutive))
        return results

    tokens = tokens_or_text
    alliterations = []
    current_streak = []
    current_phoneme = None

    for word in tokens:
        if not word.isalpha():
            continue

        if word.lower() in STOPWORDS:
            continue

        phoneme = get_initial_phoneme(word)
        if not phoneme:
            continue

        if phoneme == current_phoneme:
            current_streak.append(word)
        else:
            if len(current_streak) >= min_consecutive:
                alliterations.append({
                    "figure": "Alliteration",
                    "matched_phrase": " ".join(current_streak),
                    "phoneme": current_phoneme,
                    "method": "POS Rule",
                    "explanation": f"Consonant repetition with onset phoneme /{current_phoneme}/ across: {' '.join(current_streak)}."
                })
            current_streak = [word]
            current_phoneme = phoneme

    if len(current_streak) >= min_consecutive:
        alliterations.append({
            "figure": "Alliteration",
            "matched_phrase": " ".join(current_streak),
            "phoneme": current_phoneme,
            "method": "POS Rule",
            "explanation": f"Consonant repetition with onset phoneme /{current_phoneme}/ across: {' '.join(current_streak)}."
        })

    return alliterations


def check_wordnet_antonym(word1: str, word2: str) -> bool:
    
    syns1 = wn.synsets(word1)
    syns2 = wn.synsets(word2)

    for s1 in syns1:
        for lemma in s1.lemmas():
            for ant in lemma.antonyms():
                ant_name = ant.name().split('.')[0].lower()
                for s2 in syns2:
                    if ant_name == s2.name().split('.')[0].lower() or ant_name == word2.lower():
                        return True
    return False


def detect_oxymorons(pos_tags_or_text) -> List[Dict[str, Any]]:
    
    if isinstance(pos_tags_or_text, str):
        corpus = preprocess_text(pos_tags_or_text)
        results = []
        for item in corpus:
            results.extend(detect_oxymorons(item["pos_tags"]))
        return results

    pos_tags = pos_tags_or_text
    oxymorons = []
    tokens = [t[0] for t in pos_tags]
    tags = [t[1] for t in pos_tags]

    for i in range(len(pos_tags) - 1):
        w1, t1 = tokens[i].lower(), tags[i]
        w2, t2 = tokens[i + 1].lower(), tags[i + 1]

        if (w1, "") in KNOWN_OXYMORONS or w1 == "bittersweet":
            oxymorons.append({
                "figure": "Oxymoron",
                "matched_phrase": tokens[i],
                "method": "Lexicon Match",
                "explanation": f"Compound literary contradiction in word '{tokens[i]}'."
            })

        if (w1, w2) in KNOWN_OXYMORONS:
            oxymorons.append({
                "figure": "Oxymoron",
                "matched_phrase": f"{tokens[i]} {tokens[i + 1]}",
                "method": "Lexicon Match",
                "explanation": f"Curated literary oxymoron pairing '{tokens[i]}' and '{tokens[i + 1]}'."
            })
            continue

        if (t1.startswith('JJ') and t2.startswith('NN')) or (t1.startswith('JJ') and t2.startswith('JJ')):
            if check_wordnet_antonym(w1, w2):
                oxymorons.append({
                    "figure": "Oxymoron",
                    "matched_phrase": f"{tokens[i]} {tokens[i + 1]}",
                    "method": "WordNet Antonym",
                    "explanation": f"WordNet antonymous semantic opposition between '{tokens[i]}' and '{tokens[i + 1]}'."
                })

    return oxymorons


def detect_onomatopoeia(tokens_or_text) -> List[Dict[str, Any]]:

    if isinstance(tokens_or_text, str):
        corpus = preprocess_text(tokens_or_text)
        results = []
        for item in corpus:
            results.extend(detect_onomatopoeia(item["tokens"]))
        return results

    tokens = tokens_or_text
    matches = []

    for word in tokens:
        clean_word = word.lower().strip(string.punctuation)
        if not clean_word:
            continue

        matched_base = None
        if clean_word in ONOMATOPOEIA_LEXICON:
            matched_base = clean_word
        else:
            for pos_tag in ['v', 'n', 'a']:
                lemma = LEMMATIZER.lemmatize(clean_word, pos=pos_tag)
                if lemma in ONOMATOPOEIA_LEXICON:
                    matched_base = lemma
                    break

        if matched_base:
            matches.append({
                "figure": "Onomatopoeia",
                "matched_phrase": word,
                "base_word": matched_base,
                "method": "Lexicon Match",
                "explanation": f"Acoustic sound imitation derived from base lexeme '{matched_base}'."
            })

    return matches




def analyze_paragraph_full(paragraph: str) -> List[Dict[str, Any]]:
    
    corpus = preprocess_text(paragraph)
    full_results = []

    for item in corpus:
        similes = detect_similes(item["pos_tags"])
        alliterations = detect_alliteration(item["tokens"])
        oxymorons = detect_oxymorons(item["pos_tags"])
        onomatopoeia = detect_onomatopoeia(item["tokens"])

        combined_detections = similes + alliterations + oxymorons + onomatopoeia

        full_results.append({
            "sentence": item["sentence"],
            "detections": combined_detections
        })

    return full_results


def analyze_all_figures(text: str) -> List[DetectedFigure]:
   
    if not text or not text.strip():
        return []

    full_results = analyze_paragraph_full(text)
    detected_figures: List[DetectedFigure] = []

    for res in full_results:
        sent = res["sentence"]
        sent_start = text.find(sent)
        if sent_start == -1:
            sent_start = 0

        for d in res["detections"]:
            fig_type = d["figure"]
            phrase = d.get("matched_pattern") or d.get("matched_phrase") or ""
            if not phrase:
                continue

            phrase_start_in_sent = sent.lower().find(phrase.lower())
            if phrase_start_in_sent != -1:
                abs_start = sent_start + phrase_start_in_sent
                abs_end = abs_start + len(phrase)
                matched_slice = text[abs_start:abs_end]
            else:
               
                words = phrase.split()
                if len(words) >= 2:
                    w_first, w_last = words[0], words[-1]
                    p1 = sent.lower().find(w_first.lower())
                    p2 = sent.lower().rfind(w_last.lower())
                    if p1 != -1 and p2 != -1 and p2 >= p1:
                        abs_start = sent_start + p1
                        abs_end = sent_start + p2 + len(w_last)
                        matched_slice = text[abs_start:abs_end]
                    else:
                        abs_start = sent_start
                        abs_end = sent_start + len(sent)
                        matched_slice = phrase
                else:
                    abs_start = sent_start
                    abs_end = sent_start + len(sent)
                    matched_slice = phrase

            method = d.get("method") or FIGURE_CONFIG.get(fig_type, {}).get("method", "Classical NLP")
            explanation = d.get("explanation") or f"Detected {fig_type} in text."

            detected_figures.append(DetectedFigure(
                figure_type=fig_type,
                phrase=matched_slice,
                start=abs_start,
                end=abs_end,
                sentence=sent,
                method=method,
                explanation=explanation
            ))


    detected_figures.sort(key=lambda x: (x.start, -(x.end - x.start)))
    return detected_figures


def render_highlighted_text(text: str, detections: List[DetectedFigure]) -> str:

    if not text:
        return ""
    if not detections:
        return f'<div class="annotated-text-content">{_escape_html(text)}</div>'

    priority = {"Oxymoron": 4, "Simile": 3, "Onomatopoeia": 2, "Alliteration": 1}
    sorted_candidates = sorted(
        detections,
        key=lambda d: (-priority.get(d.figure_type, 0), -(d.end - d.start), d.start)
    )

    resolved_spans: List[DetectedFigure] = []
    occupied_indices = set()

    for item in sorted_candidates:
        item_indices = set(range(item.start, item.end))
        if not (item_indices & occupied_indices):
            resolved_spans.append(item)
            occupied_indices.update(item_indices)

    resolved_spans.sort(key=lambda d: d.start)

    html_pieces = []
    curr_idx = 0

    for span in resolved_spans:
        if span.start > curr_idx:
            html_pieces.append(_escape_html(text[curr_idx:span.start]))

        config = FIGURE_CONFIG.get(span.figure_type, {})
        bg = config.get("bg_color", "#e2e8f0")
        border = config.get("border_color", "#cbd5e1")
        text_color = config.get("text_color", "#1e293b")
        badge_bg = config.get("badge_bg", "#475569")

        snippet = _escape_html(text[span.start:span.end])
        tooltip = _escape_html(f"{span.figure_type}: {span.explanation}")

        badge_html = (
            f'<span class="fos-highlight" '
            f'title="{tooltip}" '
            f'style="background-color: {bg}; border: 1.5px solid {border}; color: {text_color}; '
            f'padding: 3px 6px; border-radius: 6px; font-weight: 500; display: inline-block; '
            f'margin: 1px 2px; line-height: 1.6; transition: all 0.2s ease;">'
            f'{snippet}'
            f'<span style="background-color: {badge_bg}; color: #ffffff; font-size: 0.70rem; '
            f'padding: 1px 6px; border-radius: 9999px; margin-left: 5px; font-weight: 600; '
            f'vertical-align: middle; text-transform: uppercase; letter-spacing: 0.03em;">'
            f'{span.figure_type}'
            f'</span></span>'
        )
        html_pieces.append(badge_html)
        curr_idx = span.end

    if curr_idx < len(text):
        html_pieces.append(_escape_html(text[curr_idx:]))

    joined_html = "".join(html_pieces)
    joined_html = joined_html.replace("\n", "<br>")

    return f'<div class="annotated-text-content" style="font-size: 1.05rem; line-height: 1.9; color: #1e293b;">{joined_html}</div>'


def _escape_html(text: str) -> str:
    """Safely escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
    )
