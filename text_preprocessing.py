from typing import List
import re
import os
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

try:
    import spacy
except Exception:
    spacy = None

_stopwords = None
_nlp = None


def _ensure_nltk():
    base = Path(__file__).resolve().parent.parent / "models" / "nltk_data"
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    os.environ.setdefault("NLTK_DATA", str(base))
    if str(base) not in nltk.data.path:
        nltk.data.path.insert(0, str(base))
    # Punkt tokenizer (newer NLTK may use 'punkt' or 'punkt_tab')
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try:
            nltk.download("punkt", quiet=True, download_dir=str(base))
        except Exception:
            pass
    # Ensure punkt_tab as well for newer NLTK distributions
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True, download_dir=str(base))
        except Exception:
            pass
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True, download_dir=str(base))


def _ensure_spacy():
    global _nlp
    if _nlp is not None:
        return
    if spacy is None:
        _nlp = None
        return
    try:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except Exception:
        _nlp = None


def preprocess_text(text: str) -> str:
    try:
        _ensure_nltk()
    except Exception:
        pass
    _ensure_spacy()
    global _stopwords
    if _stopwords is None:
        try:
            _stopwords = set(stopwords.words("english"))
        except LookupError:
            _stopwords = set()
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = _simple_word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum() and t not in _stopwords]
    if _nlp:
        doc = _nlp(" ".join(tokens))
        tokens = [t.lemma_ for t in doc if t.lemma_ and t.lemma_ != "-PRON-"]
    return " ".join(tokens)


def sentences(text: str) -> List[str]:
    try:
        _ensure_nltk()
        sents = sent_tokenize(text)
    except LookupError:
        sents = _simple_sentence_tokenize(text)
    cleaned = [re.sub(r"\s+", " ", s).strip() for s in sents if s.strip()]
    return cleaned


def _simple_word_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text)


def _simple_sentence_tokenize(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p.strip()]
