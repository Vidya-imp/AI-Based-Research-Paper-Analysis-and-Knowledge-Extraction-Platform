from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .text_preprocessing import sentences


def summarize(text: str, max_sentences: int = 5) -> Tuple[str, List[str]]:
    sents = sentences(text)
    if not sents:
        return "", []
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sents)
    scores = np.asarray(X.sum(axis=1)).ravel()
    idx = np.argsort(scores)[::-1][:max_sentences]
    key_sents = [sents[i] for i in sorted(idx)]
    summary = " ".join(key_sents)
    return summary, key_sents

