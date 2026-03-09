from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords(corpus: List[str], top_k: int = 20) -> List[List[Tuple[str, float]]]:
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(corpus)
    feature_names = np.array(vectorizer.get_feature_names_out())
    results: List[List[Tuple[str, float]]] = []
    for i in range(X.shape[0]):
        row = X[i].toarray().ravel()
        if row.sum() == 0:
            results.append([])
            continue
        idx = np.argsort(row)[::-1][:top_k]
        results.append(list(zip(feature_names[idx].tolist(), row[idx].tolist())))
    return results

