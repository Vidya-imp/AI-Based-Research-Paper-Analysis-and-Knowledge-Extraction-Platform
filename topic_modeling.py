from typing import List, Tuple, Dict
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def fit_lda(corpus: List[str], n_topics: int = 5, max_features: int = 5000, max_iter: int = 20, random_state: int = 42) -> Tuple[LatentDirichletAllocation, CountVectorizer, np.ndarray]:
    vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
    X = vectorizer.fit_transform(corpus)
    if X.shape[0] == 0 or X.shape[1] == 0:
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=random_state)
        return lda, vectorizer, np.zeros((len(corpus), n_topics))
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=max_iter, learning_method="batch", random_state=random_state)
    lda.fit(X)
    doc_topic = lda.transform(X)
    return lda, vectorizer, doc_topic


def top_words_per_topic(lda: LatentDirichletAllocation, vectorizer: CountVectorizer, top_n: int = 10) -> Dict[int, List[str]]:
    feature_names = np.array(vectorizer.get_feature_names_out())
    topics = {}
    for idx, comp in enumerate(lda.components_):
        top_idx = np.argsort(comp)[::-1][:top_n]
        topics[idx] = feature_names[top_idx].tolist()
    return topics

