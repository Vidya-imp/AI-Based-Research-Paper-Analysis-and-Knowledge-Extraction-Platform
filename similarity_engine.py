from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class SimilarityEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = None
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception:
                self.model = None
        self.embeddings = None

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            X = np.zeros((len(texts), 384), dtype=float)
            return X
        self.embeddings = np.asarray(self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True))
        return self.embeddings

    def similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings is None or len(embeddings) == 0:
            return np.zeros((0, 0))
        sim = cosine_similarity(embeddings)
        return sim

    def most_similar(self, index: int, top_k: int = 5) -> List[Tuple[int, float]]:
        if self.embeddings is None:
            return []
        sim = self.similarity_matrix(self.embeddings)
        scores = sim[index]
        idx = np.argsort(scores)[::-1]
        idx = [i for i in idx if i != index][:top_k]
        return [(i, float(scores[i])) for i in idx]

