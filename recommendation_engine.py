from typing import List, Tuple
from .similarity_engine import SimilarityEngine


class RecommendationEngine:
    def __init__(self):
        self.engine = SimilarityEngine()
        self.embeddings = None
        self.names: List[str] = []

    def fit(self, texts: List[str], names: List[str]):
        self.names = names
        self.embeddings = self.engine.encode(texts)

    def recommend(self, index: int, top_k: int = 5) -> List[Tuple[str, float]]:
        pairs = self.engine.most_similar(index, top_k=top_k)
        return [(self.names[i], score) for i, score in pairs]

