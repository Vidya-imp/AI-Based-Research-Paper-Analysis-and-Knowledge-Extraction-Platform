from typing import List, Tuple, Dict
import numpy as np
import pandas as pd


def detect_gaps(doc_topic: np.ndarray, keywords_df: pd.DataFrame, top_n: int = 5) -> Dict[str, List[str]]:
    gaps = {"topics": [], "keywords": []}
    if doc_topic.size:
        avg = doc_topic.mean(axis=0)
        low_idx = np.argsort(avg)[:top_n]
        gaps["topics"] = [f"Topic {int(i)}" for i in low_idx]
    if not keywords_df.empty:
        scores = keywords_df.sort_values("score")
        k = scores.head(top_n)["keyword"].tolist()
        gaps["keywords"] = k
    return gaps


def suggest_ideas(gaps: Dict[str, List[str]]) -> List[str]:
    ideas = []
    for t in gaps.get("topics", []):
        for k in gaps.get("keywords", []):
            ideas.append(f"Investigate {k} within {t.lower()}")
    if not ideas:
        ideas.append("Explore cross-domain applications and benchmarking on new datasets")
    return ideas[:10]

