from typing import List, Dict, Tuple
import numpy as np
import pandas as pd


def keyword_frequencies(keyword_lists: List[List[Tuple[str, float]]]) -> pd.DataFrame:
    freq: Dict[str, float] = {}
    for items in keyword_lists:
        for kw, score in items:
            freq[kw] = freq.get(kw, 0.0) + float(score)
    df = pd.DataFrame(sorted(freq.items(), key=lambda x: x[1], reverse=True), columns=["keyword", "score"])
    return df


def topic_trends(doc_topic: np.ndarray) -> pd.DataFrame:
    if doc_topic.size == 0:
        return pd.DataFrame()
    data = []
    for i, row in enumerate(doc_topic):
        for t, val in enumerate(row):
            data.append({"doc": i, "topic": t, "weight": float(val)})
    df = pd.DataFrame(data)
    summary = df.groupby("topic", as_index=False)["weight"].mean().rename(columns={"weight": "avg_weight"})
    return summary

