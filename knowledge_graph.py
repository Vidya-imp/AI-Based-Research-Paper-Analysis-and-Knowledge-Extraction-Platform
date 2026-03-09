from typing import List, Dict, Tuple
import networkx as nx
import re

try:
    import spacy
except Exception:
    spacy = None


def _load_ner():
    if spacy is None:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def extract_entities(texts: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    nlp = _load_ner()
    results: Dict[str, List[Tuple[str, str]]] = {}
    if nlp is None:
        for i, t in enumerate(texts):
            results[str(i)] = []
        return results
    for i, t in enumerate(texts):
        doc = nlp(t)
        ents = []
        for e in doc.ents:
            if e.label_ in {"PERSON", "ORG", "GPE", "NORP", "FAC", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"}:
                ents.append((e.text, e.label_))
        ents += _custom_mined_entities(t)
        results[str(i)] = ents
    return results


def build_graph(paper_names: List[str], entities: Dict[str, List[Tuple[str, str]]]) -> nx.Graph:
    G = nx.Graph()
    for idx, name in enumerate(paper_names):
        G.add_node(f"paper::{idx}", label="paper", name=name)
    for idx, name in enumerate(paper_names):
        ents = entities.get(str(idx), [])
        for text, label in ents:
            node_id = f"{label}::{text}"
            if not G.has_node(node_id):
                G.add_node(node_id, label=label, name=text)
            G.add_edge(f"paper::{idx}", node_id)
    return G


def _custom_mined_entities(text: str) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    algos = {"svm", "support vector machine", "random forest", "xgboost", "lightgbm", "naive bayes", "knn", "k-nearest neighbors", "k-means", "dbscan", "pca", "cnn", "rnn", "lstm", "transformer", "bert", "gpt"}
    for a in algos:
        if re.search(rf"\\b{re.escape(a)}\\b", text, flags=re.IGNORECASE):
            items.append((a.title(), "ALGO"))
    datasets = re.findall(r"\\b([A-Z][A-Za-z0-9\\-]+)\\s+dataset\\b", text)
    for d in datasets:
        items.append((d, "DATASET"))
    methods = re.findall(r"\\b([A-Za-z\\-]{3,}\\s+(?:method|approach|technique))\\b", text, flags=re.IGNORECASE)
    for m in methods:
        items.append((m.title(), "METHOD"))
    return items
