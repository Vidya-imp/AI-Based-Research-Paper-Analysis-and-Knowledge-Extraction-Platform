from typing import List, Dict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from wordcloud import WordCloud
from io import BytesIO
from PIL import Image


def keyword_bar(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    if df.empty:
        return go.Figure()
    top = df.head(top_n)
    fig = px.bar(top, x="keyword", y="score", title="Keyword Frequency")
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def topic_distribution(doc_topic: np.ndarray) -> go.Figure:
    if doc_topic.size == 0:
        return go.Figure()
    df = pd.DataFrame(doc_topic)
    fig = px.imshow(df.T, aspect="auto", color_continuous_scale="Blues", labels=dict(color="Weight"))
    fig.update_layout(title="Topic Distribution Across Papers", xaxis_title="Paper", yaxis_title="Topic")
    return fig


def similarity_heatmap(sim: np.ndarray, labels: List[str]) -> go.Figure:
    if sim.size == 0:
        return go.Figure()
    fig = px.imshow(sim, x=labels, y=labels, color_continuous_scale="Viridis", labels=dict(color="Similarity"))
    fig.update_layout(title="Paper Similarity Heatmap")
    return fig


def knowledge_graph_figure(G: nx.Graph) -> go.Figure:
    if G.number_of_nodes() == 0:
        return go.Figure()
    pos = nx.spring_layout(G, seed=42, k=0.5)
    edge_x = []
    edge_y = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.5, color="#888"))
    node_x = []
    node_y = []
    text = []
    color = []
    for n, data in G.nodes(data=True):
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        text.append(data.get("name", n))
        label = data.get("label", "")
        color.append("#1f77b4" if label == "paper" else "#ff7f0e")
    node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text", text=text, textposition="top center", marker=dict(size=10, color=color))
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title="Knowledge Graph", showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def keyword_wordcloud(df: pd.DataFrame, width: int = 800, height: int = 400) -> Image.Image:
    if df.empty:
        img = Image.new("RGB", (width, height), color="white")
        return img
    freq = {row["keyword"]: float(row["score"]) for _, row in df.iterrows()}
    wc = WordCloud(width=width, height=height, background_color="white").generate_from_frequencies(freq)
    img = wc.to_image()
    return img

