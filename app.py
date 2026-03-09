import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Tuple
from modules.pdf_extractor import extract_text_from_files
from modules.text_preprocessing import preprocess_text
from modules.keyword_extraction import extract_keywords
from modules.summarizer import summarize
from modules.topic_modeling import fit_lda, top_words_per_topic
from modules.similarity_engine import SimilarityEngine
from modules.trend_analyzer import keyword_frequencies, topic_trends
from modules.research_gap_detector import detect_gaps, suggest_ideas
from modules.recommendation_engine import RecommendationEngine
from modules.knowledge_graph import extract_entities, build_graph
from utils.visualization import keyword_bar, topic_distribution, similarity_heatmap, knowledge_graph_figure, keyword_wordcloud

st.set_page_config(page_title="AI Research Intelligence Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1rem; padding-bottom: 1rem; max-width: 1200px;}
    .metric-card {border: 1px solid #E6E6E6; border-radius: 10px; padding: 16px; background: #FAFAFA;}
    .section-title {font-size: 1.4rem; font-weight: 600; margin-top: .5rem;}
    .subtle {color: #666;}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("📚 Navigator")
    pages = [
        "🏠 Home / Project Overview",
        "📥 Upload Research Papers",
        "📝 Paper Summaries",
        "🔑 Keyword Analysis",
        "🧠 Topic Modeling",
        "📈 Research Trends",
        "🔍 Paper Similarity Explorer",
        "🕸️ Knowledge Graph Visualization",
        "⚠️ Research Gap Detection",
        "💡 Research Idea Generator",
    ]
    page = st.radio("Sections", options=pages, label_visibility="collapsed")
    st.divider()
    st.subheader("Settings")
    n_topics = st.slider("Number of topics", 2, 12, 6, 1, help="Controls granularity of topic modeling (LDA).")
    top_k_keywords = st.slider("Top keywords per paper", 5, 30, 15, 1, help="Number of keywords to extract per paper.")

@st.cache_data(show_spinner=False)
def _process(files: List[Tuple[str, bytes]], n_topics: int, top_k: int):
    papers = extract_text_from_files(files)
    names = [p["name"] for p in papers]
    original_texts = [p["text"] for p in papers]
    abstracts = [p["sections"].get("abstract", "") for p in papers]
    preprocessed = [preprocess_text(t) for t in original_texts]
    kw = extract_keywords(preprocessed, top_k=top_k)
    summaries = [summarize(t, max_sentences=5) for t in original_texts]
    lda, vectorizer, doc_topic = fit_lda(preprocessed, n_topics=n_topics)
    topics = top_words_per_topic(lda, vectorizer, top_n=10)
    engine = SimilarityEngine()
    embeddings = engine.encode(preprocessed)
    sim = engine.similarity_matrix(embeddings)
    trend_kw_df = keyword_frequencies(kw)
    trend_topics_df = topic_trends(doc_topic)
    entities = extract_entities(original_texts)
    return {
        "names": names,
        "original_texts": original_texts,
        "abstracts": abstracts,
        "preprocessed": preprocessed,
        "keywords": kw,
        "summaries": summaries,
        "doc_topic": doc_topic,
        "topics": topics,
        "sim": sim,
        "trend_kw_df": trend_kw_df,
        "trend_topics_df": trend_topics_df,
        "entities": entities,
    }

state = st.session_state.get("state")

if page == "🏠 Home / Project Overview":
    st.title("AI Research Paper Intelligence Dashboard")
    st.caption("An end-to-end platform for knowledge extraction, topic discovery, similarity analysis, and research insight visualization.")
    c1, c2, c3, c4 = st.columns(4)
    npapers = len(state["names"]) if state else 0
    ntopics = int(state["doc_topic"].shape[1]) if state and getattr(state["doc_topic"], "size", 0) else 0
    tkeywords = int(sum(len(k) for k in state["keywords"])) if state else 0
    clusters = int(len(set(np.argmax(state["doc_topic"], axis=1)))) if state and getattr(state["doc_topic"], "size", 0) else 0
    with c1:
        st.container().markdown('<div class="metric-card">📄<div class="section-title">Papers</div>', unsafe_allow_html=True)
        st.metric(label="Number of papers", value=npapers, help="Total uploaded research papers.")
    with c2:
        st.container().markdown('<div class="metric-card">🧠<div class="section-title">Topics</div>', unsafe_allow_html=True)
        st.metric(label="Detected topics", value=ntopics, help="Number of latent topics discovered by LDA.")
    with c3:
        st.container().markdown('<div class="metric-card">🔑<div class="section-title">Keywords</div>', unsafe_allow_html=True)
        st.metric(label="Total keywords", value=tkeywords, help="Total extracted keywords across papers.")
    with c4:
        st.container().markdown('<div class="metric-card">🧩<div class="section-title">Clusters</div>', unsafe_allow_html=True)
        st.metric(label="Research clusters", value=clusters, help="Dominant topic clusters across documents.")
    st.divider()
    st.subheader("Overview")
    st.write("Use the sidebar to navigate to upload, summaries, keyword analysis, topic modeling, trends, similarity exploration, knowledge graph, gaps, and idea generation.")
    st.stop()

if page == "📥 Upload Research Papers":
    st.title("Upload Research Papers")
    st.caption("Upload one or more PDFs; the system will extract text and compute insights.")
    uploaded = st.file_uploader("Drop PDFs here", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        st.write(pd.DataFrame([{"file": u.name, "size_kb": round(len(u.getvalue()) / 1024, 1)} for u in uploaded]))
        if st.button("Process Papers", type="primary"):
            with st.spinner("Processing papers..."):
                files = [(u.name, u.read()) for u in uploaded]
                new_state = _process(files, n_topics, top_k_keywords)
                st.session_state["state"] = new_state
            st.success("Upload and analysis complete.")
    else:
        st.info("Select PDF files to begin.")
    st.stop()

if not state and page != "📥 Upload Research Papers":
    st.info("No data found. Go to ‘Upload Research Papers’ to ingest PDFs.")
    st.stop()

names = state["names"]
original_texts = state["original_texts"]
abstracts = state["abstracts"]
preprocessed = state["preprocessed"]
keywords = state["keywords"]
summaries = state["summaries"]
doc_topic = state["doc_topic"]
topics = state["topics"]
sim = state["sim"]
trend_kw_df = state["trend_kw_df"]
trend_topics_df = state["trend_topics_df"]
entities = state["entities"]

if page == "📝 Paper Summaries":
    st.title("Paper Summaries")
    for i, name in enumerate(names):
        title_guess = name
        with st.expander(f"{i+1}. {title_guess}"):
            if abstracts[i]:
                st.markdown("**Abstract**")
                st.write(abstracts[i])
            st.markdown("**Summary**")
            st.write(summaries[i][0] or "No summary available.")
            st.caption("Top keywords")
            kw_df = pd.DataFrame(keywords[i], columns=["keyword", "score"])
            if not kw_df.empty:
                st.dataframe(kw_df, use_container_width=True, height=200)
    st.stop()

if page == "🔑 Keyword Analysis":
    st.title("Keyword Analysis")
    sel = st.selectbox("Select paper", options=list(range(len(names))), format_func=lambda i: names[i], help="View top keywords for a specific paper.")
    c1, c2 = st.columns([2, 1])
    with c1:
        fig_kw = keyword_bar(trend_kw_df, top_n=25)
        st.plotly_chart(fig_kw, use_container_width=True)
    with c2:
        img = keyword_wordcloud(trend_kw_df)
        st.image(img, caption="Keyword Wordcloud", use_column_width=True)
    st.divider()
    st.subheader("Top Keywords for Selected Paper")
    kdf = pd.DataFrame(keywords[sel], columns=["keyword", "score"])
    st.dataframe(kdf, use_container_width=True, height=300)
    st.stop()

if page == "🧠 Topic Modeling":
    st.title("Topic Modeling")
    c1, c2 = st.columns(2)
    with c1:
        if doc_topic.size:
            fig_topics = topic_distribution(doc_topic)
            st.plotly_chart(fig_topics, use_container_width=True)
    with c2:
        words = []
        for k in sorted(topics.keys()):
            words.append(f"Topic {k}: " + ", ".join(topics[k]))
        st.write("\n".join(words))
    if doc_topic.size:
        from sklearn.decomposition import PCA
        comp = min(2, doc_topic.shape[1])
        proj = PCA(n_components=comp, random_state=42).fit_transform(doc_topic)
        dfp = pd.DataFrame({"x": proj[:, 0], "y": proj[:, 1] if comp > 1 else np.zeros(len(proj)), "label": [int(np.argmax(r)) for r in doc_topic], "name": names})
        import plotly.express as px
        fig = px.scatter(dfp, x="x", y="y", color="label", hover_name="name", title="Topic Clusters")
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Each topic is represented by its top words and distribution across documents.")
    st.stop()

if page == "📈 Research Trends":
    st.title("Research Trend Analysis")
    c1, c2 = st.columns(2)
    with c1:
        fig_kw = keyword_bar(trend_kw_df, top_n=20)
        st.plotly_chart(fig_kw, use_container_width=True)
    with c2:
        if not trend_topics_df.empty:
            st.caption("Topic popularity")
            st.bar_chart(trend_topics_df.set_index("topic")["avg_weight"])
    st.divider()
    sums = [sum(score for _, score in klist) for klist in keywords]
    df_line = pd.DataFrame({"paper": names, "keyword_intensity": sums})
    st.line_chart(df_line.set_index("paper"))
    st.stop()

if page == "🔍 Paper Similarity Explorer":
    st.title("Paper Similarity Explorer")
    sc1, sc2 = st.columns([1, 2])
    with sc1:
        sel = st.selectbox("Select paper", options=list(range(len(names))), format_func=lambda i: names[i], key="simsel", help="Choose a paper to view similar ones.")
        reco = RecommendationEngine()
        reco.fit(preprocessed, names)
        recs = reco.recommend(sel, top_k=8)
        st.dataframe(pd.DataFrame(recs, columns=["paper", "similarity"]), use_container_width=True, height=350)
    with sc2:
        fig_sim = similarity_heatmap(sim, names)
        st.plotly_chart(fig_sim, use_container_width=True)
    st.stop()

if page == "🕸️ Knowledge Graph Visualization":
    st.title("Knowledge Graph")
    G = build_graph(names, entities)
    fig = knowledge_graph_figure(G)
    st.plotly_chart(fig, use_container_width=True)
    st.stop()

if page == "⚠️ Research Gap Detection":
    st.title("Research Gap Detection")
    gaps = detect_gaps(doc_topic, trend_kw_df, top_n=5)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Under-explored Topics")
        st.write(pd.DataFrame({"topics": gaps.get("topics", [])}))
    with c2:
        st.subheader("Emerging Keywords")
        st.write(pd.DataFrame({"keywords": gaps.get("keywords", [])}))
    st.stop()

if page == "💡 Research Idea Generator":
    st.title("Research Idea Generator")
    gaps = detect_gaps(doc_topic, trend_kw_df, top_n=5)
    ideas = suggest_ideas(gaps)
    st.write(pd.DataFrame({"ideas": ideas[:5]}))
    st.stop()

