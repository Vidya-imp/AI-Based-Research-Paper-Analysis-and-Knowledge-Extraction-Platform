# AI-Driven Research Paper Intelligence Platform

## Installation
1. Create and activate a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Download NLTK data: `python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"`
4. Download spaCy model: `python -m spacy download en_core_web_sm`

## Run
`streamlit run app.py`

## Workflow
1. Upload PDFs
2. The system extracts text and preprocesses
3. Generates keywords, topics, summaries
4. Computes similarity and recommendations
5. Builds trends, gaps, and ideas
6. Visualizes with interactive dashboard

## Project Structure
```
research_ai_platform/
  app.py
  modules/
    pdf_extractor.py
    text_preprocessing.py
    keyword_extraction.py
    topic_modeling.py
    summarizer.py
    similarity_engine.py
    trend_analyzer.py
    research_gap_detector.py
    recommendation_engine.py
    knowledge_graph.py
  utils/
    visualization.py
    data_cleaning.py
  data/
    sample_papers/
  models/
  requirements.txt
  README.md
```

## Notes
- If embeddings model fails to load, the app falls back to zeros. Install sentence-transformers for best results.
- For large PDFs, processing may take time; Streamlit caches intermediate results.
