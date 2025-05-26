# RAG Web Search Example

This repository demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline. It first attempts to answer queries using a **web search (DuckDuckGo)**. If the web search doesn't yield relevant results, it falls back to querying recent news articles indexed in **Elasticsearch**. The system utilizes **Large Language Models (LLMs)** for generating answers. It includes tools for ingesting news articles into Elasticsearch and a Streamlit-based web application for querying.

## Features

- **Hybrid Search Strategy**: Prioritizes **DuckDuckGo web search** for up-to-date information, with a fallback to a local **Elasticsearch** index of news articles.
- **News Article Ingestion (for Fallback)**: Scripts to preprocess and index news articles into Elasticsearch with vector embeddings for semantic search, used when web search is insufficient.
- **Streamlit Web Application**: A user-friendly interface to query information and get answers powered by LLMs, using context from either web search or Elasticsearch.
- **Customizable LLMs**: Supports multiple LLMs for answering questions, selectable via the web interface.
- **Semantic Search (Elasticsearch Fallback)**: Uses dense vector embeddings to retrieve relevant articles from Elasticsearch when web search does not provide adequate results.

---

## Components

### 1. **Streamlit Web Application**
The `app.py` script provides:
- A web-based interface for querying.
- Primary query resolution using **DuckDuckGo web search**.
- Fallback mechanism to query **Elasticsearch** for indexed news articles if web search results are insufficient or do not meet relevance criteria.
- LLM-based question answering using the retrieved context from the successful search source (either web search or Elasticsearch).

### 2. **News Article Ingestion**
The `ingest_news_articles_elastic.py` script handles the setup for the Elasticsearch fallback:
- Preprocessing and embedding news articles.
- Indexing articles into Elasticsearch with metadata and dense vectors.
- Automatic deletion of outdated articles based on a configurable threshold.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/meltemi-ai/demos.git
   cd demos/rag-web-search-example 
   
2. Create virtual environment and install dependencies:
   ```bash
   uv venv
   uv sync 
   ```

## Usage

   
1. Run the streamlit app
   ```bash 
   streamlit run app.py  --server.port 5999 # FIXME
   ```
   
2. Ingest and index document
   ```bash 
   python ingest_news_articles_elastic.py # FIXME
   ```
