# RAG Elastic News Articles Example

This repository demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline for querying recent news articles using **Elasticsearch** and **Large Language Models (LLMs)**. It includes tools for ingesting news articles into Elasticsearch and a Streamlit-based web application for querying the indexed data.

## Features

- **News Article Ingestion**: Scripts to preprocess and index news articles into Elasticsearch with vector embeddings for semantic search.
- **Streamlit Web Application**: A user-friendly interface to query recent news articles and get answers powered by LLMs.
- **Customizable LLMs**: Supports multiple LLMs for answering questions, selectable via the web interface.
- **Semantic Search**: Uses dense vector embeddings to retrieve relevant articles from Elasticsearch.

---

## Components

### 1. **News Article Ingestion**
The [`ingest_news_articles_elastic.py`](rag-elastic-news-articles-example/ingest_news_articles_elastic.py) script handles:
- Preprocessing and embedding news articles.
- Indexing articles into Elasticsearch with metadata and dense vectors.
- Automatic deletion of outdated articles based on a configurable threshold.

### 2. **Streamlit Web Application**
The [`app.py`](rag-elastic-news-articles-example/app.py) script provides:
- A web-based interface for querying news articles.
- Integration with Elasticsearch for retrieving relevant articles.
- LLM-based question answering using the retrieved context.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/meltemi-ai/demos.git
   cd rag-elastic-news-articles-example
   poetry shell
   streamlit run app.py  --server.port 5999