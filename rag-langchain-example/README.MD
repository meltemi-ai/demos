# RAG LangChain Example (DEMO)

This project is part of the **DEMOS repository** and serves as an example implementation of a **Retrieval-Augmented Generation (RAG) pipeline** using **LangChain**, **ChromaDB**, **Hugging Face Smolagents**. It is intended as a demonstration and may not be maintained actively.

The project consists of two main scripts:

1. `ingest_documents_langchain.py`: Loads, cleans, and indexes documents into ChromaDB.
2. `question-answering.py`: Retrieves relevant document chunks and uses an LLM to answer questions.

## Features

- Supports `.pdf`, `.docx`, and `.txt` documents
- Preprocesses text by removing unnecessary characters
- Uses **ChromaDB** for semantic search
- Employs **HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5** for embeddings
- Uses **Krikri-DPO** as the language model for answering questions
- Implements a **retriever tool** for fetching relevant document chunks

## Installation

1. Clone the repository and navigate to the project folder:
   ```bash
   git clone https://github.com/meltemi-ai/demos.git
   cd demos/rag-langchain-example
   ```

2. Install dependencies:
   ```bash
   pip install langchain langchain_chroma langchain_huggingface smolagents
   ```

3. Ensure you have `ChromaDB` installed:
   ```bash
   pip install chromadb
   ```

## Usage

### 1. Indexing Documents

Before querying the documents, you need to **ingest and index** them:

1. Place your documents inside the `documents/` directory. A sample .docx file has been provided.
2. Run the ingestion script:
   ```bash
   python ingest_documents_langchain.py
   ```
   This script:
   - Loads documents from `documents/`
   - Cleans and preprocesses the text
   - Splits text into chunks
   - Stores the embeddings in **ChromaDB**

### 2. Running the Question-Answering System

Once the documents are indexed, you can retrieve relevant chunks and ask questions:

```bash
python question-answering.py
```

This script:
- Loads the indexed documents from **ChromaDB**
- Retrieves the most relevant document chunks
- Uses **Krikri-DPO** LLM to generate an answer

## Configuration

The `question-answering.py` script requires an API key and base URL for the Krikri model. The project assumes that the model is hosted in a LiteLLM Gateway. These should be set in a `config.py` file:

```python
KRIKRI_BASE_URL = "https://your-api-url.com"
KRIKRI_API_KEY = "your-api-key"
```

## Example Output

```bash
Loading documents...
Initializing ChromaDB...
Querying ChromaDB: 'Πότε γίνεται το Σεμινάριο Εισαγωγή στα Μεγάλα Γλωσσικά Μοντέλα;'
Απάντηση:
Το σεμινάριο διεξάγεται στις 26 Φεβρουαρίου 2025, στις 10:00 π.μ.
```

## License

This project is licensed under the MIT License.
