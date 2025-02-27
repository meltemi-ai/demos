import os
import re

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5")

def preprocess(txt):
    txt = txt.replace("-\n", "")  # no word breaks
    txt = txt.replace("\n", " ")
    txt = txt.replace("..........", "_")
    txt = re.sub(r"\s+", " ", txt)
    txt = re.sub(r"\.+", ".", txt)
    return txt

def is_valid_line(line):
    return ((bool(line.strip()) and not re.fullmatch(r'[0-9\s]+', line)
            and not re.fullmatch(r'[^\w\s]+', line)
            and not re.fullmatch(r'([^\w\s])\1{2,}', line))
            and not re.search(r'Σελ\.\s*\d+\s*/\s*\d+', line))


def clean_documents(documents):
    cleaned_documents = []
    for doc in documents:
        if isinstance(doc, tuple):
            page_content, metadata = doc
            cleaned_content = preprocess(page_content)
            cleaned_documents.append(Document(page_content=cleaned_content, metadata=metadata))
        elif isinstance(doc, Document):
            #content, new_metadata = extract_metadata_and_content(doc.page_content)
            #cleaned_content = preprocess(doc.page_content)
            cleaned_content = preprocess(doc.page_content)
            cleaned_documents.append(Document(page_content=cleaned_content))
        else:
            raise ValueError("Unsupported document format")
    return cleaned_documents


def load_documents(dir_path):
    documents = []
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        if file.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            original_documents = loader.load()
            cleaned_documents = clean_documents(original_documents)
            documents.extend(cleaned_documents)
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            original_documents = loader.load()
            cleaned_documents = clean_documents(original_documents)
            documents.extend(cleaned_documents)
        elif file.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = preprocess(f.read())
                documents.append(Document(page_content=text, metadata={"source": file_path}))
    return documents


def vector_store_init(documents, persist_directory="chroma_langchain_db"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    print("ChromaDB: Getting splits from documents...")
    splits = text_splitter.split_documents(documents)
    print("ChromaDB: Loading embeddings model...")
    embeddings_model = get_embeddings_model()
    print("ChromaDB: Initializing database...")
    vector_store = Chroma(
        collection_name="innoHub_store",
        embedding_function=embeddings_model,
        persist_directory=persist_directory
    )
    print("ChromaDB: adding document splits...")
    vector_store.add_documents(splits)
    print("All documents successfully added to ChromaDB!")
    return vector_store

if __name__=="__main__":
    # 1: Load documents
    print("Loading documents...")
    my_documents = load_documents("documents")
    # 2: Initialize ChromaDB and add documents
    print("Initializing ChromaDB...")
    vector_store = vector_store_init(my_documents)
    # 3: Test retrieval
    query = "Σε τί αναφέρονται τα παραπάνω έγγραφα;"
    print(f"Querying ChromaDB: '{query}'")
    results = vector_store.similarity_search(query, k=3)

    for idx, result in enumerate(results):
        print(f"\nDocument {idx + 1}:")
        print(result.page_content[:500])