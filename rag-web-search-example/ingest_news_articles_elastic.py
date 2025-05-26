#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import dotenv
from pathlib import Path
import json
import pandas as pd
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch, helpers
from logging import getLogger
import logging
from utils import get_elastic_client, get_embedding_model
dotenv.load_dotenv( Path.home() / ".env")

es_logger = logging.getLogger("elastic_transport.transport")
es_logger.setLevel(logging.WARNING)
elastic_index = os.getenv("ELASTIC_INDEX")
elastic_url = os.getenv("ELASTIC_URL")
elastic_username = os.getenv("ELASTIC_USERNAME")
elastic_password = os.getenv("ELASTIC_PASSWORD")
elastic_certificate_path = os.getenv("ELASTIC_CERTIFICATE_PATH")

logger = getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger.debug (f"{elastic_url}, {elastic_username}, {elastic_password}, {elastic_index}, {elastic_certificate_path}") # We do not use all this if elastic has no security enabled

NL="\n"
TAB="\t"

sources = {'athensvoice', 'avgi', 'capital', 'documento', 'efsyn', 'ert',  'ethnos', 'euro2day', 'fosonline', 'gazzetta', 'imerisia', 'in', 'insider', 'kathimerini', 'liberal', 'lifo', 'marieclaire', 'moneyreview', 'naftemporiki', 'news247', 'newsbeast', 'ot', 'protagon', 'protothema', 'reader', 'real', 'rizospastis', 'skai', 'tanea', 'techgear', 'thepressproject', 'vima' }


index_mapping = {
    "properties": {
        "title_vector": {
            "type": "dense_vector",
            "dims": 1536,
            "index": "true",
            "similarity": "cosine",
        },
        "subtitle_vector": {
            "type": "dense_vector",
            "dims": 1536,
            "index": "true",
            "similarity": "cosine",
        },
        "content_vector": {
            "type": "dense_vector",
            "dims": 1536,
            "index": "true",
            "similarity": "cosine",
        },
        "text": {"type": "text"},
        "metadata.id": {"type": "keyword"},
        "metadata.title": {"type": "text"},
        "metadata.subtitle": {"type": "text"},
        "metadata.url": {"type": "keyword"},
        "metadata.article_id": {"type": "keyword"},
        #"metadata.pub_date": {"type": "date",  "format": "yyyyMMddHHmm"},
        "metadata.website": {"type": "keyword"},
        "metadata.category": {"type": "keyword"},        
#        "vector_id": {"type": "long"},
    }
}



def delete_old_documents(es_client: Elasticsearch, index_name: str, days: int=3, date_field: str="metadata.pub_date"):
    """
    Deletes documents from an Elasticsearch index that are older than a specified number of days
    based on the value of a date field.

    Args:
        es_client: An Elasticsearch client instance.
        index_name: The name of the Elasticsearch index.
        days: The number of days to go back. Documents older than this will be deleted.
        date_field: The name of the date field in your Elasticsearch documents.
    """
    cutoff_date = datetime.now() - timedelta(days=days)

    query = {
        "query": {
            "range": {
                date_field: {
                    "lt": cutoff_date.isoformat()
                }
            }
        }
    }

    try:
        response = es_client.delete_by_query(index=index_name, body=query)
        logger.info(f"Deleted {response['deleted']} documents from index '{index_name}' that are older than {days} days based on the '{date_field}' field.")
        if response['failures']:
            logger.info("Failures:")
            for failure in response['failures']:
                logger.info(failure)
    except Exception as e:
        logger.info(f"An error occurred: {e}")


def get_all_ids_scroll(es_client: Elasticsearch, index_name: str):
    """
    Retrieves a set of all document IDs from an Elasticsearch index using the scroll API.

    Args:
        es_client: An Elasticsearch client instance.
        index_name: The name of the Elasticsearch index.

    Returns:
        A set containing all document IDs in the index.
    """
    all_ids = set()
    page = es_client.search(
        index=index_name,
        scroll='1m',  # Keep the search context alive for 1 minute
        body={
            "query": {"match_all": {}},  # Match all documents
            "_source": False,           # We only need the _id field
            "fields": []                # Ensure no other fields are returned
        },
        size=1000  # Number of hits per scroll page (adjust as needed)
    )
    sid = page['_scroll_id']
    scroll_size = page['hits']['total']['value']

    while (scroll_size > 0):
        for hit in page['hits']['hits']:
            all_ids.add(hit['_id'])

        page = es_client.scroll(scroll_id=sid, scroll='1m')
        sid = page['_scroll_id']
        scroll_size = len(page['hits']['hits'])

    return all_ids


def delete_and_recreate_index(es, elastic_index=elastic_index):
    """
    Delete and recreate the Elasticsearch index.
    Args:
        es: The Elasticsearch client.
        elastic_index (str): The name of the Elasticsearch index.               
    """
    try:
        es.indices.delete(index=elastic_index)
        logger.info(f"Deleted index {elastic_index}")
    except:
        pass
    es.indices.create(index=elastic_index)


def get_jsonl_paths(jsonl_dir=Path.home() / "tmp/fnc/corpus_jsonl/", n=3):
    """
    Get the jsonl file paths for the last n days.
    Args:
        jsonl_dir (str): The directory where the jsonl files are stored.
        n (int): The number of days to go back.
    Returns:
        list: The list of jsonl file paths.
    """
    today = datetime.now().date()
    jsonl_paths = list()
    for i in range(0, n ):
        past_date = today - timedelta(days=i)
        jsonl_path = jsonl_dir / (past_date.strftime("%Y%m%d")+".jsonl")
        if jsonl_path.exists():
            jsonl_paths.append(jsonl_path)
    return jsonl_paths




def get_docs_df(jsonl_paths, sources=sources, already_indexed_ids=set()):
    """
    Get the documents from the jsonl files and return a dataframe.
    Args:
        jsonl_paths (list): The list of jsonl file paths.
        sources (set): The set of sources (website names) to include.
        already_indexed_ids (set): The set of already indexed ids.
    Returns:
        pd.DataFrame: The dataframe containing the documents.
    """
    docs = list()
    for jsonl_path in jsonl_paths:
        with open(jsonl_path) as inp:
            for line in inp:
                doc = json.loads(line)
                if doc["id"] in already_indexed_ids:
                    logger.debug(f"Skipping already indexed {doc['id']}")
                    continue
                if doc["metadata"]["source"] in sources:
                    #print (doc["metadata"]["source"])
                    docs.append(doc)    
    df  = pd.DataFrame(docs)
    return df


def embed_texts_in_df(df, embedding_model, max_length=256):
    """
    Embed the texts in the dataframe using the embeddings model.
    Args:
        df (pd.DataFrame): The dataframe containing the texts to embed.
        embedding_model: The embedding model to use.
        max_length (int): The maximum length of the text to embed.
    Returns:
        pd.DataFrame: The dataframe with the embedded texts.
    """
    text_list = df['text'].tolist()
    content_vectors = embedding_model.encode(text_list, 
                                 batch_size=12, 
                                 max_length=max_length, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                 )['dense_vecs']
    df["content_vector"] = list(content_vectors)

    title_list = list()
    for idx, row in df.iterrows():
        title_list.append(row["metadata"]["title"])
    title_vectors = embedding_model.encode(title_list, 
                                 batch_size=12, 
                                 max_length=max_length,
                                 )['dense_vecs']
    df["title_vector"] = list(title_vectors)
    subtitle_list = list()
    for idx, row in df.iterrows():
        if "subtitle" not in row["metadata"]:
            subtitle_list.append("")    
        else:
            subtitle_list.append(row["metadata"]["subtitle"])
    subtitle_vectors = embedding_model.encode(subtitle_list, 
                                 batch_size=12, 
                                 max_length=max_length,
                                 )['dense_vecs']
    df["subtitle_vector"] = list(subtitle_vectors)
    return df
    
    
def dataframe_to_bulk_actions(df, elastic_index):
    for index, row in df.iterrows():
        source = {
            "metadata.title": row["metadata"]["title"],
            "text": row["text"],
            "metadata.website": row["metadata"].get("source"),  
            "metadata.url": row["metadata"].get("url"),     
            "metadata.category": row["metadata"].get("source_category"), 
            "metadata.pub_date": datetime.strptime(row["metadata"].get("pub_date"), "%Y%m%d%H%M").isoformat(),
            "title_vector": row["title_vector"],
            "content_vector": row["content_vector"],
            # "vector_id": row["vector_id"],
        }

        # Conditionally add subtitle if it exists and is not null/NaN
        subtitle = row["metadata"].get("subtitle")
        if pd.notna(subtitle) and subtitle is not None and subtitle != "":
            source["metadata.subtitle"] = subtitle
            source["subtitle_vector"] = row["subtitle_vector"]

        yield {
            "_index": elastic_index,
            "_id": row["id"],
            "_source": source,
        }           

def ingest_in_elastic(df, es, elastic_index):    
    start = 0
    end = len(df)
    batch_size = 100
    for batch_start in range(start, end, batch_size):
        batch_end = min(batch_start + batch_size, end)
        batch_dataframe = df.iloc[batch_start:batch_end]
        actions = dataframe_to_bulk_actions(batch_dataframe, elastic_index)
        helpers.bulk(es, actions)


def ingest_recent_articles(es, elastic_index, embedding_model=None, rebuild_index=False, age_threshold_days=0, index_age_threshold_days=2):
    """
    Ingests recent articles into Elasticsearch.

    Args:
        es: Elasticsearch client instance.
        elastic_index: Name of the Elasticsearch index.
        embedding_model: The embedding model to use.
        rebuild_index: Whether to delete and recreate the index (default: False).
        age_threshold_days: The age threshold in days for considering articles as recent (default: 3).
        index_age_threshold_days: The age threshold in days for considering articles as recent (default: 2).
    """
    if rebuild_index:
        logger.info("Deleting and recreating ES index")
        delete_and_recreate_index(es, elastic_index)
        logger.info("ES index recreated")
    elif age_threshold_days > 0:
        logger.info(f"Deleting documents older than {age_threshold_days} days from ES index")
        date_field_name = "metadata.pub_date"  
        delete_old_documents(es, elastic_index, age_threshold_days, date_field_name)
    already_indexed_ids = get_all_ids_scroll(es, elastic_index)
    jsonl_paths = get_jsonl_paths(n=index_age_threshold_days)  
    df = get_docs_df(jsonl_paths, sources, already_indexed_ids)
    logger.info(f"Found {len(df)} new docs")
    if len(df) > 0:
        df = embed_texts_in_df(df, embedding_model=embedding_model, max_length=256)
        logger.info(f"Embedded {len(df)} docs")
        ingest_in_elastic(df, es, elastic_index)
    logger.info(f"Ingested {len(df)} docs in {elastic_index}")


if __name__ == "__main__":
    # Load environment variables
    parser = argparse.ArgumentParser(description="Ingest recent articles into Elasticsearch.")
    parser.add_argument("--elastic_index", required=False, help="Name of the Elasticsearch index.")
    parser.add_argument("--embedding_model", default="BAAI/bge-m3", required=False, help="The embedding model to use.")
    parser.add_argument("--rebuild_index", action="store_true",
                        help="Delete and recreate the index.")
    parser.add_argument("--age_threshold", type=int, default=3,
                        help="The age threshold in days for recent articles (default: 3). Older than these articles will be deleted from the index.")
    parser.add_argument("--index_age_threshold", type=int, default=2,
                        help="The index age threshold in days for recent articles (default: 2). Older than these articles will not be indexed.")    
    parser.add_argument("--es_host", default="localhost", required=False, help="Elasticsearch host (default: localhost)")
    parser.add_argument("--es_port", type=int, default=9200, required=False, help="Elasticsearch port (default: 9200)")

    args = parser.parse_args()
    dotenv.load_dotenv( Path.home() / ".env")
    if elastic_index:
        pass
    else:
        elastic_index = args.elastic_index
    if elastic_url:
        pass
    elif args.es_host:
        elastic_url = f"http://{args.es_host}:{args.es_port}"

    logger.info(f"Using elastic_url: {elastic_url}")
    logger.info(f"Ingesting recent articles into index: {elastic_index}")
    logger.info(f"Using embedding model: {args.embedding_model}")
    logger.info(f"Delete and recreate index: {args.rebuild_index}")
    logger.info(f"Age threshold (days): {args.age_threshold}")
    logger.info(f"Index age threshold (days): {args.index_age_threshold}")

    es = get_elastic_client(elastic_url=elastic_url)
    embedding_model=get_embedding_model(model_name=args.embedding_model)
    # Ingest recent articles
    ingest_recent_articles(es, elastic_index, embedding_model=embedding_model, rebuild_index=args.rebuild_index, age_threshold_days=args.age_threshold, index_age_threshold_days=args.index_age_threshold)
