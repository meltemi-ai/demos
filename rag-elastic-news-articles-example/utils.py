#!/usr/bin/env python
# coding: utf-8

import dotenv
import os
import traceback
from pathlib import Path
dotenv.load_dotenv( Path.home() / ".env")
from logging import getLogger
import logging
es_logger = logging.getLogger("elastic_transport.transport")
es_logger.setLevel(logging.WARNING)

logger = getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

from elasticsearch import Elasticsearch, helpers
from FlagEmbedding import BGEM3FlagModel

def get_embedding_model(model_name='BAAI/bge-m3', use_fp16=True):
    """
    Get the embedding model.
    Args:
        model_name (str): The name of the model to use.
        use_fp16 (bool): Whether to use fp16 precision.
    Returns:
        embedding_model: The embedding model.
    """
    embedding_model = BGEM3FlagModel(model_name,  use_fp16=use_fp16) 
    return embedding_model

def get_elastic_client(elastic_url, elastic_username=None, elastic_password=None):
    """
    Get the Elasticsearch client.
    Returns:
        es: The Elasticsearch client.
    """
    logger.info(f"Creating es client for {elastic_url}")
    
    es = Elasticsearch(
            hosts=[elastic_url],
            # ca_certs=elastic_certificate_path,
            verify_certs=True,
            max_retries=1,
            # retry_on_timeout=True,
            request_timeout=10,
            # http_auth=(elastic_username, elastic_password)
        )    
    try:
        logger.debug(es.info())
        return es
    except:
        traceback.print_exc()
        return None
    