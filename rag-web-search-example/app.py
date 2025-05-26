#!/usr/bin/env python
# coding: utf-8
from multiprocessing import context
import streamlit as st
import os
import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 
# or simply:
torch.classes.__path__ = []
from pathlib import Path
import dotenv
dotenv.load_dotenv( Path.home() / ".env")
import openai
from llama_index.core.prompts import PromptTemplate
import logging
from logging import getLogger
from utils import get_elastic_client, get_embedding_model
from duckduckgo_search import DDGS
from trafilatura import fetch_url, extract
from urllib.parse import urlparse

logger = getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

NL="\n"
TAB="\t"

es_logger = logging.getLogger("elastic_transport.transport")
es_logger.setLevel(logging.WARNING)
elastic_url = os.getenv("ELASTIC_URL")
elastic_username = os.getenv("ELASTIC_USERNAME")
elastic_password = os.getenv("ELASTIC_PASSWORD")
elastic_index = os.getenv("ELASTIC_INDEX")
elastic_certificate_path = os.getenv("ELASTIC_CERTIFICATE_PATH")
logger.debug (f"{elastic_url}, {elastic_username}, {elastic_password}, {elastic_index}, {elastic_certificate_path}") 

api_key=os.environ["LLM_PROXY_ILSP_EVAL_API_KEY"]
base_url=os.environ["LLM_PROXY_ILSP_BASE_URL"]

client = openai.OpenAI(
    api_key=api_key,
    base_url=base_url,
)
DEFAULT_PROMPT = "Είσαι ένα μοντέλο Τεχνητής Νοημοσύνης κατάλληλα εκπαιδευμένο για να βοηθάει την χρήστρια ή τον χρήστη τόσο σε άπταιστα ελληνικά όσο και σε άπταιστα αγγλικά."

DEFAULT_MODEL="krikri-dpo-latest"


def generate_text(
    client,
    model=DEFAULT_MODEL,    
    system_prompt=DEFAULT_PROMPT ,
    user_prompt="",
    temperature=0.5,
    max_tokens=5000,
    top_p=0.9,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    
):
    """
    Generates text using the litellm library.

    Args:
        model (str): The language model to use.
        system_prompt (str): The system prompt for the model.
        user_prompt (str): The user's input prompt.
        temperature (float): Controls the randomness of the output.
        max_tokens (int): Limits the length of the generated response.
        top_p (float): Controls nucleus sampling.
        frequency_penalty (float): Penalizes repeated tokens.
        presence_penalty (float): Penalizes new tokens.

    Returns:
        str: The generated text, or None if there was an error.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        return None


def get_question_embedding(question, embed_model, max_length=256):
    """
    Get the embedding for the question. 
    """
    question_embedding = embed_model.encode(question, 
                                 batch_size=12, 
                                 max_length=max_length, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                 )['dense_vecs']
    return question_embedding



def get_response(es, elastic_index, question_embedding):
    """
    Get the response from Elasticsearch based on the question embedding.
    """
    response = es.search(
        index=elastic_index,
        knn={
            "field": "content_vector",
            "query_vector": question_embedding,
            "k": 3,
            "num_candidates": 100,
            #"similarity": 50,
        },
    )
    return response


def get_context_str(response):
    """
    Get the context string from the response.
    """
    context = list()
    for hit in response["hits"]["hits"]:
        #context.append("Title: " + hit["_source"]["metadata.title"]) 
        context.append("Text: " + hit["_source"]["text"])
        
        context.append("Πηγή: " + hit["_source"]["metadata.website"])
        context.append("URL: " + hit["_source"]["metadata.url"])
        context.append("---------------------------")
    return NL.join(context)


prompt_template = PromptTemplate(
   "Είσαι ένα μοντέλο Τεχνητής Νοημοσύνης κατάλληλα εκπαιδευμένο για να βοηθάει την χρήστρια ή τον χρήστη τόσο σε άπταιστα ελληνικά όσο και σε άπταιστα αγγλικά."
   "Καθήκον σου είναι να απαντάς ερωτήσεις που σχετίζονται με την τρέχουσα επικαιρότητα με βάση μόνο τα ειδησεογραφικά άρθρα που σου παρέχονται παρακάτω."   
   "Μην χρησιμοποιήσεις καμιά άλλη προηγούμενη γνώση ή εξωτερική πληροφορία.\n"
   "---------------------\n"
   "Άρθρα:\n"
   "{context_str}\n"
   "---------------------\n"
   "Ερώτηση: {query_str}\n"
   "Οδηγίες:\n"
   "1. Διάβασε προσεκτικά τα ειδησεογραφικά άρθρα που σου δίνονται για να κατανοήσεις το περιεχόμενο τους.\n"
   "2. Αν τα άρθρα περιέχουν αρκετές πληροφορία για να απαντήσεις την ερώτηση, απάντησε με μια σαφή και συνεκτική απάντηση. Μην αναφέρεις φράσεις όπως 'σύμφωνα με τα άρθρα που παρέχονται ...'. Δώσε μόνο την απάντηση.\n"
   "3. Αν δεν μπορείς να απαντήσεις την ερώτηση με βάση τα άρθρα που παρέχονται, απάντησε 'Δεν μπορώ να απαντήσω στην ερώτηση με βάση τα άρθρα που παρέχονται'.\n"
   "4. Κάθε άρθρο συνοδεύεται από την Πηγή από την οποία προέρχεται το άρθρο και το URL από το οποίο προέρχεται το άρθρο. Αν χρησιμοποιήσεις ένα άρθρο για να συνθέσεις την απάντησή σου, παράθεσε στο τέλος την απάντησης μία λίστα από http links. Σε κάθε http link χρησιμοποίησε ως κείμενο την Πηγή από την οποία προέρχεται το άρθρο και το URL του άρθρου.\n"    
   "5. Επίστρεψε την απάντηση σε Markdown. Μην επιστρέψεις τίποτα άλλο εκτός από την απάντηση και τη λίστα με τα http links. Επίστρεψε τη λίστα με τα http links ως ξεχωριστό markdown section με το όνομα 'Πηγές'.\n"	
   "Απάντηση: "
)

def get_prompt(prompt_template, response, question):
    """
    Get the prompt for the model.
    """
    context_str = get_context_str(response)
    logger.debug(f"## Retrieved context: {context_str}")
    prompt = prompt_template.format(context_str=context_str, query_str=question)    
    return prompt


_embed_model = None

def _get_embedding_model():
    global _embed_model
    if _embed_model is None:
        # Initialize your embeddings model here
        _embed_model = get_embedding_model()
    return _embed_model

def process_question(question, elastic_index, model):
    embed_model = _get_embedding_model()    
    question_embedding = get_question_embedding(question, embed_model)
    response = get_response(es, elastic_index, question_embedding)
    prompt = get_prompt(prompt_template, response, question)
    answer = generate_text(client, model=model, user_prompt=prompt)    
    return answer



es = get_elastic_client(elastic_url) 
elastic_index="web-data"
#print((process_question("Τι δικαιούνται όσοι εργάζονται ανήμερα το Πάσχα;", elastic_index, embed_model, DEFAULT_MODEL)))

page_title = "Check recent news with Krikri and other LLMs"
st.set_page_config(
    page_title=page_title,
    page_icon="https://chat.ilsp.gr/assets/KriKri_Logo-03.svg",
)
st.title(page_title)
question = st.text_input("Enter your question:")

llm_options = os.environ["DEMO_LLMS"].split(",")
selected_llm = st.selectbox("Select an LLM:", llm_options)


def search_duckduckgo(query):
    """
    Perform a search query using DuckDuckGo and return the search results.
    """
    try:
        response = DDGS().text(query, max_results=5)
        logger.debug(response)
        context = list()
        # body_values = [obj['body'] for obj in response]
        logger.info(f"## Retrieved context from duckduckgo")
        for hit in response:
            try:
                logger.info(f"Fetching {hit['href']}")
                downloaded = fetch_url(hit["href"])
                context.append("Title: " + hit["title"]) 
                # output main content and comments as plain text
                context.append("Text: " + extract(downloaded))
                source = urlparse(hit['href']).netloc
                context.append("Πηγή: " + source)
                context.append("URL: " + hit["href"])
                context.append("---------------------------")
            except:
                logger.error(f"Error fetching {hit['href']}")
        return NL.join(context)
    except Exception as ex:
        logger.info(ex)
        return ""

def get_prompt_ddg(prompt_template, context_str, question):
    """
    Get the prompt for the model.
    """
    logger.debug(f"## Retrieved context: {context_str}")
    prompt = prompt_template.format(context_str=context_str, query_str=question)    
    return prompt

def process_question_using_ddg(question):
    with st.spinner(f"Processing question and finding relevant content from the web..."):
        context_str = search_duckduckgo(query=question)
    with st.spinner(f"Generating answer using {selected_llm}..."):
        prompt = get_prompt_ddg(prompt_template, context_str, question)
        answer = generate_text(client, model=selected_llm, user_prompt=prompt)    
        return answer

def process_question_using_elastic(question):
    answer = None
    with st.spinner(f"Processing question using cached content from Elasticsearch..."):
        logger.info(f"## Question: {question}")
        embed_model = _get_embedding_model()    
        question_embedding = get_question_embedding(question, embed_model)
    with st.spinner(f"Finding relevant content..."):
        response = get_response(es, elastic_index, question_embedding)
    with st.spinner(f"Generating answer using {selected_llm}..."):
        prompt = get_prompt(prompt_template, response, question)
        answer = generate_text(client, model=selected_llm, user_prompt=prompt)
    return answer


if st.button("Get Answer"):
    if question:
        try:
            answer = process_question_using_ddg(question)
        except Exception as e:
            # Fallback to elastic
            st.error(f"An error occurred: {e}")
            process_question_using_elastic(question)
        if not answer:
            # Fallback to elastic
            answer = process_question_using_elastic(question)
        if not answer:
            answer = "No answer found"
        logger.info(f"## Answer: {answer}")
        st.subheader(f"Answer from {selected_llm}:")
        st.markdown(answer)
    else:
        st.warning("Please enter a question.")