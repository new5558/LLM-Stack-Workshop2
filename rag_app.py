import streamlit as st
import os
import json
import re
from openai import OpenAI
from qdrant_client import QdrantClient

st.set_page_config(page_title="Workshop 2", layout="wide")
st.title("Workshop 2: Build our own RAG application: RAG")

api_key = os.getenv("OPENROUTER_API_KEY")
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
client = QdrantClient(host=qdrant_host, port=6333)

SYSTEM_PROMT = """
... Edit the prompt so that system perform as RAG agent
# Hint: Check Colab Code
"""

# Initialize OpenAI client for OpenRouter
client_or = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def get_embedding(text):
    """Fetch embedding from OpenRouter using OpenAI SDK."""
    # Hint: nvidia/llama-nemotron-embed-vl-1b-v2:free
    # Hint 2: Check Colab Code


def get_chat_response(prompt):
    """Fetch chat completion from OpenRouter using OpenAI SDK."""
    response = client_or.chat.completions.create(
        model="openrouter/free",
        messages=[
            {"role": "system", "content": SYSTEM_PROMT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def rerank_documents(query, points):
    #"""Rerank retrieved chunks by passing them to the LLM in one go."""

    # Format documents with 0-indexed positional numbers for the LLM
    doc_list = "\n".join(
        [f"[{idx}] {row.payload.get('context')[:300]}..."
         for idx, row in enumerate(points)]
    )

    prompt = f"""
    ... Edit the prompt to rerank the docunments based on the query.
    # Hint: Check Colab Code
    {query}
    {doc_list}
    ...
    """
    response = client_or.chat.completions.create(
        model="openrouter/free",
        messages=[
            {"role": "system", "content": "You are a helpful reranking assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )

    content = response.choices[0].message.content

    # Extract JSON list
    match = re.search(r'\[[\d,\s]+\]', content)

    if match:
        indices = json.loads(match.group())
        reranked = [points[i] for i in indices]
        return reranked

    else:
        print("Reranker returned invalid format. Falling back.")
        return points


def run_rag(query):
    # 1. Retrieval
    vector = ...
    results = client.query_points(
        collection_name="squad_collection", query=vector, limit=5
    )

    # 2. Rerank
    points = rerank_documents(query, [point for point in results.points])
    
    search_result = []
    for i, hit in enumerate(points):
        context = hit.payload.get('context')
        
        # TODO: Add context to search_result

    prompt = f"""
    ... Edit the prompt and add question and context we got from reranker...
    ...
    # Hint: Check Colab Code
    """

    # 3. Generation
    answer = get_chat_response(prompt)

    return answer, context_text


query = st.text_input(
    "Ask a question based on SQuAD data:",
    placeholder="e.g. What is the architecture of Qdrant?",
)

if query:
    if not api_key:
        st.error("Please set OPENROUTER_API_KEY.")
    else:
        with st.spinner("Processing RAG..."):
            answer, context_text = run_rag(query)
            st.subheader("Answer:")
            st.write(answer)
            with st.expander("Show Sources"):
                st.text(context_text)
