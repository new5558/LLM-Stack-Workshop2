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


# Initialize OpenAI client for OpenRouter
client_or = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def get_embedding(text):
    """Fetch embedding from OpenRouter using OpenAI SDK."""
    ...


def get_chat_response(prompt):
    """Fetch chat completion from OpenRouter using OpenAI SDK."""
    ...


def rerank_documents(query, points):
    """Rerank retrieved chunks by passing them to the LLM in one go."""
    ...


def run_rag(query):
    # 1. Retrieval
    ...

    # 2. Rerank
    ...

    # 3. Generation
    ...

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
            try:
                answer, context_text = run_rag(query)
                st.subheader("Answer:")
                st.write(answer)
                with st.expander("Show Sources"):
                    st.text(context_text)
            except Exception as e:
                st.error(f"Error: {e}")
