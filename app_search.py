import streamlit as st
import os
from openai import OpenAI
from qdrant_client import QdrantClient

st.set_page_config(page_title="Workshop 2", layout="wide")
st.title("Workshop 2: Build our own RAG application: Search")

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


query = st.text_input(
    "Enter your search query:", placeholder="e.g. What is the context of SQuAD?"
)

if query:
    if not api_key:
        st.error("Please set OPENROUTER_API_KEY.")
    else:
        with st.spinner("Searching..."):
            try:
                vector = get_embedding(query)
                results = client.query_points(
                    collection_name="squad_collection", query=vector, limit=3
                )

                if not results or not results.points:
                    st.warning("No results found.")
                else:
                    for i, hit in enumerate(results.points):
                        with st.expander(f"Result {i+1} (Score: {hit.score:.4f})"):
                            st.write(f"**Question:** {hit.payload.get('question')}")
                            st.write(f"**Context:** {hit.payload.get('context')}")
            except Exception as e:
                st.error(f"Error: {e}")
