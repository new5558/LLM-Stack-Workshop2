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
    response = client_or.embeddings.create(
        model="nvidia/llama-nemotron-embed-vl-1b-v2:free", 
        input=text, 
        encoding_format="float",
    )
    return response.data[0].embedding


def get_chat_response(prompt):
    """Fetch chat completion from OpenRouter using OpenAI SDK."""
    response = client_or.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def rerank_documents(query, points):
    """Rerank retrieved chunks by passing them to the LLM in one go."""
    if not points:
        return []

    st.info(f"Reranking {len(points)} candidates...")

    # Format candidates for the prompt
    doc_list = "\n".join(
        [f"[{i}] {p.payload.get('context')[:200]}..." for i, p in enumerate(points)]
    )

    prompt = f"""You are an assistant that reranks search results. 
Question: {query}

Results:
{doc_list}

Based on the question, identify which results are most relevant. 
Return ONLY a JSON list of indices in order of relevance, like this: [2, 0, 1]"""

    try:
        response = client_or.chat.completions.create(
            model="openrouter/free",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        # Simple extraction of the list from JSON result
        content = response.choices[0].message.content
        match = re.search(r"\[[\d,\s]+\]", content)
        if match:
            indices = json.loads(match.group())
            return [points[i] for i in indices if i < len(points)]
    except Exception as e:
        st.warning(f"Reranking failed: {e}. Falling back to vector order.")
        return points


def run_rag(query):
    # 1. Retrieval
    vector = get_embedding(query)

    limit = 3
    search_results = client.query_points(
        collection_name="squad_collection", query=vector, limit=limit
    )

    if not search_results or not search_results.points:
        return "No information found in the database.", ""

    points = search_results.points

    # 2. Rerank
    points = rerank_documents(query, points)

    # Take top 3 for the final answer
    top_points = points[:3]
    context_text = "\n\n".join(
        [
            f"Source {i+1}:\n{hit.payload.get('context')}"
            for i, hit in enumerate(top_points)
        ]
    )

    # 3. Generation
    prompt = f"Use the provided context to answer the question.\n\nContext:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
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
            try:
                answer, context_text = run_rag(query)
                st.subheader("Answer:")
                st.write(answer)
                with st.expander("Show Sources"):
                    st.text(context_text)
            except Exception as e:
                st.error(f"Error: {e}")
