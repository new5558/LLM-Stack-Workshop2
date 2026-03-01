# Workshop 2: Minimal RAG & Search

This workshop demonstrates how to build a Retrieval-Augmented Generation (RAG) system and a Search engine using **Qdrant**, **OpenRouter**, and **SQuAD** dataset.

## 🚀 Getting Started

### 1. Prerequisites
- Docker & Docker Compose
- OpenRouter API Key

### 2. Configuration
Create a `.env` file in this directory and add your API key:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 3. Run with Docker Compose
The entire stack can be started with:
```bash
docker-compose up --build
```

## TODO
### 1) Make App Search Works
- Fix ingest.py
- Fix app_search.py
- Rerun `docker-compose up --build` until you can do App Search via http://localhost:8501.
- App Rag Service will crash. you can ignore it for this step.
### 2) Make RAG App Works
- Fix rag_app.py
- Rerun `docker-compose up --build` until you can do RAG App via http://localhost:8502

## 🏗️ Services

| Service | Port | Description |
| --------- | ---- | ----------- |
| **Qdrant** | 6333 | The vector database. |
| **Ingest** | N/A | Automatically downloads SQuAD data and uploads embeddings to Qdrant. |
| **Search UI** | 8501 | Simple text search interface (`app_search.py`). |
| **RAG UI** | 8502 | Full RAG interface providing AI-generated answers (`rag_app.py`). |

## 🛠️ Key Components
- **Dataset**: [SQuAD](https://huggingface.co/datasets/squad)
- **Embeddings**: `nvidia/llama-nemotron-embed-vl-1b-v2:free` (2048 dimensions)
- **LLM**: `openrouter/free` via OpenRouter
- **Vector DB**: Qdrant
- **Framework**: Streamlit
