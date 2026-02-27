FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir qdrant-client datasets tqdm requests streamlit openai
COPY . .
# Default command (can be overridden in docker-compose)
CMD ["python", "ingest.py"]
