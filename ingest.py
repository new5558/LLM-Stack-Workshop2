import os
import time
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from datasets import load_dataset
from tqdm import tqdm


# Initialize OpenAI client for OpenRouter
client_or = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def get_embedding(text):
    """Fetch embedding from OpenRouter using OpenAI SDK."""
    # Hint: nvidia/llama-nemotron-embed-vl-1b-v2:free
    # Hint 2: ChatCompletion API


def main():
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")

    client = QdrantClient(host=qdrant_host, port=6333)

    for _ in range(10):
        try:
            client.get_collections()
            break
        except:
            time.sleep(2)

    client.recreate_collection(
        collection_name="squad_collection",
        vectors_config=models.VectorParams(size=2048, distance=models.Distance.COSINE),
    )

    dataset = load_dataset("squad", split="train[:100]")
    context_store = set()
    for i, item in enumerate(tqdm(dataset)):
        context = item['context']
        if context not in context_store:
            context_store.add(context)

    for i, context in enumerate(tqdm(context_store)):
        text = context
        try:
            vector = ...
            client.upsert(
                collection_name="squad_collection",
                points=[
                    models.PointStruct(
                        id=i,
                        vector=vector,
                        payload={
                            "context": text,
                        },
                    )
                ],
            )
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
