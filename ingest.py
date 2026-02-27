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
    response = client_or.embeddings.create(
        model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
        input=text,
        encoding_format="float",
    )
    return response.data[0].embedding


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

    dataset = load_dataset("squad", split="train[:10]")

    for i, item in enumerate(tqdm(dataset)):
        text = f"Question: {item['question']} Context: {item['context']}"
        try:
            vector = get_embedding(text)
            client.upsert(
                collection_name="squad_collection",
                points=[
                    models.PointStruct(
                        id=i,
                        vector=vector,
                        payload={
                            "context": item["context"],
                            "question": item["question"],
                        },
                    )
                ],
            )
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
