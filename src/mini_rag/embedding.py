import time

import requests

MODEL_NAME = "bge-m3"
OLLAMA_URL = "http://localhost:11434/api/embed"


class Embedder:
    """Turns text into vectors using a local Ollama embedding model.

    Documents and queries MUST go through the same model.
    """

    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name

    def embed_passages(self, texts, batch_size=8):
        """Embed document chunks."""
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors.extend(self._embed_with_retry(batch))
        return vectors

    def embed_query(self, text):
        """Embed one user query."""
        return self._embed_with_retry([text])[0]

    def _embed_with_retry(self, batch, attempts=6, delay_seconds=15):
        last_error = None
        for attempt in range(attempts):
            try:
                response = requests.post(
                    OLLAMA_URL,
                    json={"model": self.model_name, "input": batch},
                    timeout=120,
                )
                response.raise_for_status()
                return response.json()["embeddings"]
            except Exception as error:
                last_error = error
                print(f"  embedding request failed ({error}); retrying in {delay_seconds}s...")
                time.sleep(delay_seconds)
        raise last_error