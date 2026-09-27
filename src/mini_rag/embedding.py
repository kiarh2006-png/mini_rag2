import ollama

MODEL_NAME = "bge-m3"


class Embedder:
    """Turns text into vectors using a local Ollama embedding model.

    Documents and queries MUST go through the same model.
    """

    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self._client = ollama.Client()

    def embed_passages(self, texts, batch_size=32):
        """Embed document chunks."""
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embed(model=self.model_name, input=batch)
            vectors.extend(response.embeddings)
        return vectors

    def embed_query(self, text):
        """Embed one user query."""
        response = self._client.embed(model=self.model_name, input=[text])
        return response.embeddings[0]