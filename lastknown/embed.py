import os
import threading

import numpy as np


class LocalEmbedder:
    """sentence-transformers on this machine (pip install "lastknown[local]")."""

    def __init__(self, name="BAAI/bge-base-en-v1.5", device=None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name, device=device)
        self.lock = threading.Lock()

    def __call__(self, texts):
        with self.lock:
            return list(np.asarray(self.model.encode(list(texts), normalize_embeddings=True, batch_size=64)))


class APIEmbedder:
    """Any OpenAI-compatible /embeddings endpoint (OpenAI, Ollama, vLLM, LM Studio, ...)."""

    def __init__(self, model=None, base_url=None, api_key=None):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url or os.environ.get("LASTKNOWN_EMBED_BASE_URL")
                             or os.environ.get("LASTKNOWN_BASE_URL", "http://localhost:8000/v1"),
                             api_key=api_key or os.environ.get("LASTKNOWN_API_KEY")
                             or os.environ.get("OPENAI_API_KEY", "none"))
        self.model = model or os.environ.get("LASTKNOWN_EMBED_MODEL", "text-embedding-3-small")

    def __call__(self, texts):
        r = self.client.embeddings.create(model=self.model, input=list(texts))
        vecs = np.array([d.embedding for d in r.data], dtype=np.float32)
        return list(vecs / np.linalg.norm(vecs, axis=1, keepdims=True))
