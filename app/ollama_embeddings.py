"""
Ollama embedding service using nomic-embed-text model
"""
import logging
import requests
import json
from typing import List
from langchain_core.embeddings import Embeddings

class OllamaEmbeddings(Embeddings):
    """Ollama embedding wrapper for LangChain compatibility using nomic-embed-text"""

    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        """Initialize Ollama embeddings"""
        self.model_name = model_name
        self.base_url = base_url
        logging.info(f"Using Ollama embedding model: {model_name}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        try:
            embeddings = []
            for text in texts:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model_name, "prompt": text},
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                embeddings.append(result["embedding"])
            return embeddings
        except Exception as e:
            logging.error(f"Error embedding documents with Ollama: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result["embedding"]
        except Exception as e:
            logging.error(f"Error embedding query with Ollama: {e}")
            raise

def get_ollama_embeddings() -> OllamaEmbeddings:
    """Get Ollama embeddings instance"""
    return OllamaEmbeddings()