"""
Local embedding service using BGE-M3 model
"""
import logging
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

class BGE_M3_Embeddings(Embeddings):
    """BGE-M3 embedding wrapper for LangChain compatibility"""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """Initialize BGE-M3 model"""
        logging.info(f"Loading BGE-M3 embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            logging.info("BGE-M3 model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load BGE-M3 model: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        try:
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()
        except Exception as e:
            logging.error(f"Error embedding documents: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        try:
            embedding = self.model.encode([text], normalize_embeddings=True)
            return embedding[0].tolist()
        except Exception as e:
            logging.error(f"Error embedding query: {e}")
            raise

def get_local_embeddings() -> BGE_M3_Embeddings:
    """Get BGE-M3 embeddings instance"""
    return BGE_M3_Embeddings()