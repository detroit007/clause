from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field 

from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient


# api/.env
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=ENV_FILE,
        extra="ignore"
    )

    # App
    app_name: str = "Clause"
    data_path: str = "data/shop_handbook.pdf"

    # Chat
    chat_provider: str = "google_genai"
    chat_model: str = "gemini-3.5-flash-lite"
    model_api_key: str | None = None  # loaded from RAG_MODEL_API_KEY in .env
    chat_temperature: float = 0

    # Embed
    embed_provider: str = "ollama"
    embed_model: str = "nomic-embed-text:latest"
    embed_dim: int = 768

    # Indexing
    chunk_size: int = 600
    chunk_overlap: int = 100

    # Retrieval
    top_k: int = 2
    score_threshold: float = 0.7

    # Qdrant
    collection: str = "shop_handbook"  # table
    vector_store: str = "./vector_store" # vector store
    # qdrant_url

    # Prompts
    system_prompt: str = (
        "You are a helpful Clause Shop customer support assistant.\n"
        "Follow these rules strictly:\n"
        "1. Answer ONLY using facts explicitly stated in the provided Context. Do not guess or use outside knowledge.\n"
        "2. Be direct and include exact policy details when relevant (such as days, amounts, fees, conditions, or operating hours), and do not mix up rules across different categories or cities.\n"
        "3. Never share information marked \"Internal\" or \"staff only\" (such as internal review rules, internal teams, or staff extensions). If asked about internal procedures, politely state that you cannot share internal information and suggest contacting customer support.\n"
        "4. If the Context does not answer the question, reply: \"I don't know. The Clause shop handbook doesn't cover that.\""
    )


@lru_cache
def get_settings():
    return Settings()

def get_embeddings(settings: Settings):
    return init_embeddings(
        model=settings.embed_model,
        provider=settings.embed_provider
    )

def get_qdrant_client(settings: Settings):
    return QdrantClient(
        path=settings.vector_store
    )