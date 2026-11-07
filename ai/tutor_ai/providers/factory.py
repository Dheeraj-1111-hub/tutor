"""Build the configured provider once at start-up (D27)."""

from ..settings import Settings
from .base import ModelProvider
from .mock import MockProvider
from .ollama import OllamaProvider


class ProviderNotConfigured(RuntimeError):
    pass


def build_provider(settings: Settings) -> ModelProvider:
    if settings.provider == "mock":
        return MockProvider(dimensions=settings.embedding_dimensions)
    if settings.provider == "ollama":
        return OllamaProvider(
            base_url=settings.ollama_url,
            chat_model=settings.ollama_chat_model,
            embedding_model=settings.ollama_embed_model,
            dimensions=settings.embedding_dimensions,
            keep_alive=settings.ollama_keep_alive,
            timeout_seconds=settings.ollama_timeout_seconds,
            connect_timeout=settings.ollama_connect_timeout,
            max_batch_size=settings.ollama_max_batch_size,
        )
    raise ProviderNotConfigured("The hosted provider is chosen before real users (D23) and isn't implemented yet.")
