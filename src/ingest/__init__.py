"""
Módulo de Ingestão do NeuroTriage-AI.

Este pacote contém componentes para ingestão de áudio:
- AudioChunker: Divide áudio em chunks com VAD
- PubSubConsumer: Processa mensagens do Google Cloud Pub/Sub
"""

from src.ingest.audio_chunker import (
    AudioChunker,
    AudioChunk,
    ChunkerConfig,
)
from src.ingest.pubsub_consumer import (
    PubSubConsumer,
    AudioMessage,
    PubSubConfig,
)

__all__ = [
    # Audio Chunker
    "AudioChunker",
    "AudioChunk",
    "ChunkerConfig",
    # Pub/Sub Consumer
    "PubSubConsumer",
    "AudioMessage",
    "PubSubConfig",
]
