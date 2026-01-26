"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           Pub/Sub Consumer: Ingestão de Áudio via Google Cloud Pub/Sub      ║
║                                                                              ║
║  Consumer assíncrono para processar mensagens de áudio do Pub/Sub.          ║
║  Integra com o pipeline de triagem para processamento real-time.            ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 O QUE ESTE ARQUIVO FAZ?
---------------------------
1. Conecta ao tópico Pub/Sub configurado
2. Recebe mensagens com chunks de áudio
3. Processa através do pipeline de triagem
4. Gerencia ACK/NACK e dead letter queue

📊 FLUXO:
---------
Pub/Sub Topic -> Consumer -> AudioChunker -> Triage Graph -> Result Storage
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import signal
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Awaitable

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

class PubSubConfig(BaseModel):
    """Configuração do Pub/Sub consumer."""
    
    project_id: str = Field(
        ...,
        description="ID do projeto GCP",
    )
    subscription_id: str = Field(
        ...,
        description="ID da subscription Pub/Sub",
    )
    max_messages: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Máximo de mensagens a buscar por vez",
    )
    ack_deadline_seconds: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Deadline para ACK em segundos",
    )
    max_concurrent: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Processamentos concorrentes máximos",
    )
    
    @classmethod
    def from_env(cls) -> "PubSubConfig":
        return cls(
            project_id=os.getenv("GCP_PROJECT_ID", ""),
            subscription_id=os.getenv("PUBSUB_SUBSCRIPTION_ID", "audio-chunks-sub"),
        )


# ============================================================================
# TIPOS DE MENSAGEM
# ============================================================================

@dataclass
class AudioMessage:
    """
    Mensagem de áudio recebida do Pub/Sub.
    
    Formato esperado do payload JSON:
    {
        "conversation_id": "conv_123",
        "chunk_id": "chunk_001",
        "audio_base64": "...",
        "format": "wav",
        "metadata": {...}
    }
    """
    message_id: str
    conversation_id: str
    chunk_id: str
    audio_bytes: bytes
    format: str
    metadata: dict = field(default_factory=dict)
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @classmethod
    def from_pubsub_message(cls, message) -> "AudioMessage":
        """Cria AudioMessage a partir de mensagem Pub/Sub."""
        # Decodificar payload JSON
        payload = json.loads(message.data.decode("utf-8"))
        
        # Decodificar áudio de base64
        audio_base64 = payload.get("audio_base64", "")
        audio_bytes = base64.b64decode(audio_base64) if audio_base64 else b""
        
        return cls(
            message_id=message.message_id,
            conversation_id=payload.get("conversation_id", "unknown"),
            chunk_id=payload.get("chunk_id", message.message_id),
            audio_bytes=audio_bytes,
            format=payload.get("format", "wav"),
            metadata=payload.get("metadata", {}),
        )


@dataclass
class ProcessingResult:
    """Resultado do processamento de uma mensagem."""
    message_id: str
    success: bool
    error: str | None = None
    processing_time_ms: float = 0.0


# ============================================================================
# CONSUMER
# ============================================================================

class PubSubConsumer:
    """
    Consumer assíncrono de mensagens do Google Cloud Pub/Sub.
    
    Gerencia:
    - Recebimento de mensagens
    - Processamento paralelo com limite de concorrência
    - ACK/NACK automático
    - Graceful shutdown
    
    Example:
        >>> async def process(msg: AudioMessage) -> bool:
        ...     # Processar mensagem
        ...     return True
        ...
        >>> consumer = PubSubConsumer.from_env()
        >>> await consumer.run(process)
    """
    
    def __init__(self, config: PubSubConfig):
        self.config = config
        self._subscriber = None
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._semaphore: asyncio.Semaphore | None = None
        
    @classmethod
    def from_env(cls) -> "PubSubConsumer":
        return cls(PubSubConfig.from_env())
    
    @property
    def subscriber(self):
        """Lazy loading do cliente Pub/Sub."""
        if self._subscriber is None:
            try:
                from google.cloud import pubsub_v1
                self._subscriber = pubsub_v1.SubscriberClient()
            except ImportError as e:
                raise ImportError(
                    "google-cloud-pubsub não instalado. "
                    "Execute: pip install google-cloud-pubsub"
                ) from e
        return self._subscriber
    
    @property
    def subscription_path(self) -> str:
        """Caminho completo da subscription."""
        return self.subscriber.subscription_path(
            self.config.project_id,
            self.config.subscription_id,
        )
    
    async def run(
        self,
        processor: Callable[[AudioMessage], Awaitable[bool]],
    ) -> None:
        """
        Executa o consumer em loop.
        
        Args:
            processor: Função async que processa cada mensagem.
                      Deve retornar True para ACK, False para NACK.
        """
        self._running = True
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        # Configurar signal handlers para graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._request_shutdown)
        
        logger.info(
            "consumer_starting",
            project=self.config.project_id,
            subscription=self.config.subscription_id,
        )
        
        try:
            while self._running:
                await self._pull_and_process(processor)
                
                # Check for shutdown
                if self._shutdown_event.is_set():
                    break
                
                # Pequeno delay entre pulls
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error("consumer_error", error=str(e))
            raise
        finally:
            logger.info("consumer_stopped")
    
    async def _pull_and_process(
        self,
        processor: Callable[[AudioMessage], Awaitable[bool]],
    ) -> None:
        """Busca mensagens e processa."""
        from google.cloud import pubsub_v1
        
        # Pull síncrono (Pub/Sub não tem API async nativa)
        response = self.subscriber.pull(
            request={
                "subscription": self.subscription_path,
                "max_messages": self.config.max_messages,
            },
            timeout=10.0,  # Timeout de pull
        )
        
        if not response.received_messages:
            return
        
        logger.debug("messages_received", count=len(response.received_messages))
        
        # Processar em paralelo com limite de concorrência
        tasks = [
            self._process_message(msg, processor)
            for msg in response.received_messages
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # ACK/NACK baseado nos resultados
        ack_ids = []
        nack_ids = []
        
        for msg, result in zip(response.received_messages, results):
            if isinstance(result, Exception):
                logger.error("message_processing_error", error=str(result))
                nack_ids.append(msg.ack_id)
            elif result:
                ack_ids.append(msg.ack_id)
            else:
                nack_ids.append(msg.ack_id)
        
        # Enviar ACKs
        if ack_ids:
            self.subscriber.acknowledge(
                request={
                    "subscription": self.subscription_path,
                    "ack_ids": ack_ids,
                }
            )
            logger.debug("messages_acked", count=len(ack_ids))
        
        # NACKs (modify ack deadline para 0)
        if nack_ids:
            self.subscriber.modify_ack_deadline(
                request={
                    "subscription": self.subscription_path,
                    "ack_ids": nack_ids,
                    "ack_deadline_seconds": 0,
                }
            )
            logger.debug("messages_nacked", count=len(nack_ids))
    
    async def _process_message(
        self,
        received_message,
        processor: Callable[[AudioMessage], Awaitable[bool]],
    ) -> bool:
        """Processa uma mensagem individual."""
        async with self._semaphore:
            try:
                # Converter para AudioMessage
                audio_msg = AudioMessage.from_pubsub_message(received_message.message)
                
                logger.info(
                    "processing_message",
                    message_id=audio_msg.message_id,
                    conversation_id=audio_msg.conversation_id,
                )
                
                # Processar
                result = await processor(audio_msg)
                
                return result
                
            except Exception as e:
                logger.error(
                    "message_parse_error",
                    message_id=received_message.message.message_id,
                    error=str(e),
                )
                return False
    
    def _request_shutdown(self) -> None:
        """Handler de shutdown graceful."""
        logger.info("shutdown_requested")
        self._running = False
        self._shutdown_event.set()
    
    async def stop(self) -> None:
        """Para o consumer."""
        self._request_shutdown()


# ============================================================================
# PROCESSADOR DE EXEMPLO (Integração com Pipeline)
# ============================================================================

async def create_triage_processor():
    """
    Cria processador que integra com o pipeline de triagem.
    
    Returns:
        Função processadora para usar com PubSubConsumer.run()
    """
    from src.agents.graph import build_triage_graph
    
    graph = build_triage_graph()
    
    async def process(msg: AudioMessage) -> bool:
        """Processa mensagem através do grafo de triagem."""
        try:
            # Preparar estado inicial
            state = {
                "audio_chunk_id": msg.chunk_id,
                "audio_bytes": msg.audio_bytes,
                "conversation_id": msg.conversation_id,
                "iteration_count": 0,
            }
            
            # Executar grafo
            result = await graph.ainvoke(
                state,
                config={"configurable": {"thread_id": msg.conversation_id}},
            )
            
            logger.info(
                "triage_complete",
                conversation_id=msg.conversation_id,
                risk_level=result.get("risk_assessment", {}).get("level", "unknown"),
                emergency=result.get("priority_alert", False),
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "triage_error",
                conversation_id=msg.conversation_id,
                error=str(e),
            )
            return False
    
    return process


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    async def main():
        print("Iniciando Pub/Sub Consumer...")
        print("-" * 50)
        
        consumer = PubSubConsumer.from_env()
        processor = await create_triage_processor()
        
        print(f"Project: {consumer.config.project_id}")
        print(f"Subscription: {consumer.config.subscription_id}")
        print("Aguardando mensagens... (Ctrl+C para parar)")
        print()
        
        await consumer.run(processor)
    
    asyncio.run(main())
