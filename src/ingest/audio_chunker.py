"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           Audio Chunker: Divisão Inteligente de Áudio com VAD               ║
║                                                                              ║
║  Divide streams de áudio em chunks processáveis usando Voice Activity       ║
║  Detection (VAD) para identificar pausas naturais na fala.                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 O QUE ESTE ARQUIVO FAZ?
---------------------------
1. Recebe stream de áudio contínuo (consulta de telemedicina)
2. Detecta atividade de voz vs silêncio (VAD)
3. Divide em chunks de ~30s respeitando pausas naturais
4. Gera IDs únicos para rastreamento de cada chunk

📊 POR QUE VAD?
---------------
- Evita cortar no meio de uma palavra/frase
- Chunks mais naturais = transcrição melhor
- Economia de processamento (ignora silêncios longos)
"""

from __future__ import annotations

import hashlib
import io
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator, Iterator

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

class ChunkerConfig(BaseModel):
    """Configuração do audio chunker."""
    
    max_chunk_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=60.0,
        description="Duração máxima de cada chunk em segundos",
    )
    min_chunk_seconds: float = Field(
        default=3.0,
        ge=1.0,
        description="Duração mínima para considerar chunk válido",
    )
    silence_threshold_db: float = Field(
        default=-40.0,
        description="Threshold em dB para considerar silêncio",
    )
    min_silence_ms: int = Field(
        default=500,
        ge=100,
        description="Silêncio mínimo em ms para considerar pausa",
    )
    sample_rate: int = Field(
        default=16000,
        description="Taxa de amostragem esperada (Hz)",
    )
    channels: int = Field(
        default=1,
        description="Número de canais (1=mono, 2=stereo)",
    )


# ============================================================================
# TIPOS DE DADOS
# ============================================================================

@dataclass
class AudioChunk:
    """
    Representa um chunk de áudio pronto para processamento.
    
    Attributes:
        chunk_id: ID único para rastreamento
        audio_bytes: Dados de áudio em bytes (WAV format)
        duration_seconds: Duração do chunk
        start_offset: Posição inicial no áudio original (segundos)
        end_offset: Posição final no áudio original (segundos)
        timestamp: Momento de criação do chunk
        metadata: Dados adicionais (conversation_id, etc.)
    """
    chunk_id: str
    audio_bytes: bytes
    duration_seconds: float
    start_offset: float
    end_offset: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        audio_bytes: bytes,
        duration: float,
        start: float,
        end: float,
        conversation_id: str | None = None,
    ) -> "AudioChunk":
        """Factory method para criar chunk com ID gerado."""
        chunk_id = cls._generate_id(audio_bytes, start)
        metadata = {}
        if conversation_id:
            metadata["conversation_id"] = conversation_id
        
        return cls(
            chunk_id=chunk_id,
            audio_bytes=audio_bytes,
            duration_seconds=duration,
            start_offset=start,
            end_offset=end,
            metadata=metadata,
        )
    
    @staticmethod
    def _generate_id(audio_bytes: bytes, offset: float) -> str:
        """Gera ID único baseado em hash do conteúdo."""
        content_hash = hashlib.sha256(audio_bytes[:1000]).hexdigest()[:8]
        unique = uuid.uuid4().hex[:8]
        return f"chunk_{int(offset)}s_{content_hash}_{unique}"


# ============================================================================
# CHUNKER PRINCIPAL
# ============================================================================

class AudioChunker:
    """
    Divisor de áudio em chunks usando VAD.
    
    Suporta:
    - Áudio completo (batch)
    - Stream de áudio (real-time)
    - Múltiplos formatos via pydub
    
    Example:
        >>> chunker = AudioChunker()
        >>> for chunk in chunker.chunk_audio(audio_bytes):
        ...     process(chunk)
    """
    
    def __init__(self, config: ChunkerConfig | None = None):
        """
        Args:
            config: Configuração do chunker
        """
        self.config = config or ChunkerConfig()
        self._pydub_available = self._check_pydub()
        
    def _check_pydub(self) -> bool:
        """Verifica se pydub está disponível."""
        try:
            from pydub import AudioSegment
            return True
        except ImportError:
            logger.warning("pydub_not_available")
            return False
    
    def chunk_audio(
        self,
        audio_bytes: bytes,
        format: str = "wav",
        conversation_id: str | None = None,
    ) -> Iterator[AudioChunk]:
        """
        Divide áudio em chunks.
        
        Args:
            audio_bytes: Áudio completo em bytes
            format: Formato do áudio (wav, mp3, etc.)
            conversation_id: ID da conversa para metadados
            
        Yields:
            AudioChunk para cada segmento
        """
        if not self._pydub_available:
            # Fallback: chunk por tempo fixo sem VAD
            yield from self._chunk_simple(audio_bytes, conversation_id)
            return
        
        from pydub import AudioSegment
        from pydub.silence import split_on_silence
        
        logger.info(
            "chunking_audio",
            size_bytes=len(audio_bytes),
            format=format,
        )
        
        # Carregar áudio
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=format)
        
        # Configurar para mono 16kHz se necessário
        if audio.channels != self.config.channels:
            audio = audio.set_channels(self.config.channels)
        if audio.frame_rate != self.config.sample_rate:
            audio = audio.set_frame_rate(self.config.sample_rate)
        
        total_duration = len(audio) / 1000.0  # ms -> seconds
        logger.debug("audio_loaded", duration=total_duration)
        
        # Dividir em silêncios
        segments = split_on_silence(
            audio,
            min_silence_len=self.config.min_silence_ms,
            silence_thresh=self.config.silence_threshold_db,
            keep_silence=300,  # Manter 300ms de silêncio nas bordas
        )
        
        # Combinar segmentos pequenos até atingir max_chunk
        current_offset = 0.0
        current_segment = AudioSegment.empty()
        chunk_count = 0
        
        for segment in segments:
            segment_duration = len(segment) / 1000.0
            current_duration = len(current_segment) / 1000.0
            
            # Se adicionar este segmento excede o máximo, yield o atual
            if current_duration + segment_duration > self.config.max_chunk_seconds:
                if current_duration >= self.config.min_chunk_seconds:
                    chunk_count += 1
                    yield self._create_chunk(
                        current_segment,
                        current_offset,
                        current_offset + current_duration,
                        conversation_id,
                    )
                    current_offset += current_duration
                
                # Resetar para novo chunk
                current_segment = segment
            else:
                current_segment += segment
        
        # Yield do último chunk
        final_duration = len(current_segment) / 1000.0
        if final_duration >= self.config.min_chunk_seconds:
            chunk_count += 1
            yield self._create_chunk(
                current_segment,
                current_offset,
                current_offset + final_duration,
                conversation_id,
            )
        
        logger.info("chunking_complete", num_chunks=chunk_count)
    
    def _create_chunk(
        self,
        segment,  # AudioSegment
        start: float,
        end: float,
        conversation_id: str | None,
    ) -> AudioChunk:
        """Converte AudioSegment para AudioChunk."""
        # Exportar como WAV
        buffer = io.BytesIO()
        segment.export(buffer, format="wav")
        audio_bytes = buffer.getvalue()
        
        return AudioChunk.create(
            audio_bytes=audio_bytes,
            duration=end - start,
            start=start,
            end=end,
            conversation_id=conversation_id,
        )
    
    def _chunk_simple(
        self,
        audio_bytes: bytes,
        conversation_id: str | None,
    ) -> Iterator[AudioChunk]:
        """
        Fallback: divide por tempo fixo sem VAD.
        
        Usado quando pydub não está disponível.
        """
        # Estimar duração (assume WAV 16kHz mono 16-bit)
        header_size = 44  # WAV header
        bytes_per_second = self.config.sample_rate * 2  # 16-bit = 2 bytes
        
        audio_data = audio_bytes[header_size:]  # Remove header
        total_seconds = len(audio_data) / bytes_per_second
        
        chunk_bytes = int(self.config.max_chunk_seconds * bytes_per_second)
        offset = 0.0
        chunk_start = 0
        
        while chunk_start < len(audio_data):
            chunk_end = min(chunk_start + chunk_bytes, len(audio_data))
            chunk_data = audio_bytes[:header_size] + audio_data[chunk_start:chunk_end]
            
            duration = (chunk_end - chunk_start) / bytes_per_second
            
            if duration >= self.config.min_chunk_seconds:
                yield AudioChunk.create(
                    audio_bytes=chunk_data,
                    duration=duration,
                    start=offset,
                    end=offset + duration,
                    conversation_id=conversation_id,
                )
            
            offset += duration
            chunk_start = chunk_end
    
    async def chunk_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        conversation_id: str | None = None,
    ) -> AsyncIterator[AudioChunk]:
        """
        Divide stream de áudio em tempo real.
        
        Args:
            audio_stream: Async iterator de bytes de áudio
            conversation_id: ID da conversa
            
        Yields:
            AudioChunk conforme acumula dados suficientes
        """
        buffer = io.BytesIO()
        chunk_count = 0
        
        async for data in audio_stream:
            buffer.write(data)
            
            # Verificar se temos dados suficientes
            buffer_size = buffer.tell()
            bytes_for_max = int(
                self.config.max_chunk_seconds * 
                self.config.sample_rate * 2  # 16-bit
            )
            
            if buffer_size >= bytes_for_max:
                # Processar buffer atual
                buffer.seek(0)
                audio_bytes = buffer.read()
                
                for chunk in self.chunk_audio(audio_bytes, "raw", conversation_id):
                    chunk_count += 1
                    yield chunk
                
                # Resetar buffer
                buffer = io.BytesIO()
        
        # Processar dados restantes
        if buffer.tell() > 0:
            buffer.seek(0)
            audio_bytes = buffer.read()
            for chunk in self.chunk_audio(audio_bytes, "raw", conversation_id):
                chunk_count += 1
                yield chunk
        
        logger.info("stream_chunking_complete", total_chunks=chunk_count)


# ============================================================================
# CLI PARA TESTES
# ============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    def main():
        if len(sys.argv) < 2:
            print("Uso: python audio_chunker.py <arquivo_audio>")
            sys.exit(1)
        
        file_path = Path(sys.argv[1])
        if not file_path.exists():
            print(f"Arquivo não encontrado: {file_path}")
            sys.exit(1)
        
        print(f"Processando: {file_path}")
        print("-" * 50)
        
        chunker = AudioChunker()
        audio_bytes = file_path.read_bytes()
        
        for i, chunk in enumerate(chunker.chunk_audio(audio_bytes, file_path.suffix[1:])):
            print(f"Chunk {i+1}:")
            print(f"  ID: {chunk.chunk_id}")
            print(f"  Duração: {chunk.duration_seconds:.1f}s")
            print(f"  Offset: {chunk.start_offset:.1f}s - {chunk.end_offset:.1f}s")
            print(f"  Bytes: {len(chunk.audio_bytes):,}")
            print()
    
    main()
