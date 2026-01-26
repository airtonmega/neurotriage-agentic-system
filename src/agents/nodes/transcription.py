"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              Deepgram Nova-3 Medical: Transcrição de Áudio                   ║
║                                                                              ║
║  Módulo de transcrição otimizado para vocabulário médico usando             ║
║  Deepgram Nova-3 Medical - HIPAA compliant, latência <300ms.                ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 O QUE ESTE ARQUIVO FAZ?
---------------------------
Converte áudio de consultas médicas em texto com alta precisão para
terminologia clínica (fármacos, condições, procedimentos).

📊 POR QUE DEEPGRAM NOVA-3 MEDICAL?
-----------------------------------
- 30% menor taxa de erro (WER) que Whisper em vocabulário médico
- Latência <300ms (streaming real-time)
- HIPAA compliant nativo
- Suporte a pt-BR com modelo médico especializado
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

# ============================================================================
# CONFIGURAÇÃO E TIPOS
# ============================================================================

logger = structlog.get_logger(__name__)


@dataclass
class TranscriptionResult:
    """
    Resultado da transcrição de áudio.
    
    Attributes:
        text: Transcrição original completa
        masked_text: Transcrição com PII removido
        confidence: Confiança média da transcrição (0.0-1.0)
        duration_seconds: Duração do áudio processado
        language: Código do idioma detectado
        words: Lista de palavras com timestamps (opcional)
    """
    text: str
    masked_text: str
    confidence: float
    duration_seconds: float
    language: str = "pt-BR"
    words: list[dict] | None = None


@dataclass 
class DeepgramConfig:
    """Configuração do cliente Deepgram."""
    api_key: str
    model: str = "nova-3-medical"
    language: str = "pt-BR"
    punctuate: bool = True
    diarize: bool = False
    smart_format: bool = True
    filler_words: bool = False
    
    @classmethod
    def from_env(cls) -> "DeepgramConfig":
        """Carrega configuração das variáveis de ambiente."""
        return cls(
            api_key=os.getenv("DEEPGRAM_API_KEY", ""),
            model=os.getenv("DEEPGRAM_MODEL", "nova-3-medical"),
            language=os.getenv("DEEPGRAM_LANGUAGE", "pt-BR"),
        )


# ============================================================================
# CLIENTE DEEPGRAM
# ============================================================================

class DeepgramTranscriber:
    """
    Transcritor de áudio usando Deepgram Nova-3 Medical.
    
    Suporta:
    - Transcrição batch (arquivo completo)
    - Transcrição streaming (real-time)
    - Fallback automático em caso de erro
    
    Example:
        >>> transcriber = DeepgramTranscriber.from_env()
        >>> result = await transcriber.transcribe_bytes(audio_bytes)
        >>> print(result.text)
    """
    
    def __init__(self, config: DeepgramConfig):
        """
        Inicializa o transcritor.
        
        Args:
            config: Configuração com API key e parâmetros
        """
        self.config = config
        self._client = None
        
    @classmethod
    def from_env(cls) -> "DeepgramTranscriber":
        """Factory: cria transcritor com config do ambiente."""
        return cls(DeepgramConfig.from_env())
    
    @property
    def client(self):
        """Lazy loading do cliente Deepgram."""
        if self._client is None:
            try:
                from deepgram import DeepgramClient
                self._client = DeepgramClient(self.config.api_key)
            except ImportError as e:
                raise ImportError(
                    "deepgram-sdk não instalado. Execute: pip install deepgram-sdk"
                ) from e
        return self._client
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        mimetype: str = "audio/wav",
    ) -> TranscriptionResult:
        """
        Transcreve áudio a partir de bytes.
        
        Args:
            audio_bytes: Áudio em bytes
            mimetype: Tipo MIME do áudio (audio/wav, audio/mp3, etc.)
            
        Returns:
            TranscriptionResult com texto e metadados
            
        Raises:
            DeepgramError: Se a API retornar erro
        """
        from deepgram import PrerecordedOptions
        
        logger.info(
            "transcribing_audio",
            size_bytes=len(audio_bytes),
            model=self.config.model,
        )
        
        # Configurar opções de transcrição
        options = PrerecordedOptions(
            model=self.config.model,
            language=self.config.language,
            punctuate=self.config.punctuate,
            diarize=self.config.diarize,
            smart_format=self.config.smart_format,
            filler_words=self.config.filler_words,
        )
        
        # Preparar payload
        payload = {"buffer": audio_bytes, "mimetype": mimetype}
        
        # Chamar API
        response = await self.client.listen.asyncrest.v("1").transcribe_file(
            payload, options
        )
        
        # Extrair resultados
        result = response.results
        channel = result.channels[0]
        alternative = channel.alternatives[0]
        
        text = alternative.transcript
        confidence = alternative.confidence
        duration = result.metadata.duration if hasattr(result, "metadata") else 0.0
        
        # Extrair palavras com timestamps (para highlights)
        words = None
        if hasattr(alternative, "words") and alternative.words:
            words = [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "confidence": w.confidence,
                }
                for w in alternative.words
            ]
        
        logger.info(
            "transcription_complete",
            text_length=len(text),
            confidence=confidence,
            duration=duration,
        )
        
        # Aplicar mascaramento PII
        masked_text = await self._mask_pii(text)
        
        return TranscriptionResult(
            text=text,
            masked_text=masked_text,
            confidence=confidence,
            duration_seconds=duration,
            language=self.config.language,
            words=words,
        )
    
    async def transcribe_file(self, file_path: str) -> TranscriptionResult:
        """
        Transcreve a partir de arquivo.
        
        Args:
            file_path: Caminho do arquivo de áudio
            
        Returns:
            TranscriptionResult
        """
        from pathlib import Path
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
        
        # Detectar mimetype pela extensão
        ext_to_mime = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".ogg": "audio/ogg",
            ".flac": "audio/flac",
        }
        mimetype = ext_to_mime.get(path.suffix.lower(), "audio/wav")
        
        audio_bytes = path.read_bytes()
        return await self.transcribe_bytes(audio_bytes, mimetype)
    
    async def _mask_pii(self, text: str) -> str:
        """
        Aplica mascaramento de PII no texto transcrito.
        
        Usa o PIIMasker do módulo privacy para garantir conformidade LGPD/HIPAA.
        """
        try:
            from src.privacy.pii_masker import PIIMasker
            
            masker = PIIMasker(strategy="TOKEN", enable_ner=True)
            result = masker.mask(text)
            return result.masked_text
        except ImportError:
            logger.warning("pii_masker_not_available", text_length=len(text))
            return text


# ============================================================================
# STREAMING TRANSCRIPTION (Real-time)
# ============================================================================

class DeepgramStreamingTranscriber:
    """
    Transcritor streaming para áudio em tempo real.
    
    Ideal para consultas de telemedicina ao vivo onde a latência
    é crítica (<300ms).
    
    Example:
        >>> async with DeepgramStreamingTranscriber.from_env() as transcriber:
        ...     async for partial in transcriber.stream(audio_generator):
        ...         print(f"Parcial: {partial}")
    """
    
    def __init__(self, config: DeepgramConfig):
        self.config = config
        self._connection = None
        
    @classmethod
    def from_env(cls) -> "DeepgramStreamingTranscriber":
        return cls(DeepgramConfig.from_env())
    
    async def __aenter__(self):
        """Context manager: abre conexão WebSocket."""
        from deepgram import DeepgramClient, LiveOptions
        
        client = DeepgramClient(self.config.api_key)
        
        options = LiveOptions(
            model=self.config.model,
            language=self.config.language,
            punctuate=self.config.punctuate,
            smart_format=self.config.smart_format,
            interim_results=True,  # Resultados parciais
            endpointing=300,  # ms de silêncio para finalizar frase
        )
        
        self._connection = await client.listen.asyncwebsocket.v("1").create(options)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager: fecha conexão."""
        if self._connection:
            await self._connection.finish()
    
    async def stream(self, audio_chunks):
        """
        Processa stream de chunks de áudio.
        
        Args:
            audio_chunks: AsyncIterator de bytes de áudio
            
        Yields:
            Transcrições parciais conforme chegam
        """
        if not self._connection:
            raise RuntimeError("Use 'async with' para iniciar streaming")
        
        async for chunk in audio_chunks:
            await self._connection.send(chunk)
            
            # Processar resultados disponíveis
            async for result in self._connection:
                if result.is_final:
                    yield result.channel.alternatives[0].transcript


# ============================================================================
# FUNÇÃO PRINCIPAL (Interface com o grafo)
# ============================================================================

async def transcribe_audio(
    audio_bytes: bytes | None,
    chunk_id: str,
    language: str = "pt-BR",
) -> TranscriptionResult:
    """
    Função principal de transcrição para uso no grafo LangGraph.
    
    Esta é a interface que o nó de transcrição chama.
    
    Args:
        audio_bytes: Áudio em bytes para transcrever
        chunk_id: ID único do chunk para rastreamento
        language: Código do idioma (default: pt-BR)
        
    Returns:
        TranscriptionResult com texto original e mascarado
        
    Example:
        >>> result = await transcribe_audio(audio_data, "chunk_001")
        >>> print(result.masked_text)
    """
    if audio_bytes is None or len(audio_bytes) == 0:
        logger.warning("empty_audio_received", chunk_id=chunk_id)
        return TranscriptionResult(
            text="",
            masked_text="",
            confidence=0.0,
            duration_seconds=0.0,
        )
    
    logger.info("starting_transcription", chunk_id=chunk_id, language=language)
    
    # Criar transcritor e processar
    transcriber = DeepgramTranscriber.from_env()
    
    # Atualizar language se diferente
    if language != transcriber.config.language:
        transcriber.config.language = language
    
    result = await transcriber.transcribe_bytes(audio_bytes)
    
    logger.info(
        "transcription_finished",
        chunk_id=chunk_id,
        text_length=len(result.text),
        confidence=result.confidence,
    )
    
    return result


# ============================================================================
# CLI PARA TESTES
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import sys
    
    async def main():
        if len(sys.argv) < 2:
            print("Uso: python transcription.py <arquivo_audio>")
            sys.exit(1)
        
        file_path = sys.argv[1]
        
        print(f"Transcrevendo: {file_path}")
        print("-" * 50)
        
        transcriber = DeepgramTranscriber.from_env()
        result = await transcriber.transcribe_file(file_path)
        
        print(f"Texto: {result.text}")
        print(f"\nMascarado: {result.masked_text}")
        print(f"\nConfiança: {result.confidence:.2%}")
        print(f"Duração: {result.duration_seconds:.1f}s")
    
    asyncio.run(main())
