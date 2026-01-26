"""
Testes Unitários para os Módulos de Nodes do NeuroTriage-AI.

Testes focados em validar a lógica de triagem médica sem dependências externas.
"""

from __future__ import annotations

import pytest


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_transcription() -> str:
    """Transcrição de exemplo para testes."""
    return (
        "Paciente relata dor de cabeça há 2 dias, de intensidade moderada, "
        "localizada na região frontal. A dor piora com luz forte. "
        "Também refere náuseas leves. Nega febre ou vômitos."
    )


@pytest.fixture
def emergency_transcription() -> str:
    """Transcrição de caso de emergência."""
    return (
        "Paciente com dor forte no peito há 30 minutos, irradiando para o braço esquerdo. "
        "Está suando frio e sente falta de ar. Tem histórico de hipertensão."
    )


# ============================================================================
# TESTES DO RISK GUARDRAIL
# ============================================================================

class TestRiskGuardrail:
    """Testes para o módulo de avaliação de risco."""

    def test_red_flags_detection_emergency(self):
        """Deve detectar red flags em sintomas de emergência."""
        from src.agents.nodes.risk_guardrail import RiskEvaluator
        from src.agents.graph import SymptomEntity
        
        symptoms = [
            SymptomEntity(name="dor_toracica", severity="critical", body_region="tórax"),
            SymptomEntity(name="sudorese", severity="high"),
            SymptomEntity(name="dispneia", severity="high"),
        ]
        
        evaluator = RiskEvaluator()
        result = evaluator.evaluate(symptoms)
        
        assert result.level == "emergency"
        assert len(result.red_flags_found) > 0
        assert result.confidence >= 0.7

    def test_routine_classification(self):
        """Deve classificar sintomas leves como rotina."""
        from src.agents.nodes.risk_guardrail import RiskEvaluator
        from src.agents.graph import SymptomEntity
        
        symptoms = [
            SymptomEntity(name="cefaleia_leve", severity="low", duration="2 dias"),
        ]
        
        evaluator = RiskEvaluator()
        result = evaluator.evaluate(symptoms)
        
        assert result.level == "routine"
        assert len(result.red_flags_found) == 0

    def test_urgent_classification(self):
        """Deve classificar síndrome respiratória como urgente."""
        from src.agents.nodes.risk_guardrail import RiskEvaluator
        from src.agents.graph import SymptomEntity
        
        symptoms = [
            SymptomEntity(name="febre_alta", severity="high", duration="3 dias"),
            SymptomEntity(name="tosse_produtiva", severity="medium"),
            SymptomEntity(name="dispneia", severity="medium"),
        ]
        
        evaluator = RiskEvaluator()
        result = evaluator.evaluate(symptoms)
        
        assert result.level in ["urgent", "emergency"]
        assert result.total_score >= 50

    def test_empty_symptoms(self):
        """Deve retornar rotina para lista vazia de sintomas."""
        from src.agents.nodes.risk_guardrail import RiskEvaluator
        
        evaluator = RiskEvaluator()
        result = evaluator.evaluate([])
        
        assert result.level == "routine"
        assert result.symptoms_analyzed == 0


# ============================================================================
# TESTES DO TRANSCRIPTION
# ============================================================================

class TestTranscription:
    """Testes para o módulo de transcrição."""

    def test_transcription_result_dataclass(self):
        """Deve criar TranscriptionResult corretamente."""
        from src.agents.nodes.transcription import TranscriptionResult
        
        result = TranscriptionResult(
            text="Texto original",
            masked_text="Texto mascarado",
            confidence=0.95,
            duration_seconds=30.5,
        )
        
        assert result.text == "Texto original"
        assert result.masked_text == "Texto mascarado"
        assert result.confidence == 0.95
        assert result.duration_seconds == 30.5
        assert result.language == "pt-BR"

    def test_deepgram_config_from_env(self):
        """Deve criar config do Deepgram a partir de env vars."""
        import os
        from src.agents.nodes.transcription import DeepgramConfig
        
        # Simular env var
        os.environ["DEEPGRAM_MODEL"] = "nova-3-medical"
        
        config = DeepgramConfig.from_env()
        
        assert config.model == "nova-3-medical"
        assert config.language == "pt-BR"


# ============================================================================
# TESTES DO MEDGEMMA EXTRACTOR
# ============================================================================

class TestMedGemmaExtractor:
    """Testes para o módulo de extração MedGemma."""

    def test_medgemma_config_from_env(self):
        """Deve criar config do MedGemma a partir de env vars."""
        import os
        from src.agents.nodes.medgemma_extractor import MedGemmaConfig
        
        os.environ["MEDGEMMA_MODEL"] = "medgemma-1.5-27b"
        config = MedGemmaConfig.from_env()
        
        assert config.model_name == "medgemma-1.5-27b"
        assert config.enable_cot is True

    @pytest.mark.asyncio
    async def test_extractor_initialization(self, mocker):
        """Deve inicializar extrator corretamente (mocked)."""
        from src.agents.nodes.medgemma_extractor import MedGemmaExtractor
        
        # Mock ChatVertexAI para evitar chamadas reais
        mocker.patch("langchain_google_vertexai.ChatVertexAI")
        
        extractor = MedGemmaExtractor()
        assert extractor.config.model_name == "medgemma-1.5-27b"

    def test_convert_to_legacy_format(self):
        """Deve converter ClinicalAssessment para formato legado."""
        from src.agents.nodes.medgemma_extractor import (
            ClinicalAssessment, 
            ExtractedSymptom, 
            convert_to_legacy_format
        )
        
        assessment = ClinicalAssessment(
            symptoms=[
                ExtractedSymptom(
                    name="cefaleia", 
                    severity="medium",
                    body_region="frontal",
                    duration="2 dias"
                )
            ]
        )
        
        legacy = convert_to_legacy_format(assessment)
        
        assert len(legacy) == 1
        assert legacy[0].name == "cefaleia"
        assert legacy[0].severity == "medium"
        assert legacy[0].body_region == "frontal"


# ============================================================================
# TESTES DO SOAP GENERATOR
# ============================================================================

class TestSOAPGenerator:
    """Testes para o gerador de prontuário SOAP."""

    def test_soap_template_generation(self):
        """Deve gerar SOAP usando template simples."""
        from src.agents.nodes.soap_generator import generate_soap_template
        from src.agents.graph import SymptomEntity, RiskAssessment
        
        symptoms = [
            SymptomEntity(name="cefaleia", severity="medium", body_region="frontal"),
        ]
        
        risk = RiskAssessment(
            level="routine",
            confidence=0.85,
            rationale="Cefaleia tensional típica.",
            red_flags=[],
        )
        
        soap = generate_soap_template(
            transcription="Dor de cabeça há 2 dias...",
            symptoms=symptoms,
            risk=risk,
        )
        
        assert "SUBJETIVO" in soap
        assert "OBJETIVO" in soap
        assert "AVALIAÇÃO" in soap
        assert "PLANO" in soap
        assert "ROTINA" in soap


# ============================================================================
# TESTES DO AUDIO CHUNKER
# ============================================================================

class TestAudioChunker:
    """Testes para o divisor de áudio."""

    def test_chunker_config_defaults(self):
        """Deve criar config com valores padrão."""
        from src.ingest.audio_chunker import ChunkerConfig
        
        config = ChunkerConfig()
        
        assert config.max_chunk_seconds == 30.0
        assert config.min_chunk_seconds == 3.0
        assert config.sample_rate == 16000

    def test_audio_chunk_create(self):
        """Deve criar AudioChunk com ID gerado."""
        from src.ingest.audio_chunker import AudioChunk
        
        audio_bytes = b"fake audio data" * 100
        chunk = AudioChunk.create(
            audio_bytes=audio_bytes,
            duration=10.5,
            start=0.0,
            end=10.5,
            conversation_id="test_conv",
        )
        
        assert chunk.chunk_id.startswith("chunk_")
        assert chunk.duration_seconds == 10.5
        assert chunk.metadata["conversation_id"] == "test_conv"


# ============================================================================
# TESTES DO PUBSUB CONSUMER
# ============================================================================

class TestPubSubConsumer:
    """Testes para o consumer Pub/Sub."""

    def test_pubsub_config_validation(self):
        """Deve validar configuração do Pub/Sub."""
        from src.ingest.pubsub_consumer import PubSubConfig
        
        config = PubSubConfig(
            project_id="test-project",
            subscription_id="test-sub",
        )
        
        assert config.project_id == "test-project"
        assert config.max_messages == 10
        assert config.max_concurrent == 5
