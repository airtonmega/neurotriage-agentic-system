"""
Módulo de Nodes do NeuroTriage-AI.

Este pacote contém os nós (nodes) do grafo LangGraph que processam
cada etapa da triagem médica.
"""

from src.agents.nodes.transcription import (
    transcribe_audio,
    TranscriptionResult,
    DeepgramTranscriber,
)
from src.agents.nodes.analysis import (
    extract_symptoms,
    SymptomExtractor,
    HybridRetriever,
)
from src.agents.nodes.risk_guardrail import (
    evaluate_risk,
    RiskEvaluator,
)
from src.agents.nodes.soap_generator import (
    generate_soap,
    SOAPGenerator,
)

__all__ = [
    # Transcription
    "transcribe_audio",
    "TranscriptionResult",
    "DeepgramTranscriber",
    # Analysis
    "extract_symptoms",
    "SymptomExtractor",
    "HybridRetriever",
    # Risk Guardrail
    "evaluate_risk",
    "RiskEvaluator",
    # SOAP Generator
    "generate_soap",
    "SOAPGenerator",
]
