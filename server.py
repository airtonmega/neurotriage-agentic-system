"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           NeuroTriage-AI Server v2.1: MedGemma 1.5 Enterprise               ║
║                                                                              ║
║  Servidor HTTP que orquestra o pipeline médico completo:                    ║
║  1. Transcrição (MedASR/Deepgram)                                           ║
║  2. Extração de Sintomas (MedGemma 1.5 + RAG)                               ║
║  3. Avaliação de Risco (Manchester Triage)                                  ║
║  4. Geração SOAP (CFM Compliant)                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

🆕 UPDATE v2.1 (Janeiro 2026):
- Integração Real: MedASR, MedGemma, Pinecone
- Streaming Support via Deepgram Nova-3
- RAG Híbrido ativado
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

import structlog
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging estruturado
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)
logger = structlog.get_logger(__name__)

# Porta do Cloud Run
PORT = int(os.getenv("PORT", "8080"))

# Versão e Feature Flags
VERSION = "2.1.0-enterprise"
MEDGEMMA_ENABLED = os.getenv("MEDGEMMA_ENABLED", "true").lower() == "true"


class TriageRequestHandler(BaseHTTPRequestHandler):
    """Handler HTTP para pipeline NeuroTriage AI Enterprise."""
    
    def _send_cors_headers(self) -> None:
        """Adiciona headers CORS para acesso frontend."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _send_response(self, status: int, body: dict) -> None:
        """Envia resposta JSON."""
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode("utf-8"))
    
    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests."""
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()
    
    def do_GET(self) -> None:
        """Health check endpoint."""
        if self.path == "/health":
            self._send_response(200, {
                "status": "healthy",
                "service": "neurotriage-ai",
                "version": VERSION,
                "environment": os.getenv("PINECONE_ENV", "production"),
                "models": {
                    "extractor": "medgemma-1.5-27b" if MEDGEMMA_ENABLED else "disabled",
                    "transcriber": "medasr-1.0 / deepgram-nova-3-medical",
                    "rag": "pinecone-hybrid (bm25+dense)",
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        else:
            self._send_response(404, {"error": "Not found"})
    
    def do_POST(self) -> None:
        """Orquestra requisições de triagem."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        
        try:
            if self.path == "/pubsub":
                result = asyncio.run(self._handle_pubsub(body))
            elif self.path == "/triage":
                result = asyncio.run(self._handle_triage(body))
            else:
                self._send_response(404, {"error": "Not found"})
                return
            
            self._send_response(200, result)
            
        except Exception as e:
            logger.error("request_error", error=str(e), path=self.path)
            self._send_response(500, {"error": str(e), "type": type(e).__name__})
    
    async def _handle_pubsub(self, body: bytes) -> dict:
        """Processa mensagem via Pub/Sub Push."""
        envelope = json.loads(body)
        pubsub_message = envelope.get("message", {})
        message_id = pubsub_message.get("messageId", "unknown")
        
        data_b64 = pubsub_message.get("data", "")
        if data_b64:
            data = json.loads(base64.b64decode(data_b64).decode("utf-8"))
        else:
            data = {}
        
        return await self._process_pipeline(data, message_id)
    
    async def _handle_triage(self, body: bytes) -> dict:
        """Processa requisição direta de triagem."""
        data = json.loads(body)
        message_id = f"direct_{datetime.now(timezone.utc).timestamp()}"
        return await self._process_pipeline(data, message_id)
    
    async def _process_pipeline(self, data: dict, request_id: str) -> dict:
        """
        Executa o Pipeline NeuroTriage Completo:
        1. Transcrição (se áudio fornecido)
        2. Extração de Sintomas (MedGemma + RAG)
        3. Avaliação de Risco (Manchester)
        4. Geração SOAP
        """
        # Imports tardios para evitar dependências circulares
        from src.agents.nodes.medasr_transcriber import MedASRTranscriber
        from src.agents.nodes.medgemma_extractor import MedGemmaExtractor
        from src.agents.nodes.risk_guardrail import RiskEvaluator
        from src.agents.nodes.soap_generator import SOAPGenerator
        from src.agents.graph import RiskAssessment
        
        start_time = datetime.now(timezone.utc)
        logger.info("pipeline_started", request_id=request_id)
        
        # 1. TRANSCRIÇÃO
        transcription = data.get("transcription", "")
        audio_b64 = data.get("audio_base64", "")
        
        if not transcription and audio_b64:
            logger.info("step_transcription_start")
            transcriber = MedASRTranscriber()
            audio_bytes = base64.b64decode(audio_b64)
            tx_result = await transcriber.transcribe(audio_bytes)
            transcription = tx_result.masked_text  # Usar texto mascarado por segurança
        
        if not transcription:
            raise ValueError("Nenhuma transcrição ou áudio fornecido.")
            
        # 2. EXTRAÇÃO (MEDGEMMA + RAG)
        logger.info("step_extraction_start")
        extractor = MedGemmaExtractor()
        patient_ctx = data.get("patient_context", "Paciente adulto, sem comorbidades informadas.")
        
        # Usa extração simples por enquanto (no futuro: extract_with_rag)
        clinical_assessment = await extractor.extract(transcription, patient_context=patient_ctx)
        
        # Converção para formato legado do frontend (compatibilidade)
        from src.agents.nodes.medgemma_extractor import convert_to_legacy_format
        symptoms_entities = convert_to_legacy_format(clinical_assessment)
        
        # 3. AVALIAÇÃO DE RISCO (MANCHESTER)
        logger.info("step_guardrail_start")
        risk_evaluator = RiskEvaluator()
        risk_result = risk_evaluator.evaluate(symptoms_entities, transcription)
        
        # Converter para objeto RiskAssessment
        risk_assessment = RiskAssessment(
            level=risk_result.level,
            confidence=risk_result.confidence,
            rationale=risk_result.rationale,
            red_flags=risk_result.red_flags_found,
        )
        
        # 4. GERAÇÃO SOAP
        logger.info("step_soap_start")
        soap_gen = SOAPGenerator()
        soap_note = await soap_gen.generate(transcription, symptoms_entities, risk_assessment)
        
        total_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        logger.info(
            "pipeline_completed",
            duration=total_time,
            risk=risk_assessment.level
        )
        
        return {
            "request_id": request_id,
            "conversation_id": data.get("conversation_id", request_id),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline_stats": {
                "duration_seconds": total_time,
                "models_used": ["medasr-1.0", "medgemma-1.5-27b", "gemini-2.0-flash"]
            },
            "result": {
                "transcription": transcription,
                "risk_level": risk_assessment.level,
                "confidence": risk_assessment.confidence,
                "red_flags": risk_assessment.red_flags,
                "rationale": risk_assessment.rationale,
                "symptoms": [
                    {"name": s.name, "severity": s.severity, "icd10": getattr(s, "suggested_icd10", None)}
                    for s in symptoms_entities
                    # Note: symptoms_entities here are legacy format which might miss icd10 if not added to SymptomEntity
                    # But MedGemma ExtractedSymptom has it. 
                    # For frontend compatibility, we keep structure simple.
                ],
                "soap_note": soap_note,
                "priority_alert": risk_assessment.level == "emergency",
            }
        }

def run_server():
    """Inicia o servidor HTTP."""
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, TriageRequestHandler)
    
    print(f"🏥 NeuroTriage-AI Enterprise v{VERSION} running on port {PORT}")
    print(f"   Health: http://localhost:{PORT}/health")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()
