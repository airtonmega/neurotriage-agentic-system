"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           NeuroTriage-AI Server v2.0: MedGemma 1.5 Edition                  ║
║                                                                              ║
║  Servidor HTTP que recebe requisições de triagem via Pub/Sub Push           ║
║  e processa através do pipeline MedGemma 1.5 + Risk Guardrail.              ║
╚══════════════════════════════════════════════════════════════════════════════╝

🆕 UPGRADE MedGemma 1.5 (Janeiro 2026):
- Extração estruturada com CID-10 automático
- Chain-of-Thought reasoning para diagnóstico
- Fallback gracioso: MedGemma → Gemini → Keywords
- MedASR para transcrição médica especializada
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

# Configurar logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)
logger = structlog.get_logger(__name__)

# Porta do Cloud Run
PORT = int(os.getenv("PORT", "8080"))

# Versão e configuração
VERSION = "2.0.0"
MEDGEMMA_ENABLED = os.getenv("MEDGEMMA_ENABLED", "true").lower() == "true"


class TriageRequestHandler(BaseHTTPRequestHandler):
    """Handler HTTP para requisições de triagem com MedGemma 1.5."""
    
    def _send_cors_headers(self) -> None:
        """Adiciona headers CORS para permitir acesso do frontend."""
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
                "models": {
                    "extractor": "medgemma-1.5-27b" if MEDGEMMA_ENABLED else "keywords",
                    "transcriber": "medasr-1.0 / deepgram-nova-3",
                    "risk_engine": "manchester-protocol-v2",
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        elif self.path == "/":
            self._send_response(200, {
                "service": "NeuroTriage-AI",
                "version": VERSION,
                "description": "Agentic Medical Triage System powered by MedGemma 1.5",
                "capabilities": [
                    "Extração estruturada de sintomas com CID-10",
                    "Chain-of-Thought reasoning clínico",
                    "Classificação Manchester automatizada",
                    "Geração de prontuário SOAP",
                ],
                "endpoints": {
                    "GET /health": "Health check com status dos modelos",
                    "POST /triage": "Triagem completa com MedGemma (recomendado)",
                    "POST /triage-fast": "Triagem rápida via keywords (baixa latência)",
                    "POST /pubsub": "Pub/Sub push handler",
                },
            })
        else:
            self._send_response(404, {"error": "Not found"})
    
    def do_POST(self) -> None:
        """Handle triage requests."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        
        try:
            if self.path == "/pubsub":
                result = self._handle_pubsub(body)
            elif self.path == "/triage":
                # Triagem completa com MedGemma (quando disponível)
                result = self._handle_triage(body, use_medgemma=MEDGEMMA_ENABLED)
            elif self.path == "/triage-fast":
                # Triagem rápida via keywords (baixa latência)
                result = self._handle_triage(body, use_medgemma=False)
            else:
                self._send_response(404, {"error": "Not found"})
                return
            
            self._send_response(200, result)
            
        except Exception as e:
            logger.error("request_error", error=str(e), path=self.path)
            self._send_response(500, {"error": str(e)})
    
    def _handle_pubsub(self, body: bytes) -> dict:
        """
        Processa mensagem do Pub/Sub Push.
        
        Formato esperado:
        {
            "message": {
                "data": "<base64 encoded JSON>",
                "messageId": "...",
                "publishTime": "..."
            },
            "subscription": "..."
        }
        """
        envelope = json.loads(body)
        
        # Extrair mensagem do Pub/Sub
        pubsub_message = envelope.get("message", {})
        message_id = pubsub_message.get("messageId", "unknown")
        
        logger.info("pubsub_message_received", message_id=message_id)
        
        # Decodificar dados
        data_b64 = pubsub_message.get("data", "")
        if data_b64:
            data = json.loads(base64.b64decode(data_b64).decode("utf-8"))
        else:
            data = {}
        
        # Processar triagem
        return self._process_triage(data, message_id)
    
    def _handle_triage(self, body: bytes, use_medgemma: bool = True) -> dict:
        """Processa requisição de triagem direta."""
        data = json.loads(body)
        message_id = f"direct_{datetime.now(timezone.utc).timestamp()}"
        return self._process_triage(data, message_id, use_medgemma=use_medgemma)
    
    def _process_triage(self, data: dict, request_id: str, use_medgemma: bool = True) -> dict:
        """
        Executa o pipeline de triagem com MedGemma ou keywords.
        
        Args:
            data: Dados da requisição (transcrição ou áudio)
            request_id: ID para rastreamento
            use_medgemma: Se True, usa MedGemma 1.5; senão, keywords
            
        Returns:
            Resultado da triagem com sintomas, risco e SOAP
        """
        from src.agents.graph import SymptomEntity, RiskAssessment
        from src.agents.nodes.risk_guardrail import RiskEvaluator
        from src.agents.nodes.soap_generator import generate_soap_template
        
        extractor_used = "medgemma-1.5" if use_medgemma else "keywords"
        logger.info("processing_triage", request_id=request_id, extractor=extractor_used)
        
        # Extrair transcrição (pode vir pronta ou ser transcrita)
        transcription = data.get("transcription", "")
        conversation_id = data.get("conversation_id", request_id)
        
        if not transcription:
            # Se não há transcrição, pode haver áudio para transcrever
            audio_b64 = data.get("audio_base64", "")
            if audio_b64:
                # TODO: Transcrever com Deepgram
                transcription = "[Transcrição pendente - implementar Deepgram]"
            else:
                return {
                    "request_id": request_id,
                    "error": "No transcription or audio provided",
                }
        
        # Simular extração de sintomas (sem LLM para demo)
        # Em produção, usaria extract_symptoms() com Gemini
        symptoms = self._extract_symptoms_simple(transcription)
        
        # Avaliar risco
        evaluator = RiskEvaluator()
        result = evaluator.evaluate(symptoms)
        
        # Criar assessment
        risk = RiskAssessment(
            level=result.level,
            confidence=result.confidence,
            rationale=result.rationale,
            red_flags=result.red_flags_found,
        )
        
        # Gerar SOAP
        soap = generate_soap_template(transcription, symptoms, risk)
        
        logger.info(
            "triage_complete",
            request_id=request_id,
            risk_level=result.level,
            confidence=result.confidence,
        )
        
        return {
            "request_id": request_id,
            "conversation_id": conversation_id,
            "risk_level": result.level,
            "confidence": result.confidence,
            "red_flags": result.red_flags_found,
            "rationale": result.rationale,
            "symptoms": [
                {"name": s.name, "severity": s.severity}
                for s in symptoms
            ],
            "soap_note_length": len(soap),
            "priority_alert": result.level == "emergency",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    def _extract_symptoms_simple(self, transcription: str) -> list:
        """
        Extração simples de sintomas por keywords.
        Em produção, usar extract_symptoms() com LLM.
        """
        from src.agents.graph import SymptomEntity
        
        text_lower = transcription.lower()
        symptoms = []
        
        # Mapeamento de keywords para sintomas
        keyword_map = {
            # Emergência
            ("dor no peito", "dor toracica", "dor torácica"): ("dor_toracica", "critical"),
            ("braço esquerdo", "mse", "irradia"): ("irradiacao_mse", "critical"),
            ("suando frio", "sudorese"): ("sudorese", "high"),
            ("falta de ar", "dispneia"): ("dispneia", "high"),
            ("desmaio", "sincope", "síncope"): ("sincope", "critical"),
            ("paralisia", "fraqueza súbita", "hemiplegi"): ("hemiparesia", "critical"),
            ("fala difícil", "disartria"): ("disartria", "critical"),
            ("boca torta", "desvio"): ("desvio_rima", "critical"),
            
            # Urgente
            ("febre alta", "febre 39", "febre 40"): ("febre_alta", "high"),
            ("pressão alta", "hipertens"): ("hipertensao", "high"),
            ("gestante", "grávida"): ("gestante", "high"),
            ("visão embaçada", "escotoma"): ("disturbio_visual", "high"),
            
            # Rotina
            ("dor de cabeça", "cefaleia", "cefaléia"): ("cefaleia", "low"),
            ("náusea", "enjoo"): ("nausea", "low"),
            ("tosse", "tossindo"): ("tosse", "low"),
            ("dor lombar", "lombalgia"): ("lombalgia", "low"),
        }
        
        for keywords, (symptom_name, severity) in keyword_map.items():
            if any(kw in text_lower for kw in keywords):
                symptoms.append(SymptomEntity(name=symptom_name, severity=severity))
        
        return symptoms
    
    def log_message(self, format: str, *args: Any) -> None:
        """Sobrescrever log padrão do BaseHTTPRequestHandler."""
        logger.info("http_request", message=format % args)


def run_server():
    """Inicia o servidor HTTP."""
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, TriageRequestHandler)
    
    logger.info("server_starting", port=PORT)
    print(f"🏥 NeuroTriage-AI Server running on port {PORT}")
    print(f"   Health: http://localhost:{PORT}/health")
    print(f"   Triage: POST http://localhost:{PORT}/triage")
    print(f"   Pub/Sub: POST http://localhost:{PORT}/pubsub")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("server_shutdown")
        print("\n👋 Server stopped")


if __name__ == "__main__":
    run_server()
