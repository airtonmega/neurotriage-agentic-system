"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           SOAP Generator: Prontuário Médico Estruturado                      ║
║                                                                              ║
║  Gera prontuário no formato SOAP (Subjetivo, Objetivo, Avaliação, Plano)    ║
║  conforme padrões CFM (Conselho Federal de Medicina) brasileiro.            ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 O QUE ESTE ARQUIVO FAZ?
---------------------------
1. Recebe sintomas, avaliação de risco e transcrição
2. Gera prontuário SOAP estruturado usando LLM
3. Inclui CID-10 sugerido quando aplicável
4. Formata saída para integração com sistemas de saúde

📋 FORMATO SOAP:
----------------
- S (Subjetivo): O que o paciente relatou (queixa, história)
- O (Objetivo): Observações e dados objetivos
- A (Avaliação/Assessment): Hipótese diagnóstica, classificação
- P (Plano): Conduta, orientações, encaminhamentos
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from src.agents.graph import SymptomEntity, RiskAssessment

logger = structlog.get_logger(__name__)


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

class SOAPConfig(BaseModel):
    """Configuração do gerador SOAP."""
    
    gemini_model: str = Field(
        default="gemini-2.0-flash",
        description="Modelo Gemini para geração SOAP",
    )
    include_cid10: bool = Field(
        default=True,
        description="Incluir sugestão de CID-10",
    )
    include_timestamp: bool = Field(
        default=True,
        description="Incluir timestamp no prontuário",
    )
    
    @classmethod
    def from_env(cls) -> "SOAPConfig":
        return cls(
            gemini_model=os.getenv("VERTEX_AI_MODEL", "gemini-2.0-flash"),
        )


# ============================================================================
# PROMPTS
# ============================================================================

SOAP_SYSTEM_PROMPT = """
Você é um médico documentador experiente que gera prontuários SOAP (Subjetivo, Objetivo, 
Avaliação, Plano) de alta qualidade para telemedicina.

## Sua Tarefa
Gere um prontuário SOAP completo baseado na transcrição da consulta, sintomas identificados 
e avaliação de risco fornecidos.

## Regras CRÍTICAS
1. Use APENAS informações presentes nos dados fornecidos
2. NÃO invente sintomas, medicamentos ou diagnósticos
3. Seja objetivo e use terminologia médica apropriada
4. Inclua CID-10 quando houver suspeita diagnóstica clara
5. O plano deve ser factível para telemedicina

## Formato de Saída
Gere o prontuário em texto estruturado, seguindo exatamente este template:

═══════════════════════════════════════════════════════════════════
                    PRONTUÁRIO DE TELEMEDICINA
═══════════════════════════════════════════════════════════════════

📋 SUBJETIVO (S)
----------------
[Queixa principal, história da doença atual, antecedentes relevantes]

📊 OBJETIVO (O)
---------------
[Dados objetivos disponíveis, classificação de risco]

🔍 AVALIAÇÃO (A)
----------------
[Hipótese diagnóstica, diagnósticos diferenciais]
CID-10 Sugerido: [código] - [descrição]

📝 PLANO (P)
------------
[Conduta, orientações, encaminhamentos]

═══════════════════════════════════════════════════════════════════
CLASSIFICAÇÃO: [ROTINA/URGENTE/EMERGÊNCIA]
═══════════════════════════════════════════════════════════════════
"""

SOAP_USER_TEMPLATE = """
## Transcrição da Consulta
```
{transcription}
```

## Sintomas Identificados
{symptoms_formatted}

## Avaliação de Risco
- Nível: {risk_level}
- Confiança: {risk_confidence:.0%}
- Red Flags: {red_flags}
- Justificativa: {risk_rationale}

Gere o prontuário SOAP completo.
"""


# ============================================================================
# GERADOR SOAP
# ============================================================================

class SOAPGenerator:
    """
    Gerador de prontuários SOAP usando LLM.
    
    Combina transcrição, sintomas estruturados e avaliação de risco
    para gerar prontuário médico completo e padronizado.
    
    Example:
        >>> generator = SOAPGenerator()
        >>> soap = await generator.generate(transcription, symptoms, risk)
        >>> print(soap)
    """
    
    def __init__(self, config: SOAPConfig | None = None):
        self.config = config or SOAPConfig.from_env()
        self._llm = None
        
    @property
    def llm(self):
        """Lazy loading do LLM."""
        if self._llm is None:
            from langchain_google_vertexai import ChatVertexAI
            
            self._llm = ChatVertexAI(
                model=self.config.gemini_model,
                temperature=0.1,  # Ligeiramente criativo para redação
                max_output_tokens=2000,
            )
        return self._llm
    
    def _format_symptoms(self, symptoms: list["SymptomEntity"]) -> str:
        """Formata lista de sintomas para o prompt."""
        if not symptoms:
            return "- Nenhum sintoma identificado"
        
        lines = []
        for i, s in enumerate(symptoms, 1):
            line = f"{i}. **{s.name}**"
            details = []
            
            if s.severity:
                severity_map = {
                    "low": "Leve",
                    "medium": "Moderado",
                    "high": "Intenso",
                    "critical": "CRÍTICO",
                }
                details.append(f"Severidade: {severity_map.get(s.severity, s.severity)}")
            
            if s.body_region:
                details.append(f"Região: {s.body_region}")
            
            if s.duration:
                details.append(f"Duração: {s.duration}")
            
            if s.onset:
                details.append(f"Início: {s.onset}")
            
            if details:
                line += f" ({', '.join(details)})"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
    )
    async def generate(
        self,
        transcription: str,
        symptoms: list["SymptomEntity"],
        risk: "RiskAssessment",
    ) -> str:
        """
        Gera prontuário SOAP completo.
        
        Args:
            transcription: Texto mascarado da consulta
            symptoms: Lista de sintomas extraídos
            risk: Avaliação de risco
            
        Returns:
            Prontuário SOAP formatado como string
        """
        logger.info(
            "generating_soap",
            num_symptoms=len(symptoms),
            risk_level=risk.level,
        )
        
        # Formatar sintomas
        symptoms_formatted = self._format_symptoms(symptoms)
        
        # Formatar red flags
        red_flags_str = ", ".join(risk.red_flags) if risk.red_flags else "Nenhum"
        
        # Preparar prompt
        user_prompt = SOAP_USER_TEMPLATE.format(
            transcription=transcription,
            symptoms_formatted=symptoms_formatted,
            risk_level=risk.level.upper(),
            risk_confidence=risk.confidence,
            red_flags=red_flags_str,
            risk_rationale=risk.rationale,
        )
        
        # Chamar LLM
        response = await self.llm.ainvoke([
            {"role": "system", "content": SOAP_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
        
        soap_note = response.content
        
        # Adicionar timestamp se configurado
        if self.config.include_timestamp:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            soap_note = f"Data/Hora: {timestamp}\n\n{soap_note}"
        
        logger.info("soap_generated", length=len(soap_note))
        
        return soap_note


# ============================================================================
# FUNÇÃO PRINCIPAL (Interface com o grafo)
# ============================================================================

async def generate_soap(
    transcription: str,
    symptoms: list["SymptomEntity"],
    risk: "RiskAssessment | None",
) -> str:
    """
    Função principal de geração SOAP para uso no grafo LangGraph.
    
    Esta é a interface que o nó de SOAP generator chama.
    
    Args:
        transcription: Texto mascarado da consulta
        symptoms: Lista de sintomas extraídos
        risk: Avaliação de risco (pode ser None)
        
    Returns:
        Prontuário SOAP formatado
        
    Example:
        >>> soap = await generate_soap(transcription, symptoms, risk)
        >>> print(soap)
    """
    from src.agents.graph import RiskAssessment
    
    # Garantir que temos uma avaliação de risco
    if risk is None:
        risk = RiskAssessment(
            level="routine",
            confidence=0.5,
            rationale="Avaliação de risco não disponível.",
            red_flags=[],
        )
    
    generator = SOAPGenerator()
    return await generator.generate(transcription, symptoms, risk)


# ============================================================================
# TEMPLATE SOAP SIMPLIFICADO (Fallback sem LLM)
# ============================================================================

def generate_soap_template(
    transcription: str,
    symptoms: list["SymptomEntity"],
    risk: "RiskAssessment",
) -> str:
    """
    Gera SOAP usando template simples (sem LLM).
    
    Útil como fallback ou para testes quando LLM não disponível.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Formatar sintomas
    symptoms_list = "\n".join(
        f"  - {s.name} ({s.severity})"
        for s in symptoms
    ) if symptoms else "  - Nenhum sintoma identificado"
    
    # Mapear nível de risco
    level_map = {
        "emergency": "🔴 EMERGÊNCIA",
        "urgent": "🟠 URGENTE",
        "routine": "🟢 ROTINA",
    }
    level_display = level_map.get(risk.level, risk.level.upper())
    
    return f"""
═══════════════════════════════════════════════════════════════════
                    PRONTUÁRIO DE TELEMEDICINA
═══════════════════════════════════════════════════════════════════
Data/Hora: {timestamp}

📋 SUBJETIVO (S)
----------------
Relato do paciente:
{transcription[:500]}{"..." if len(transcription) > 500 else ""}

📊 OBJETIVO (O)
---------------
Sintomas identificados:
{symptoms_list}

Classificação de Risco: {level_display}
Confiança: {risk.confidence:.0%}

🔍 AVALIAÇÃO (A)
----------------
{risk.rationale}

Red Flags: {", ".join(risk.red_flags) if risk.red_flags else "Nenhum identificado"}

📝 PLANO (P)
------------
Conduta conforme classificação de risco.
{
    "- ENCAMINHAMENTO IMEDIATO para emergência"
    if risk.level == "emergency"
    else "- Avaliação presencial em até 1 hora"
    if risk.level == "urgent"
    else "- Agendamento de consulta programada"
}

═══════════════════════════════════════════════════════════════════
CLASSIFICAÇÃO: {level_display}
═══════════════════════════════════════════════════════════════════
"""


# ============================================================================
# CLI PARA TESTES
# ============================================================================

if __name__ == "__main__":
    import asyncio
    from src.agents.graph import SymptomEntity, RiskAssessment
    
    async def main():
        transcription = """
        Paciente relata dor de cabeça há 2 dias, de intensidade moderada,
        localizada na região frontal. A dor piora com luz forte (fotofobia).
        Também refere náuseas leves. Nega febre ou vômitos.
        Já usou paracetamol com melhora parcial.
        """
        
        symptoms = [
            SymptomEntity(
                name="cefaleia",
                severity="medium",
                body_region="frontal",
                duration="2 dias",
            ),
            SymptomEntity(
                name="fotofobia",
                severity="medium",
            ),
            SymptomEntity(
                name="nausea",
                severity="low",
            ),
        ]
        
        risk = RiskAssessment(
            level="routine",
            confidence=0.85,
            rationale="Cefaleia tensional típica sem sinais de alarme.",
            red_flags=[],
        )
        
        print("Gerando prontuário SOAP...")
        print("=" * 60)
        
        soap = await generate_soap(transcription, symptoms, risk)
        print(soap)
    
    asyncio.run(main())
