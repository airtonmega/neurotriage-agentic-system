"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           Guardrail de Risco: Avaliação Clínica e Detecção de Emergências    ║
║                                                                              ║
║  Implementa protocolo Manchester Triage System adaptado para telemedicina   ║
║  brasileira. Detecta red flags e classifica urgência automaticamente.       ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 O QUE ESTE ARQUIVO FAZ?
---------------------------
1. Recebe lista de sintomas extraídos
2. Avalia red flags (sinais de emergência)
3. Calcula score de risco baseado no Manchester Triage
4. Retorna classificação: routine, urgent, emergency

🚨 RED FLAGS (Emergência Imediata):
-----------------------------------
- Dor torácica / precordial
- Dispneia severa / falta de ar intensa
- Alteração de consciência
- Sintomas de AVC (FAST)
- Sangramento ativo significativo
- Dor abdominal intensa com rigidez
- Síncope / desmaio
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.agents.graph import SymptomEntity, RiskAssessment

logger = structlog.get_logger(__name__)


# ============================================================================
# CONSTANTES: RED FLAGS E SEVERIDADE
# ============================================================================

# Sintomas que indicam EMERGÊNCIA imediata
RED_FLAGS: set[str] = {
    # Cardiovascular
    "dor_toracica", "dor_no_peito", "dor_precordial", "dor_torácica",
    "angina", "infarto", "sca", "sindrome_coronariana",
    "irradiacao_mse", "irradiação_braco_esquerdo",
    
    # Respiratório
    "dispneia_severa", "falta_de_ar_intensa", "insuficiencia_respiratoria",
    "cianose", "saturacao_baixa",
    
    # Neurológico (AVC - FAST)
    "hemiparesia", "hemiplegia", "paralisia_facial", "desvio_rima",
    "disartria", "afasia", "confusao_mental", "alteracao_consciencia",
    "sincope", "desmaio", "convulsao",
    
    # Abdominal
    "abdome_rigido", "dor_abdominal_intensa", "peritonite",
    "hemorragia_digestiva", "hematemese", "melena",
    
    # Trauma/Hemorragia
    "sangramento_ativo", "choque", "hemorragia",
    
    # Outros
    "anafilaxia", "sepse", "febre_alta_crianca",
}

# Sintomas que sugerem URGÊNCIA
URGENT_FLAGS: set[str] = {
    "febre_alta", "febre_persistente",
    "dispneia", "falta_de_ar",
    "dor_abdominal", "vomitos_persistentes",
    "cefaleia_intensa", "cefaleia_subita",
    "gestante", "pre_eclampsia", "eclampsia",
    "tosse_com_sangue", "hemoptise",
    "dor_lombar_intensa",
}

# Mapeamento de severidade para score
SEVERITY_SCORES: dict[str, int] = {
    "critical": 100,
    "high": 70,
    "medium": 40,
    "low": 10,
}


# ============================================================================
# SCHEMA DE RESULTADO
# ============================================================================

class RiskEvaluationResult(BaseModel):
    """Resultado interno da avaliação de risco."""
    
    level: Literal["routine", "urgent", "emergency"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    red_flags_found: list[str]
    total_score: int
    symptoms_analyzed: int


# ============================================================================
# AVALIADOR DE RISCO
# ============================================================================

class RiskEvaluator:
    """
    Avaliador de risco clínico baseado em regras e scoring.
    
    Implementa uma versão simplificada do protocolo Manchester Triage
    adaptada para telemedicina brasileira.
    
    Níveis de classificação:
    - EMERGENCY (Vermelho): Atendimento imediato
    - URGENT (Laranja/Amarelo): Atendimento em até 1h
    - ROUTINE (Verde/Azul): Atendimento programado
    
    Example:
        >>> evaluator = RiskEvaluator()
        >>> result = evaluator.evaluate(symptoms)
        >>> print(f"Nível: {result.level}")
    """
    
    def __init__(
        self,
        emergency_threshold: int = 80,
        urgent_threshold: int = 50,
    ):
        """
        Args:
            emergency_threshold: Score mínimo para classificar como emergência
            urgent_threshold: Score mínimo para classificar como urgente
        """
        self.emergency_threshold = emergency_threshold
        self.urgent_threshold = urgent_threshold
    
    def _normalize_symptom_name(self, name: str) -> str:
        """Normaliza nome do sintoma para matching."""
        return (
            name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("ç", "c")
            .replace("ã", "a")
            .replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
        )
    
    def _check_red_flags(self, symptoms: list["SymptomEntity"]) -> list[str]:
        """Identifica red flags nos sintomas."""
        found = []
        
        for symptom in symptoms:
            normalized = self._normalize_symptom_name(symptom.name)
            
            # Verifica match direto
            if normalized in RED_FLAGS:
                found.append(symptom.name)
                continue
            
            # Verifica match parcial
            for flag in RED_FLAGS:
                if flag in normalized or normalized in flag:
                    found.append(symptom.name)
                    break
            
            # Sintomas críticos são automaticamente red flags
            if symptom.severity == "critical":
                if symptom.name not in found:
                    found.append(symptom.name)
        
        return found
    
    def _check_urgent_flags(self, symptoms: list["SymptomEntity"]) -> list[str]:
        """Identifica sintomas urgentes."""
        found = []
        
        for symptom in symptoms:
            normalized = self._normalize_symptom_name(symptom.name)
            
            if normalized in URGENT_FLAGS:
                found.append(symptom.name)
                continue
            
            for flag in URGENT_FLAGS:
                if flag in normalized or normalized in flag:
                    found.append(symptom.name)
                    break
            
            # Sintomas de alta severidade são urgentes
            if symptom.severity == "high":
                if symptom.name not in found:
                    found.append(symptom.name)
        
        return found
    
    def _calculate_score(self, symptoms: list["SymptomEntity"]) -> int:
        """Calcula score total baseado em severidade."""
        if not symptoms:
            return 0
        
        total = 0
        for symptom in symptoms:
            base_score = SEVERITY_SCORES.get(symptom.severity, 10)
            
            # Multiplicadores
            multiplier = 1.0
            
            # Red flags boost
            normalized = self._normalize_symptom_name(symptom.name)
            if normalized in RED_FLAGS:
                multiplier *= 1.5
            
            # Urgentes boost moderado
            if normalized in URGENT_FLAGS:
                multiplier *= 1.2
            
            total += int(base_score * multiplier)
        
        # Normaliza para máximo de 100
        return min(total, 100)
    
    def evaluate(
        self,
        symptoms: list["SymptomEntity"],
        transcription: str | None = None,
    ) -> RiskEvaluationResult:
        """
        Avalia risco clínico baseado nos sintomas.
        
        Args:
            symptoms: Lista de sintomas extraídos
            transcription: Texto original (para análise adicional)
            
        Returns:
            RiskEvaluationResult com nível, confiança e justificativa
        """
        logger.info(
            "evaluating_risk",
            num_symptoms=len(symptoms),
        )
        
        # Caso sem sintomas
        if not symptoms:
            return RiskEvaluationResult(
                level="routine",
                confidence=0.5,
                rationale="Nenhum sintoma identificado. Classificado como rotina por padrão.",
                red_flags_found=[],
                total_score=0,
                symptoms_analyzed=0,
            )
        
        # Identificar red flags
        red_flags = self._check_red_flags(symptoms)
        urgent_flags = self._check_urgent_flags(symptoms)
        
        # Calcular score
        score = self._calculate_score(symptoms)
        
        # Determinar nível
        level: Literal["routine", "urgent", "emergency"]
        rationale: str
        confidence: float
        
        if red_flags:
            level = "emergency"
            confidence = min(0.95, 0.7 + (len(red_flags) * 0.1))
            rationale = (
                f"EMERGÊNCIA detectada. Red flags identificados: {', '.join(red_flags)}. "
                f"Score de risco: {score}/100. Requer atendimento IMEDIATO."
            )
            
        elif score >= self.emergency_threshold:
            level = "emergency"
            confidence = 0.85
            rationale = (
                f"Score de risco elevado ({score}/100) sem red flags explícitos. "
                f"Combinação de sintomas sugere situação crítica."
            )
            
        elif urgent_flags or score >= self.urgent_threshold:
            level = "urgent"
            confidence = min(0.90, 0.6 + (len(urgent_flags) * 0.1))
            flags_str = ", ".join(urgent_flags) if urgent_flags else "nenhum específico"
            rationale = (
                f"Classificação URGENTE. Sintomas urgentes: {flags_str}. "
                f"Score: {score}/100. Requer avaliação em até 1 hora."
            )
            
        else:
            level = "routine"
            confidence = max(0.5, 1.0 - (score / 100))
            rationale = (
                f"Classificação ROTINA. Sem red flags ou urgências identificadas. "
                f"Score: {score}/100. Pode aguardar atendimento programado."
            )
        
        result = RiskEvaluationResult(
            level=level,
            confidence=confidence,
            rationale=rationale,
            red_flags_found=red_flags,
            total_score=score,
            symptoms_analyzed=len(symptoms),
        )
        
        logger.info(
            "risk_evaluated",
            level=result.level,
            score=result.total_score,
            red_flags=len(red_flags),
            confidence=result.confidence,
        )
        
        return result


# ============================================================================
# FUNÇÃO PRINCIPAL (Interface com o grafo)
# ============================================================================

async def evaluate_risk(
    symptoms: list["SymptomEntity"],
    transcription: str,
) -> "RiskAssessment":
    """
    Função principal de avaliação de risco para uso no grafo LangGraph.
    
    Esta é a interface que o nó de guardrail chama.
    
    Args:
        symptoms: Lista de sintomas extraídos pela análise
        transcription: Texto mascarado da consulta
        
    Returns:
        RiskAssessment com nível, confiança, rationale e red_flags
        
    Example:
        >>> assessment = await evaluate_risk(symptoms, transcription)
        >>> if assessment.level == "emergency":
        ...     trigger_alert()
    """
    from src.agents.graph import RiskAssessment
    
    evaluator = RiskEvaluator()
    result = evaluator.evaluate(symptoms, transcription)
    
    # Converter para RiskAssessment (schema do grafo)
    return RiskAssessment(
        level=result.level,
        confidence=result.confidence,
        rationale=result.rationale,
        red_flags=result.red_flags_found,
    )


# ============================================================================
# CLI PARA TESTES
# ============================================================================

if __name__ == "__main__":
    import asyncio
    from src.agents.graph import SymptomEntity
    
    async def main():
        # Caso 1: Emergência
        emergency_symptoms = [
            SymptomEntity(name="dor_toracica", severity="critical", body_region="tórax"),
            SymptomEntity(name="sudorese", severity="high"),
            SymptomEntity(name="dispneia", severity="high"),
        ]
        
        print("=== CASO 1: Suspeita de IAM ===")
        result = await evaluate_risk(emergency_symptoms, "Dor no peito...")
        print(f"Nível: {result.level}")
        print(f"Confiança: {result.confidence:.0%}")
        print(f"Red flags: {result.red_flags}")
        print(f"Rationale: {result.rationale}")
        print()
        
        # Caso 2: Rotina
        routine_symptoms = [
            SymptomEntity(name="cefaleia_leve", severity="low", duration="2 dias"),
        ]
        
        print("=== CASO 2: Cefaleia leve ===")
        result = await evaluate_risk(routine_symptoms, "Dor de cabeça leve...")
        print(f"Nível: {result.level}")
        print(f"Confiança: {result.confidence:.0%}")
        print(f"Red flags: {result.red_flags}")
        print()
        
        # Caso 3: Urgente
        urgent_symptoms = [
            SymptomEntity(name="febre_alta", severity="high", duration="3 dias"),
            SymptomEntity(name="tosse_produtiva", severity="medium"),
            SymptomEntity(name="dispneia", severity="medium"),
        ]
        
        print("=== CASO 3: Síndrome respiratória ===")
        result = await evaluate_risk(urgent_symptoms, "Febre alta há 3 dias...")
        print(f"Nível: {result.level}")
        print(f"Confiança: {result.confidence:.0%}")
        print(f"Red flags: {result.red_flags}")
    
    asyncio.run(main())
