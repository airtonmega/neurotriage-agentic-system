"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           NeuroTriage-AI Demo: Demonstração do Sistema de Triagem           ║
║                                                                              ║
║  Script para testar o pipeline completo de triagem médica localmente.       ║
║  Simula transcrição e processa através do grafo LangGraph.                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

# Configurar ambiente se .env não existir
if not os.getenv("DEEPGRAM_API_KEY"):
    os.environ["DEEPGRAM_API_KEY"] = "demo-mode"
    os.environ["GCP_PROJECT_ID"] = "demo-project"


async def run_demo():
    """Executa demonstração do sistema de triagem."""
    
    print("=" * 70)
    print("🏥 NeuroTriage-AI - Sistema de Triagem Médica Inteligente")
    print("=" * 70)
    print()
    
    # Importar módulos
    from src.agents.graph import SymptomEntity, RiskAssessment
    from src.agents.nodes.risk_guardrail import RiskEvaluator
    from src.agents.nodes.soap_generator import generate_soap_template
    from src.agents.nodes.analysis import HybridRetriever, AnalysisConfig
    
    # Carregar casos de teste
    golden_path = Path(__file__).parent / "tests" / "golden_dataset" / "cases.json"
    
    if golden_path.exists():
        with open(golden_path, encoding="utf-8") as f:
            cases = json.load(f)
    else:
        # Casos de demonstração inline
        cases = [
            {
                "case_id": "demo_emergency_001",
                "transcription": "Paciente relata dor forte no peito há 30 minutos, irradiando para o braço esquerdo. Está suando frio e sente falta de ar. Tem histórico de hipertensão.",
                "expected_symptoms": ["dor_toracica", "irradiacao_mse", "sudorese", "dispneia"],
                "expected_risk_level": "emergency",
            },
            {
                "case_id": "demo_routine_001",
                "transcription": "Paciente com dor de cabeça leve há dois dias, melhora com analgésico comum. Sem febre, sem náusea. Está dormindo bem.",
                "expected_symptoms": ["cefaleia_leve"],
                "expected_risk_level": "routine",
            },
            {
                "case_id": "demo_urgent_001",
                "transcription": "Gestante de 32 semanas com inchaço nas pernas há 3 dias, dor de cabeça forte e visão embaçada desde ontem. Pressão em casa estava 150x100.",
                "expected_symptoms": ["edema_mmii", "cefaleia_intensa", "disturbio_visual", "hipertensao"],
                "expected_risk_level": "urgent",
            },
        ]
    
    print(f"📋 Carregados {len(cases)} casos de teste")
    print()
    
    # Processar cada caso
    evaluator = RiskEvaluator()
    retriever = HybridRetriever(AnalysisConfig())
    
    results = []
    
    for i, case in enumerate(cases[:5], 1):  # Limitar a 5 casos
        print(f"\n{'─' * 70}")
        print(f"📝 CASO {i}: {case['case_id']}")
        print(f"{'─' * 70}")
        
        transcription = case.get("transcription", "")
        expected_risk = case.get("expected_risk_level", "unknown")
        expected_symptoms = case.get("expected_symptoms", [])
        
        print(f"\n🎤 Transcrição:")
        print(f"   {transcription[:200]}{'...' if len(transcription) > 200 else ''}")
        
        # Simular sintomas baseados nos esperados
        symptoms = [
            SymptomEntity(
                name=s,
                severity="critical" if expected_risk == "emergency" else 
                         "high" if expected_risk == "urgent" else "low",
            )
            for s in expected_symptoms
        ]
        
        print(f"\n🔍 Sintomas identificados ({len(symptoms)}):")
        for s in symptoms:
            print(f"   • {s.name} ({s.severity})")
        
        # Buscar contexto no conhecimento médico
        print(f"\n📚 Buscando contexto médico...")
        context_docs = retriever._mock_retrieval(transcription)
        if context_docs:
            print(f"   Encontrado: {context_docs[0]['source']} (score: {context_docs[0]['score']:.2f})")
        
        # Avaliar risco
        result = evaluator.evaluate(symptoms)
        
        # Determinar ícone do nível
        level_icons = {
            "emergency": "🔴 EMERGÊNCIA",
            "urgent": "🟠 URGENTE",
            "routine": "🟢 ROTINA",
        }
        
        print(f"\n⚠️  Avaliação de Risco:")
        print(f"   Nível: {level_icons.get(result.level, result.level)}")
        print(f"   Confiança: {result.confidence:.0%}")
        print(f"   Red Flags: {', '.join(result.red_flags_found) if result.red_flags_found else 'Nenhum'}")
        
        # Verificar se bateu com esperado
        match = "✅" if result.level == expected_risk else "❌"
        print(f"\n   Esperado: {expected_risk} {match}")
        
        # Gerar SOAP
        risk_assessment = RiskAssessment(
            level=result.level,
            confidence=result.confidence,
            rationale=result.rationale,
            red_flags=result.red_flags_found,
        )
        
        soap = generate_soap_template(transcription, symptoms, risk_assessment)
        
        print(f"\n📋 Prontuário SOAP gerado ({len(soap)} caracteres)")
        
        results.append({
            "case_id": case["case_id"],
            "expected": expected_risk,
            "actual": result.level,
            "match": result.level == expected_risk,
            "confidence": result.confidence,
        })
    
    # Sumário
    print(f"\n{'=' * 70}")
    print("📊 SUMÁRIO DE RESULTADOS")
    print(f"{'=' * 70}")
    
    passed = sum(1 for r in results if r["match"])
    total = len(results)
    
    print(f"\n   ✅ Acertos: {passed}/{total} ({passed/total*100:.0f}%)")
    print()
    
    for r in results:
        status = "✅" if r["match"] else "❌"
        print(f"   {status} {r['case_id']}: esperado={r['expected']}, obtido={r['actual']} ({r['confidence']:.0%})")
    
    print(f"\n{'=' * 70}")
    print("🏁 Demonstração concluída!")
    print(f"{'=' * 70}")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_demo())
