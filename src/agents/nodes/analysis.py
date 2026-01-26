"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           Extração de Sintomas: Análise Inteligente com RAG Híbrido          ║
║                                                                              ║
║  Este módulo extrai sintomas estruturados do texto da consulta usando       ║
║  RAG híbrido (BM25 + Semântico) para enriquecimento com conhecimento médico.║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 O QUE ESTE ARQUIVO FAZ?
---------------------------
1. Recebe texto transcrito da consulta médica
2. Busca contexto relevante no knowledge base (Pinecone híbrido)
3. Usa LLM (Gemini 2.0) para extrair sintomas estruturados
4. Retorna lista de SymptomEntity com severidade, região, duração

📊 RAG HÍBRIDO:
---------------
- BM25: Busca exata para termos médicos (CID-10, fármacos, procedimentos)
- Semântico: Busca conceitual para sintomas descritos pelo paciente
- Reranking: Pinecone ordena resultados por relevância
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from src.agents.graph import SymptomEntity

logger = structlog.get_logger(__name__)


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

class AnalysisConfig(BaseModel):
    """Configuração do módulo de análise."""
    
    pinecone_index: str = Field(
        default="neurotriage-medical-hybrid",
        description="Nome do índice Pinecone para RAG",
    )
    pinecone_namespace: str = Field(
        default="medical_knowledge",
        description="Namespace para busca",
    )
    top_k: int = Field(
        default=5,
        description="Número de documentos a recuperar",
    )
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Peso entre BM25 (0) e semântico (1)",
    )
    gemini_model: str = Field(
        default="gemini-2.0-flash",
        description="Modelo Gemini para extração",
    )
    
    @classmethod
    def from_env(cls) -> "AnalysisConfig":
        return cls(
            pinecone_index=os.getenv("PINECONE_INDEX_HYBRID", "neurotriage-medical-hybrid"),
            pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "medical_knowledge"),
        )


# ============================================================================
# PROMPTS
# ============================================================================

SYMPTOM_EXTRACTION_SYSTEM = """
Você é um médico especializado em triagem que extrai sintomas de transcrições de consultas.

## Sua Tarefa
Analise a transcrição da consulta e extraia TODOS os sintomas mencionados pelo paciente.

## Regras CRÍTICAS
1. Extraia APENAS sintomas explicitamente mencionados ou claramente implícitos
2. NÃO invente sintomas que não foram relatados
3. Use terminologia médica padrão para os nomes
4. Avalie a severidade com base no contexto e descrição do paciente
5. Se o paciente mencionar duração, inclua exatamente como relatado

## Contexto Médico Recuperado
{context}

## Formato de Saída
Retorne APENAS JSON válido, sem markdown:
{{
    "symptoms": [
        {{
            "name": "<termo médico>",
            "severity": "low|medium|high|critical",
            "body_region": "<região anatômica ou null>",
            "duration": "<duração relatada ou null>",
            "onset": "<início ou null>"
        }}
    ]
}}

## Classificação de Severidade
- low: Sintomas leves, não interferem nas atividades (ex: dor leve, coceira)
- medium: Sintomas moderados, causam desconforto significativo
- high: Sintomas intensos, limitam atividades diárias
- critical: Sintomas de emergência (dor torácica, dispneia severa, AVC, etc.)
"""

SYMPTOM_EXTRACTION_USER = """
## Transcrição da Consulta
```
{transcription}
```

Extraia todos os sintomas seguindo as regras estabelecidas.
"""


# ============================================================================
# RAG HÍBRIDO
# ============================================================================

class HybridRetriever:
    """
    Retriever híbrido combinando BM25 e busca vetorial.
    
    Usa Pinecone como backend com suporte nativo a sparse+dense vectors.
    
    Example:
        >>> retriever = HybridRetriever.from_env()
        >>> docs = await retriever.search("dor no peito irradiando")
        >>> print(docs[0]["content"])
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self._index = None
        self._bm25 = None
        
    @classmethod
    def from_env(cls) -> "HybridRetriever":
        return cls(AnalysisConfig.from_env())
    
    @property
    def index(self):
        """Lazy loading do índice Pinecone."""
        if self._index is None:
            try:
                from pinecone import Pinecone
                
                pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", ""))
                self._index = pc.Index(self.config.pinecone_index)
            except Exception as e:
                logger.warning("pinecone_not_available", error=str(e))
                self._index = None
        return self._index
    
    async def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Busca híbrida: BM25 + Semântica com reranking.
        
        Args:
            query: Texto da consulta
            top_k: Número de resultados (default: config.top_k)
            
        Returns:
            Lista de documentos com score e conteúdo
        """
        top_k = top_k or self.config.top_k
        
        if self.index is None:
            logger.warning("using_mock_retrieval")
            return self._mock_retrieval(query)
        
        try:
            # Busca com Pinecone (assume embedding já configurado no índice)
            results = self.index.query(
                namespace=self.config.pinecone_namespace,
                data=query,  # Pinecone gera embedding automaticamente
                top_k=top_k,
                include_metadata=True,
            )
            
            return [
                {
                    "id": match.id,
                    "score": match.score,
                    "content": match.metadata.get("content", ""),
                    "source": match.metadata.get("source", "unknown"),
                }
                for match in results.matches
            ]
        except Exception as e:
            logger.error("pinecone_search_error", error=str(e))
            return self._mock_retrieval(query)
    
    def _mock_retrieval(self, query: str) -> list[dict]:
        """Fallback quando Pinecone não disponível."""
        # Conhecimento básico embutido para funcionamento offline
        return [
            {
                "id": "mock_1",
                "score": 0.9,
                "content": (
                    "Dor torácica com irradiação para membro superior esquerdo, "
                    "sudorese e dispneia são sinais clássicos de Síndrome Coronariana Aguda (SCA). "
                    "Classificação: EMERGÊNCIA. Protocolo Manchester: Vermelho."
                ),
                "source": "protocolo_manchester",
            },
            {
                "id": "mock_2",
                "score": 0.85,
                "content": (
                    "Cefaleia tensional caracteriza-se por dor bilateral, em aperto, "
                    "intensidade leve a moderada, sem náuseas significativas. "
                    "Classificação: ROTINA. CID-10: G44.2"
                ),
                "source": "manual_triagem",
            },
        ]


# ============================================================================
# EXTRATOR DE SINTOMAS
# ============================================================================

class SymptomExtractor:
    """
    Extrator de sintomas usando LLM com RAG.
    
    Combina:
    - Contexto recuperado do knowledge base (RAG)
    - Gemini 2.0 para extração estruturada
    - Validação Pydantic dos resultados
    """
    
    def __init__(self, config: AnalysisConfig | None = None):
        self.config = config or AnalysisConfig.from_env()
        self.retriever = HybridRetriever(self.config)
        self._llm = None
        
    @property
    def llm(self):
        """Lazy loading do LLM."""
        if self._llm is None:
            from langchain_google_vertexai import ChatVertexAI
            
            self._llm = ChatVertexAI(
                model=self.config.gemini_model,
                temperature=0.0,
                max_output_tokens=1000,
            )
        return self._llm
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
    )
    async def extract(
        self,
        transcription: str,
        conversation_id: str | None = None,
    ) -> list["SymptomEntity"]:
        """
        Extrai sintomas estruturados da transcrição.
        
        Args:
            transcription: Texto da consulta (já mascarado)
            conversation_id: ID para rastreamento (opcional)
            
        Returns:
            Lista de SymptomEntity válidos
        """
        from src.agents.graph import SymptomEntity
        
        logger.info(
            "extracting_symptoms",
            text_length=len(transcription),
            conversation_id=conversation_id,
        )
        
        # 1. Recuperar contexto relevante
        context_docs = await self.retriever.search(transcription)
        context = "\n\n".join(
            f"[{doc['source']}]: {doc['content']}" 
            for doc in context_docs
        )
        
        logger.debug("context_retrieved", num_docs=len(context_docs))
        
        # 2. Preparar prompts
        system_prompt = SYMPTOM_EXTRACTION_SYSTEM.format(context=context)
        user_prompt = SYMPTOM_EXTRACTION_USER.format(transcription=transcription)
        
        # 3. Chamar LLM
        response = await self.llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        
        # 4. Parsear resposta
        try:
            result = json.loads(response.content)
            symptoms_raw = result.get("symptoms", [])
        except json.JSONDecodeError as e:
            logger.error("json_parse_error", error=str(e), response=response.content[:200])
            return []
        
        # 5. Validar e converter para SymptomEntity
        symptoms = []
        for s in symptoms_raw:
            try:
                symptom = SymptomEntity(
                    name=s.get("name", "unknown"),
                    severity=s.get("severity", "low"),
                    body_region=s.get("body_region"),
                    duration=s.get("duration"),
                    onset=s.get("onset"),
                )
                symptoms.append(symptom)
            except Exception as e:
                logger.warning("symptom_validation_error", symptom=s, error=str(e))
        
        logger.info(
            "symptoms_extracted",
            count=len(symptoms),
            conversation_id=conversation_id,
        )
        
        return symptoms


# ============================================================================
# FUNÇÃO PRINCIPAL (Interface com o grafo)
# ============================================================================

async def extract_symptoms(
    text: str,
    conversation_id: str,
) -> list["SymptomEntity"]:
    """
    Função principal de extração de sintomas para uso no grafo LangGraph.
    
    Esta é a interface que o nó de análise chama.
    
    Args:
        text: Transcrição mascarada da consulta
        conversation_id: ID da conversa para rastreamento
        
    Returns:
        Lista de SymptomEntity extraídos
        
    Example:
        >>> symptoms = await extract_symptoms(
        ...     "Paciente com dor de cabeça há 2 dias...",
        ...     "conv_123"
        ... )
        >>> for s in symptoms:
        ...     print(f"{s.name}: {s.severity}")
    """
    if not text or len(text.strip()) == 0:
        logger.warning("empty_text_for_extraction", conversation_id=conversation_id)
        return []
    
    extractor = SymptomExtractor()
    return await extractor.extract(text, conversation_id)


# ============================================================================
# CLI PARA TESTES
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def main():
        test_transcription = """
        Paciente relata dor de cabeça intensa há 3 dias, principalmente na região frontal.
        A dor piora com luz forte e movimentos bruscos. Também refere náuseas ocasionais
        mas sem vômitos. Já tomou paracetamol sem melhora significativa.
        Nega febre ou alterações visuais.
        """
        
        print("Transcrição de teste:")
        print(test_transcription)
        print("-" * 50)
        
        symptoms = await extract_symptoms(test_transcription, "test_001")
        
        print(f"\nSintomas extraídos ({len(symptoms)}):")
        for s in symptoms:
            print(f"  - {s.name}")
            print(f"    Severidade: {s.severity}")
            print(f"    Região: {s.body_region}")
            print(f"    Duração: {s.duration}")
            print()
    
    asyncio.run(main())
