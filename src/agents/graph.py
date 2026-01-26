"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    NeuroTriage-AI: Grafo de Triagem Médica                   ║
║                                                                              ║
║  Este é o arquivo PRINCIPAL do sistema. Ele define o "cérebro" da IA que    ║
║  processa consultas médicas. Usamos LangGraph para criar um fluxo de        ║
║  processamento inteligente que pode tomar decisões.                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 O QUE ESTE ARQUIVO FAZ?
---------------------------
Imagine uma linha de produção em uma fábrica, onde cada estação faz uma tarefa:
1. Estação 1 (Transcrição): Converte áudio em texto
2. Estação 2 (Análise): Identifica sintomas no texto
3. Estação 3 (Guardrail): Verifica se é emergência
4. Estação 4 (SOAP): Gera o prontuário médico

Este arquivo cria essa "linha de produção" usando LangGraph.

📚 CONCEITOS IMPORTANTES:
-------------------------
- GRAFO: Uma estrutura com "nós" (estações) conectados por "edges" (caminhos)
- NÓ: Cada função que processa parte da informação
- ESTADO: As informações que passam de um nó para outro
- EDGE: A conexão entre dois nós (pode ter condições!)

🔗 DEPENDÊNCIAS:
----------------
- langgraph: Framework para criar agentes de IA com grafos
- pydantic: Biblioteca para validação de dados (garante que os dados estão corretos)
"""

# ============================================================================
# IMPORTS - Bibliotecas que precisamos usar
# ============================================================================

# "from __future__ import annotations" permite usar tipos mais modernos do Python
# Isso é útil para escrever código mais limpo e legível
from __future__ import annotations

# "operator" é uma biblioteca padrão do Python que fornece funções matemáticas
# Usamos "operator.add" para combinar listas automaticamente
import operator

# "typing" fornece ferramentas para dizer que tipo de dado cada variável deve ter
# Isso ajuda a evitar erros e torna o código mais fácil de entender
from typing import (
    Annotated,  # Permite adicionar metadados aos tipos
    TypedDict,  # Cria dicionários com tipos específicos para cada chave
    Literal,    # Limita um valor a opções específicas (ex: "sim" ou "não")
)

# "dataclass" simplifica a criação de classes que armazenam dados
from dataclasses import dataclass, field

# LangGraph - O framework principal para criar nosso grafo de IA
from langgraph.graph import StateGraph, END  # StateGraph cria o grafo, END marca o fim
from langgraph.checkpoint.base import BaseCheckpointSaver  # Para salvar o estado

# Pydantic - Valida os dados automaticamente e gera erros claros se algo estiver errado
from pydantic import BaseModel, Field


# ============================================================================
# DEFINIÇÃO DO ESTADO
# ============================================================================
# O "Estado" é como uma mochila que carrega todas as informações de um nó
# para o outro. Cada nó pode ler o que está na mochila, modificar, e passar
# adiante para o próximo nó.
# ============================================================================

class SymptomEntity(BaseModel):
    """
    🩺 Representa UM sintoma identificado na consulta.
    
    Esta classe é como uma "ficha" que guarda informações sobre cada sintoma.
    O Pydantic (BaseModel) garante que os dados estão no formato correto.
    
    Exemplos de uso:
        >>> sintoma = SymptomEntity(
        ...     name="cefaleia",           # Nome técnico: dor de cabeça
        ...     severity="medium",          # Severidade média
        ...     body_region="cabeça",      # Onde dói
        ...     duration="2 dias"          # Há quanto tempo
        ... )
    """
    
    # "Field(...)" significa que o campo é OBRIGATÓRIO (não pode ser vazio)
    # "description" explica o que o campo significa (útil para documentação)
    
    name: str = Field(
        ...,  # Os "..." significam "obrigatório"
        description="Nome do sintoma em terminologia médica (ex: cefaleia, dispneia)"
    )
    
    # "Literal" limita as opções - só pode ser um desses 4 valores!
    severity: Literal["low", "medium", "high", "critical"] = Field(
        ...,
        description="Severidade: low=leve, medium=moderado, high=alto, critical=crítico"
    )
    
    # "str | None" significa que pode ser texto OU pode ser vazio (None)
    body_region: str | None = Field(
        None,  # None = valor padrão se não for informado
        description="Região do corpo afetada (ex: tórax, abdome, cabeça)"
    )
    
    duration: str | None = Field(
        None,
        description="Há quanto tempo o paciente tem o sintoma (ex: 2 dias, 1 semana)"
    )
    
    onset: str | None = Field(
        None,
        description="Como começou? Súbito ou gradual? (ex: acordou com dor)"
    )


class RiskAssessment(BaseModel):
    """
    ⚠️ Resultado da avaliação de risco do paciente.
    
    Depois de analisar os sintomas, o sistema classifica o risco:
    - routine: Pode esperar, não é urgente
    - urgent: Precisa de atendimento em breve  
    - emergency: EMERGÊNCIA! Atendimento imediato!
    
    Exemplo:
        >>> risco = RiskAssessment(
        ...     level="emergency",
        ...     confidence=0.95,  # 95% de certeza
        ...     rationale="Dor torácica com irradiação sugere SCA",
        ...     red_flags=["dor_toracica", "sudorese"]
        ... )
    """
    
    level: Literal["routine", "urgent", "emergency"] = Field(
        ...,
        description="Nível de urgência conforme protocolo de Manchester"
    )
    
    # "ge=0.0, le=1.0" significa: maior ou igual a 0, menor ou igual a 1
    # Isso garante que a confiança é uma porcentagem válida (0% a 100%)
    confidence: float = Field(
        ...,
        ge=0.0,  # ge = greater or equal (maior ou igual)
        le=1.0,  # le = less or equal (menor ou igual)
        description="Confiança da IA na avaliação (0.0 = 0%, 1.0 = 100%)"
    )
    
    rationale: str = Field(
        ...,
        description="Explicação do porquê dessa classificação de risco"
    )
    
    # "default_factory=list" cria uma lista vazia [] como padrão
    red_flags: list[str] = Field(
        default_factory=list,
        description="Sinais de alerta identificados (ex: dor_toracica, dispneia)"
    )


class TriageState(TypedDict):
    """
    📦 ESTADO PRINCIPAL - A "mochila" que passa por todos os nós.
    
    Esta classe define TODAS as informações que são compartilhadas entre
    os nós do grafo. Cada nó pode ler e modificar esses dados.
    
    TypedDict é como um dicionário Python, mas com tipos definidos:
        estado["audio_chunk_id"] --> sempre será string
        estado["symptoms"] --> sempre será lista de SymptomEntity
    
    🔄 Fluxo de dados:
    1. Entrada: audio_bytes (áudio bruto)
    2. Transcrição: preenche "transcription"
    3. Análise: adiciona itens em "symptoms"
    4. Guardrail: preenche "risk_assessment"
    5. Saída: "soap_note" com o prontuário final
    """
    
    # === ENTRADA (dados que chegam do mundo externo) ===
    audio_chunk_id: str  # ID único para rastrear este pedaço de áudio
    audio_bytes: bytes | None  # O áudio em si (em bytes, formato que o computador entende)
    
    # === PROCESSAMENTO (dados criados durante o processamento) ===
    transcription: str  # Texto original da transcrição
    transcription_masked: str  # Texto com dados pessoais removidos (CPF, nome, etc.)
    
    # === ANÁLISE (resultados da análise de IA) ===
    # "Annotated[list[SymptomEntity], operator.add]" é uma mágica do LangGraph!
    # Significa: quando dois nós adicionam sintomas, COMBINE as listas automaticamente
    # Em vez de substituir, ele junta: [sintoma1] + [sintoma2] = [sintoma1, sintoma2]
    symptoms: Annotated[list[SymptomEntity], operator.add]
    
    risk_assessment: RiskAssessment | None  # Resultado da avaliação de risco
    
    # === SAÍDA (resultado final) ===
    soap_note: str | None  # Prontuário no formato SOAP (Subjetivo, Objetivo, Avaliação, Plano)
    priority_alert: bool  # True = EMERGÊNCIA! Precisa de atenção imediata
    
    # === METADADOS (informações sobre o processamento) ===
    conversation_id: str  # ID único desta conversa/consulta
    iteration_count: int  # Quantas vezes já passamos pelo loop (máximo 3)
    requires_human_review: bool  # True = um humano precisa revisar esta triagem


# ============================================================================
# DEFINIÇÃO DOS NÓS (FUNÇÕES DE PROCESSAMENTO)
# ============================================================================
# Cada função abaixo é um "nó" no grafo. Ela recebe o estado atual,
# faz algum processamento, e retorna as mudanças que devem ser feitas
# no estado.
#
# ⚠️ IMPORTANTE: Os nós NÃO modificam o estado diretamente!
# Eles retornam um dicionário com as mudanças, e o LangGraph aplica.
# ============================================================================

async def transcription_node(state: TriageState) -> dict:
    """
    🎤 NÓ 1: TRANSCRIÇÃO
    
    Este nó converte áudio em texto usando Whisper (IA de transcrição).
    
    Como funciona:
    1. Recebe o áudio em bytes do estado
    2. Envia para o Whisper processar
    3. Retorna o texto transcrito + texto com dados pessoais mascarados
    
    Args:
        state: O estado atual com audio_bytes
    
    Returns:
        Dicionário com as novas informações:
        {
            "transcription": "Olá doutor, estou com dor de cabeça...",
            "transcription_masked": "Olá doutor, estou com dor de cabeça..."
        }
    
    💡 "async def" significa que esta função é ASSÍNCRONA:
       - Ela pode "pausar" enquanto espera o Whisper responder
       - Isso permite processar outras coisas enquanto espera
       - É mais eficiente do que "travar" o programa esperando
    """
    # Importamos aqui dentro para evitar erros se o módulo não existir
    # (útil para testes onde não temos o módulo real)
    from src.agents.nodes.transcription import transcribe_audio
    
    # Chama a função de transcrição com os dados do estado
    # "await" significa "espere o resultado" (pois é assíncrono)
    result = await transcribe_audio(
        audio_bytes=state["audio_bytes"],   # O áudio a ser transcrito
        chunk_id=state["audio_chunk_id"],   # ID para rastreamento
        language="pt-BR",                    # Idioma: português brasileiro
    )
    
    # Retorna as mudanças a serem aplicadas no estado
    # O LangGraph vai fazer: state["transcription"] = result.text
    return {
        "transcription": result.text,
        "transcription_masked": result.masked_text,
    }


async def analysis_node(state: TriageState) -> dict:
    """
    🔍 NÓ 2: ANÁLISE DE SINTOMAS
    
    Este nó usa IA (Gemini 2.0) para identificar sintomas no texto.
    
    Como funciona:
    1. Recebe a transcrição (já sem dados pessoais)
    2. Envia para o Gemini com um prompt especial
    3. O Gemini identifica: "dor de cabeça" → SymptomEntity(name="cefaleia", ...)
    4. Retorna lista de sintomas estruturados
    
    Exemplo de entrada (transcrição):
        "Estou com dor de cabeça há 2 dias, também sinto náusea"
    
    Exemplo de saída:
        [
            SymptomEntity(name="cefaleia", severity="medium", duration="2 dias"),
            SymptomEntity(name="náusea", severity="low")
        ]
    """
    from src.agents.nodes.analysis import extract_symptoms
    
    # Chama o extrator de sintomas
    symptoms = await extract_symptoms(
        text=state["transcription_masked"],  # Texto SEM dados pessoais
        conversation_id=state["conversation_id"],
    )
    
    # Retorna os sintomas encontrados
    # Como declaramos "Annotated[..., operator.add]", os sintomas são ADICIONADOS
    # Se chamarmos este nó múltiplas vezes, os sintomas se acumulam
    return {"symptoms": symptoms}


async def risk_guardrail_node(state: TriageState) -> dict:
    """
    🚨 NÓ 3: GUARDRAIL DE RISCO (O mais importante!)
    
    Este nó é o "guardião" que decide se o paciente está em perigo.
    Ele usa o protocolo de Manchester (padrão internacional de triagem).
    
    SINAIS DE EMERGÊNCIA que disparam alerta:
    - Dor no peito / torácica
    - Falta de ar severa  
    - Alteração de consciência
    - Sangramento ativo
    - Sintomas de AVC (FAST: Face, Arm, Speech, Time)
    
    Também decide se um HUMANO precisa revisar:
    - Se é emergência → humano precisa ver
    - Se confiança < 70% → humano precisa ver
    - Se muitos sinais de alerta → humano precisa ver
    """
    from src.agents.nodes.risk_guardrail import evaluate_risk
    
    # Avalia o risco baseado nos sintomas identificados
    assessment = await evaluate_risk(
        symptoms=state["symptoms"],
        transcription=state["transcription_masked"],
    )
    
    # Lógica para decidir se precisa de revisão humana:
    # 1. É emergência? → SIM, humano precisa ver
    # 2. IA tem menos de 70% de certeza? → SIM, humano precisa ver
    # 3. Mais de 2 sinais de alerta? → SIM, humano precisa ver
    requires_review = (
        assessment.level == "emergency" or
        assessment.confidence < 0.7 or
        len(assessment.red_flags) > 2
    )
    
    return {
        "risk_assessment": assessment,
        "priority_alert": assessment.level == "emergency",
        "requires_human_review": requires_review,
    }


async def soap_generation_node(state: TriageState) -> dict:
    """
    📋 NÓ 4: GERAÇÃO DO PRONTUÁRIO SOAP
    
    SOAP é um formato padrão de prontuário médico:
    - S (Subjetivo): O que o paciente relatou
    - O (Objetivo): O que foi observado/medido
    - A (Avaliação): Diagnóstico/impressão do profissional
    - P (Plano): O que fazer a seguir
    
    Exemplo de saída SOAP:
    
    === SUBJETIVO ===
    Paciente relata dor de cabeça há 2 dias, de intensidade moderada,
    sem melhora com analgésicos comuns.
    
    === OBJETIVO ===
    Paciente consciente, orientado, afebril.
    
    === AVALIAÇÃO ===
    Cefaleia tensional - CID: G44.2
    Risco: ROTINA
    
    === PLANO ===
    1. Analgésico conforme prescrição
    2. Retorno se piora ou novos sintomas
    """
    from src.agents.nodes.soap_generator import generate_soap
    
    # Gera o prontuário SOAP
    soap = await generate_soap(
        transcription=state["transcription_masked"],
        symptoms=state["symptoms"],
        risk=state["risk_assessment"],
    )
    
    return {"soap_note": soap}


# ============================================================================
# FUNÇÕES DE DECISÃO (EDGES CONDICIONAIS)
# ============================================================================
# Estas funções decidem qual caminho o grafo deve seguir.
# São como "sinais de trânsito" que direcionam o fluxo.
# ============================================================================

def should_escalate(state: TriageState) -> Literal["escalate", "continue"]:
    """
    🔀 Decisão: Escalar para emergência ou continuar normalmente?
    
    Esta função é chamada DEPOIS do guardrail de risco.
    Se for emergência, pula direto para o handler de emergência.
    Se não, continua para gerar o prontuário SOAP.
    
    Returns:
        "escalate": É emergência! Vai para emergency_handler
        "continue": Não é emergência, vai para soap_generation
    
    Visualização do fluxo:
    
        risk_guardrail
              │
              ▼
        should_escalate?
           /        \
     "escalate"   "continue"
          │            │
          ▼            ▼
    emergency      soap_generation
       handler
    """
    # Verifica se o alerta de prioridade foi ativado
    if state.get("priority_alert"):
        return "escalate"
    return "continue"


def should_loop(state: TriageState) -> Literal["loop", "end"]:
    """
    🔄 Decisão: Fazer mais uma iteração ou finalizar?
    
    Em casos complexos, podemos querer "refinar" a análise.
    O sistema pode voltar e analisar novamente até 3 vezes.
    
    Quando fazer loop:
    - iteration_count < 3 (não atingiu o máximo)
    - E requires_human_review = True (precisa melhorar)
    
    Returns:
        "loop": Volta para analysis_node para refinar
        "end": Finaliza o processamento
    
    Visualização:
    
        soap_generation
              │
              ▼
         should_loop?
           /        \
        "loop"     "end"
           │          │
           ▼          ▼
      analysis_node   FIM
      (refinar)
    """
    # Se já passou 3 vezes OU não precisa de revisão → finaliza
    if state["iteration_count"] < 3 and state.get("requires_human_review"):
        return "loop"
    return "end"


# ============================================================================
# CONSTRUÇÃO DO GRAFO
# ============================================================================
# Aqui é onde conectamos todos os nós e edges para criar o fluxo completo.
# É como montar um quebra-cabeça onde cada peça é um nó.
# ============================================================================

def build_triage_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> StateGraph:
    """
    🏗️ FUNÇÃO PRINCIPAL: Constrói o grafo de triagem médica.
    
    Esta função monta toda a "linha de produção" conectando os nós.
    
    Args:
        checkpointer: Opcional. Onde salvar o estado entre execuções.
                     Útil para pausar e continuar depois, ou para
                     permitir que um humano revise antes de continuar.
    
    Returns:
        StateGraph: O grafo montado e pronto para usar.
    
    📊 Estrutura do grafo:
    
        ┌─────────────────┐
        │   INÍCIO        │
        │ (entry point)   │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  transcription  │ ← Converte áudio em texto
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │    analysis     │ ← Extrai sintomas
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ risk_guardrail  │ ← Avalia risco
        └────────┬────────┘
                 │
           should_escalate?
              /        \
       "escalate"   "continue"
            │            │
            ▼            ▼
        ┌────────┐  ┌─────────────────┐
        │EMERGÊN-│  │ soap_generation │
        │  CIA   │  └────────┬────────┘
        └───┬────┘           │
            │          should_loop?
            │            /        \
            │       "loop"       "end"
            │          │            │
            │          ▼            │
            │     [volta para      │
            │      analysis]       │
            │                      │
            └──────────┬───────────┘
                       │
                       ▼
                ┌─────────────────┐
                │      FIM        │
                └─────────────────┘
    
    Exemplo de uso:
        >>> graph = build_triage_graph()
        >>> result = await graph.ainvoke({
        ...     "audio_bytes": audio_data,
        ...     "audio_chunk_id": "chunk_001",
        ...     "conversation_id": "conv_xyz",
        ...     "iteration_count": 0,
        ... })
        >>> print(result["soap_note"])
    """
    
    # PASSO 1: Criar o grafo com o tipo de estado que definimos
    # Think of it as: "Crie uma fábrica que use o TriageState como modelo"
    workflow = StateGraph(TriageState)
    
    # PASSO 2: Adicionar os nós (estações de trabalho)
    # add_node("nome", função) → adiciona uma estação chamada "nome"
    workflow.add_node("transcription", transcription_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("risk_guardrail", risk_guardrail_node)
    workflow.add_node("soap_generation", soap_generation_node)
    
    # Nó de emergência - uma função simples (lambda) que só marca o alerta
    # Lambda é uma função "mini" de uma linha: lambda estado: {retorno}
    workflow.add_node("emergency_handler", lambda s: {"priority_alert": True})
    
    # PASSO 3: Definir onde o grafo COMEÇA
    # set_entry_point("nome") → o processamento começa neste nó
    workflow.set_entry_point("transcription")
    
    # PASSO 4: Conectar os nós com edges (caminhos)
    # add_edge("de", "para") → cria uma conexão direta
    workflow.add_edge("transcription", "analysis")  # Transcrição → Análise
    workflow.add_edge("analysis", "risk_guardrail")  # Análise → Guardrail
    
    # PASSO 5: Edges CONDICIONAIS (caminhos com decisão)
    # add_conditional_edges("nó", função_decisão, {resultado: destino})
    workflow.add_conditional_edges(
        "risk_guardrail",      # Depois deste nó...
        should_escalate,        # ...executa esta função para decidir...
        {
            "escalate": "emergency_handler",  # Se retornar "escalate", vai para cá
            "continue": "soap_generation",    # Se retornar "continue", vai para cá
        }
    )
    
    # Outro edge condicional: decidir se faz loop ou termina
    workflow.add_conditional_edges(
        "soap_generation",
        should_loop,
        {
            "loop": "analysis",  # Volta para refinar a análise
            "end": END,          # END é uma constante especial que marca o fim
        }
    )
    
    # O handler de emergência sempre termina
    workflow.add_edge("emergency_handler", END)
    
    # PASSO 6: Compilar o grafo (prepara para execução)
    # Se tiver checkpointer, ele será usado para salvar estado
    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    
    return workflow.compile()


# ============================================================================
# FUNÇÃO DE FÁBRICA PARA PRODUÇÃO
# ============================================================================

def create_production_graph() -> StateGraph:
    """
    🏭 Cria o grafo configurado para ambiente de PRODUÇÃO.
    
    Esta função é um "atalho" que cria o grafo com todas as
    configurações corretas para rodar em produção:
    - Checkpointer do Databricks para salvar estado
    - Configurações de retry
    - Tracing do MLFlow
    
    Uso:
        >>> graph = create_production_graph()
        >>> # Agora está pronto para processar consultas de verdade!
    """
    from src.agents.checkpointer import DatabricksCheckpointer
    
    # Cria o checkpointer apontando para as tabelas de produção
    checkpointer = DatabricksCheckpointer(
        catalog="healthcare_prod",    # Catálogo do Unity Catalog
        schema="triage_v1",           # Schema onde ficam as tabelas
        table="checkpoints",          # Tabela que guarda os checkpoints
    )
    
    # Retorna o grafo compilado com o checkpointer
    return build_triage_graph(checkpointer=checkpointer)
