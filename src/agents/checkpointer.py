"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              Databricks Checkpointer: Salvando Estado do Agente              ║
║                                                                              ║
║  Este arquivo permite "pausar" e "continuar" o processamento do agente.     ║
║  É como o "save game" em um videogame - você pode parar e voltar depois!    ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 O QUE ESTE ARQUIVO FAZ?
---------------------------
Imagine que o agente está no meio de uma triagem quando:
- O computador desliga
- Um humano precisa revisar
- O paciente pausou a consulta

Sem o Checkpointer, teríamos que começar tudo de novo!
Com ele, salvamos o "estado" atual e continuamos depois.

📚 CONCEITOS IMPORTANTES:
-------------------------
- CHECKPOINT: Um "snapshot" (foto) do estado em um momento
- STATE: Todas as informações do agente naquele momento
- THREAD_ID: Identificador único da conversa
- THREAD_TS: Timestamp (momento exato) de cada checkpoint
- DELTA LAKE: Formato especial de armazenamento do Databricks (melhor que CSV/JSON)

🔗 DEPENDÊNCIAS:
----------------
- langgraph: Framework de agentes
- pyspark: Para acessar o Databricks/Delta Lake
- pickle: Serialização de objetos Python
"""

# ============================================================================
# IMPORTS
# ============================================================================

from __future__ import annotations

# json: Converter dados Python ↔ JSON (texto)
import json

# pickle: Converter objetos Python ↔ bytes (para salvar em arquivo)
# É como "esmagar" um objeto complexo em bytes para guardar, e depois "desesmagar"
import pickle

# datetime: Trabalhar com datas e horas
from datetime import datetime, timezone

# typing: Definir tipos das variáveis
from typing import Any, Iterator, Sequence

# dataclass: Criar classes simples para dados
from dataclasses import dataclass

# LangGraph - Interfaces do checkpointer
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,    # Classe base que precisamos herdar
    Checkpoint,             # O checkpoint em si
    CheckpointMetadata,     # Metadados sobre o checkpoint
    CheckpointTuple,        # Tupla com checkpoint + metadados
)

# Pydantic para validação (não usado diretamente, mas pode ser útil)
from pydantic import BaseModel


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

@dataclass
class DatabricksCheckpointerConfig:
    """
    ⚙️ Configurações do checkpointer.
    
    @dataclass cria automaticamente __init__ e outras funções úteis.
    É uma forma prática de criar classes que só guardam dados.
    
    Atributos:
        catalog: Nome do catálogo no Unity Catalog (ex: "healthcare_prod")
        schema: Nome do schema dentro do catálogo (ex: "triage_v1")
        table: Nome da tabela de checkpoints (ex: "langgraph_checkpoints")
    
    Exemplo:
        >>> config = DatabricksCheckpointerConfig(
        ...     catalog="healthcare_prod",
        ...     schema="triage_v1",
        ...     table="checkpoints"
        ... )
        >>> print(config.full_table_name)
        "healthcare_prod.triage_v1.checkpoints"
    """
    catalog: str    # Unity Catalog (nível mais alto de organização)
    schema: str     # Schema (agrupamento de tabelas relacionadas)
    table: str = "langgraph_checkpoints"  # Nome da tabela (tem valor padrão)
    
    @property
    def full_table_name(self) -> str:
        """
        Retorna o nome completo da tabela no formato Databricks.
        
        Formato: catalog.schema.table
        Exemplo: healthcare_prod.triage_v1.langgraph_checkpoints
        """
        return f"{self.catalog}.{self.schema}.{self.table}"


# ============================================================================
# IMPLEMENTAÇÃO PRINCIPAL
# ============================================================================

class DatabricksCheckpointer(BaseCheckpointSaver):
    """
    💾 Checkpointer que salva estado no Databricks Delta Lake.
    
    Esta classe herda de BaseCheckpointSaver (do LangGraph) e implementa
    os métodos necessários para salvar/carregar checkpoints.
    
    POR QUE DELTA LAKE?
    - ACID: Transações seguras (não corrompe dados)
    - Time Travel: Pode voltar a versões anteriores
    - Unity Catalog: Governança e controle de acesso
    - Performance: Otimizado para grandes volumes
    
    COMO USAR:
    
        >>> # 1. Criar o checkpointer
        >>> checkpointer = DatabricksCheckpointer(
        ...     catalog="healthcare",
        ...     schema="triage",
        ...     table="checkpoints"
        ... )
        >>> 
        >>> # 2. Usar ao criar o grafo
        >>> from src.agents.graph import build_triage_graph
        >>> graph = build_triage_graph(checkpointer=checkpointer)
        >>> 
        >>> # 3. Executar com thread_id para rastrear
        >>> result = await graph.ainvoke(
        ...     initial_state,
        ...     config={"configurable": {"thread_id": "conv_123"}}
        ... )
        >>> 
        >>> # 4. Depois, pode retomar de onde parou!
        >>> result = await graph.ainvoke(
        ...     None,  # Sem estado inicial = carrega do checkpoint
        ...     config={"configurable": {"thread_id": "conv_123"}}
        ... )
    """
    
    def __init__(
        self,
        catalog: str,
        schema: str,
        table: str = "langgraph_checkpoints",
    ):
        """
        Inicializa o checkpointer.
        
        Args:
            catalog: Nome do Unity Catalog (ex: "healthcare_prod")
            schema: Nome do schema (ex: "triage_v1")
            table: Nome da tabela (default: "langgraph_checkpoints")
        
        O que acontece:
        1. Chama o __init__ da classe pai (BaseCheckpointSaver)
        2. Salva as configurações
        3. Cria a tabela no Databricks se não existir
        """
        # Chama construtor da classe pai
        super().__init__()
        
        # Salva configurações em objeto organizado
        self.config = DatabricksCheckpointerConfig(
            catalog=catalog,
            schema=schema,
            table=table,
        )
        
        # SparkSession será criado sob demanda (lazy)
        self._spark = None
        
        # Cria a tabela se não existir
        self._ensure_table_exists()
    
    @property
    def spark(self):
        """
        🔥 Retorna o SparkSession (conexão com Databricks).
        
        @property transforma o método em um atributo.
        Em vez de chamar objeto.spark(), você usa objeto.spark
        
        "Lazy loading": Só cria a conexão quando realmente precisar.
        Isso evita criar conexões desnecessárias.
        
        Returns:
            SparkSession conectado ao Databricks
        
        Raises:
            ImportError: Se PySpark não estiver instalado
        """
        if self._spark is None:
            try:
                from pyspark.sql import SparkSession
                # getOrCreate: Pega sessão existente ou cria nova
                self._spark = SparkSession.builder.getOrCreate()
            except ImportError:
                raise ImportError(
                    "PySpark é necessário para DatabricksCheckpointer. "
                    "Instale com: pip install pyspark"
                )
        return self._spark
    
    def _ensure_table_exists(self) -> None:
        """
        📋 Cria a tabela de checkpoints se não existir.
        
        A tabela tem a seguinte estrutura:
        - thread_id: Identificador da conversa
        - thread_ts: Timestamp do checkpoint (ISO format)
        - parent_ts: Timestamp do checkpoint anterior (para histórico)
        - checkpoint_data: Os dados serializados (BINARY)
        - metadata_json: Metadados em JSON
        - created_at: Quando foi criado
        
        TBLPROPERTIES:
        - enableChangeDataFeed: Permite rastrear mudanças
        - autoOptimize: Otimiza automaticamente para performance
        """
        # SQL para criar a tabela
        # Usamos CREATE TABLE IF NOT EXISTS para não dar erro se já existir
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.config.full_table_name} (
            thread_id STRING NOT NULL,
            thread_ts STRING NOT NULL,
            parent_ts STRING,
            checkpoint_data BINARY NOT NULL,
            metadata_json STRING,
            created_at TIMESTAMP NOT NULL,
            PRIMARY KEY (thread_id, thread_ts)
        )
        USING DELTA
        TBLPROPERTIES (
            'delta.enableChangeDataFeed' = 'true',
            'delta.autoOptimize.optimizeWrite' = 'true',
            'delta.autoOptimize.autoCompact' = 'true'
        )
        COMMENT 'LangGraph checkpoints para NeuroTriage-AI'
        """
        
        try:
            self.spark.sql(create_table_sql)
        except Exception as e:
            # Ignora erro se tabela já existe ou se não estamos no Databricks
            if "already exists" not in str(e).lower():
                # Apenas avisa, não falha (pode estar em ambiente local de testes)
                import warnings
                warnings.warn(f"Não foi possível criar tabela de checkpoint: {e}")
    
    def put(
        self,
        config: dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> dict[str, Any]:
        """
        💾 SALVA um checkpoint no Databricks.
        
        Este método é chamado automaticamente pelo LangGraph quando
        o estado muda e precisa ser salvo.
        
        Args:
            config: Configuração com thread_id (identificador da conversa)
            checkpoint: O estado atual do grafo
            metadata: Informações extras sobre o checkpoint
        
        Returns:
            Config atualizado com o thread_ts do novo checkpoint
        
        COMO FUNCIONA:
        1. Extrai o thread_id do config
        2. Gera um timestamp único (thread_ts)
        3. Serializa o checkpoint com pickle (converte objeto → bytes)
        4. Converte metadata para JSON
        5. Insere na tabela Delta
        6. Retorna o config atualizado
        
        Exemplo de uso (pelo LangGraph internamente):
            >>> config = {"configurable": {"thread_id": "conv_123"}}
            >>> checkpointer.put(config, current_state, metadata)
        """
        # Extrai o thread_id do dicionário de configuração
        # "configurable" é onde o LangGraph guarda essas informações
        thread_id = config["configurable"]["thread_id"]
        
        # Gera timestamp único para este checkpoint
        # .isoformat() converte para texto: "2025-01-24T22:30:00+00:00"
        thread_ts = datetime.now(timezone.utc).isoformat()
        
        # Pega o timestamp do checkpoint anterior (se existir)
        parent_ts = config["configurable"].get("thread_ts")
        
        # SERIALIZAÇÃO: Converte o objeto checkpoint em bytes
        # pickle.dumps(objeto) → bytes
        # pickle.loads(bytes) → objeto (para recuperar depois)
        checkpoint_data = pickle.dumps(checkpoint)
        
        # Converte metadata para JSON (mais legível que pickle)
        metadata_json = json.dumps(metadata) if metadata else None
        
        # Monta e executa o SQL de inserção
        insert_sql = f"""
        INSERT INTO {self.config.full_table_name}
        (thread_id, thread_ts, parent_ts, checkpoint_data, metadata_json, created_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP())
        """
        
        self.spark.sql(
            insert_sql,
            [thread_id, thread_ts, parent_ts, checkpoint_data, metadata_json]
        )
        
        # Retorna config atualizado com o timestamp do novo checkpoint
        return {
            "configurable": {
                "thread_id": thread_id,
                "thread_ts": thread_ts,
            }
        }
    
    def get(self, config: dict[str, Any]) -> CheckpointTuple | None:
        """
        📂 CARREGA um checkpoint do Databricks.
        
        Busca e retorna um checkpoint salvo anteriormente.
        
        Args:
            config: Configuração com thread_id e opcionalmente thread_ts
                   - Se thread_ts não for fornecido, retorna o mais recente
        
        Returns:
            CheckpointTuple com o checkpoint e metadados
            None se não encontrar
        
        Exemplo:
            >>> # Carregar checkpoint específico
            >>> config = {"configurable": {"thread_id": "conv_123", "thread_ts": "2025-01-24T22:30:00"}}
            >>> checkpoint = checkpointer.get(config)
            >>> 
            >>> # Carregar checkpoint mais recente
            >>> config = {"configurable": {"thread_id": "conv_123"}}
            >>> checkpoint = checkpointer.get(config)  # Pega o último
        """
        thread_id = config["configurable"]["thread_id"]
        thread_ts = config["configurable"].get("thread_ts")
        
        if thread_ts:
            # Busca checkpoint ESPECÍFICO pelo timestamp
            query = f"""
            SELECT thread_ts, parent_ts, checkpoint_data, metadata_json
            FROM {self.config.full_table_name}
            WHERE thread_id = ? AND thread_ts = ?
            """
            result = self.spark.sql(query, [thread_id, thread_ts]).collect()
        else:
            # Busca checkpoint MAIS RECENTE
            query = f"""
            SELECT thread_ts, parent_ts, checkpoint_data, metadata_json
            FROM {self.config.full_table_name}
            WHERE thread_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """
            result = self.spark.sql(query, [thread_id]).collect()
        
        # Se não encontrou nada, retorna None
        if not result:
            return None
        
        # Pega a primeira (e única) linha
        row = result[0]
        
        # DESERIALIZAÇÃO: Converte bytes de volta para objeto
        checkpoint = pickle.loads(row.checkpoint_data)
        
        # Converte JSON de volta para dicionário
        metadata = json.loads(row.metadata_json) if row.metadata_json else {}
        
        # Monta e retorna o CheckpointTuple
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "thread_ts": row.thread_ts,
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            # parent_config aponta para o checkpoint anterior (para navegação)
            parent_config={
                "configurable": {
                    "thread_id": thread_id,
                    "thread_ts": row.parent_ts,
                }
            } if row.parent_ts else None,
        )
    
    def list(
        self,
        config: dict[str, Any],
        *,
        filter: dict[str, Any] | None = None,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """
        📋 LISTA todos os checkpoints de uma conversa.
        
        Retorna um "Iterator" (gerador) de checkpoints.
        Iterator é mais eficiente que lista quando há muitos itens,
        pois não carrega tudo na memória de uma vez.
        
        Args:
            config: Configuração com thread_id
            filter: Filtros adicionais (não implementado)
            before: Retornar apenas checkpoints anteriores a este
            limit: Número máximo de resultados
        
        Yields:
            CheckpointTuple para cada checkpoint encontrado
        
        Exemplo:
            >>> config = {"configurable": {"thread_id": "conv_123"}}
            >>> for checkpoint in checkpointer.list(config, limit=10):
            ...     print(f"Timestamp: {checkpoint.config['configurable']['thread_ts']}")
        """
        thread_id = config["configurable"]["thread_id"]
        
        # Monta query base
        query = f"""
        SELECT thread_ts, parent_ts, checkpoint_data, metadata_json
        FROM {self.config.full_table_name}
        WHERE thread_id = ?
        """
        
        params = [thread_id]
        
        # Adiciona filtro "before" se fornecido
        if before:
            before_ts = before["configurable"]["thread_ts"]
            query += " AND thread_ts < ?"
            params.append(before_ts)
        
        # Ordena do mais recente para o mais antigo
        query += " ORDER BY created_at DESC"
        
        # Adiciona limite se fornecido
        if limit:
            query += f" LIMIT {limit}"
        
        # Executa query
        result = self.spark.sql(query, params).collect()
        
        # Itera sobre resultados e retorna checkpoints
        # "yield" transforma esta função em um generator (Iterator)
        for row in result:
            checkpoint = pickle.loads(row.checkpoint_data)
            metadata = json.loads(row.metadata_json) if row.metadata_json else {}
            
            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "thread_ts": row.thread_ts,
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config={
                    "configurable": {
                        "thread_id": thread_id,
                        "thread_ts": row.parent_ts,
                    }
                } if row.parent_ts else None,
            )
    
    def get_tuple(self, config: dict[str, Any]) -> CheckpointTuple | None:
        """
        Alias (apelido) para get().
        
        Existe para compatibilidade com diferentes versões do LangGraph.
        Faz exatamente a mesma coisa que get().
        """
        return self.get(config)
    
    def put_writes(
        self,
        config: dict[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """
        📝 Salva writes intermediários.
        
        Writes são mudanças individuais que acontecem durante uma execução.
        Útil para debugging detalhado.
        
        Esta implementação atual é simplificada (não salva nada).
        Em produção, você pode querer uma tabela separada para writes.
        
        Args:
            config: Configuração com thread_id
            writes: Lista de tuplas (chave, valor) das mudanças
            task_id: ID da tarefa que gerou os writes
        """
        # TODO: Implementar se necessário para debugging avançado
        pass


# ============================================================================
# FUNÇÕES DE FÁBRICA
# ============================================================================

def create_production_checkpointer() -> DatabricksCheckpointer:
    """
    🏭 Cria checkpointer para ambiente de PRODUÇÃO.
    
    Usa variáveis de ambiente para pegar as configurações.
    Isso permite mudar configurações sem alterar código.
    
    Variáveis de ambiente usadas:
    - DATABRICKS_CATALOG (default: healthcare_prod)
    - DATABRICKS_SCHEMA (default: triage_v1)
    
    Returns:
        Checkpointer configurado para produção
    """
    import os
    
    return DatabricksCheckpointer(
        catalog=os.getenv("DATABRICKS_CATALOG", "healthcare_prod"),
        schema=os.getenv("DATABRICKS_SCHEMA", "triage_v1"),
        table="langgraph_checkpoints",
    )


def create_dev_checkpointer() -> DatabricksCheckpointer:
    """
    🧪 Cria checkpointer para ambiente de DESENVOLVIMENTO.
    
    Usa tabelas separadas das de produção para evitar
    misturar dados de teste com dados reais.
    
    Returns:
        Checkpointer configurado para desenvolvimento
    """
    return DatabricksCheckpointer(
        catalog="healthcare_dev",
        schema="triage_dev",
        table="langgraph_checkpoints_dev",
    )
