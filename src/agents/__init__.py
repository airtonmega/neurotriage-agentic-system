# ============================================================================
# NeuroTriage-AI: Sistema Agêntico de Triagem para Telemedicina
# ============================================================================
#
# 📚 O QUE É ESTE ARQUIVO?
# ------------------------
# Este é o arquivo __init__.py do pacote "agents".
# Em Python, __init__.py faz duas coisas:
# 
# 1. MARCA O DIRETÓRIO COMO PACOTE
#    Sem este arquivo, Python não reconhece a pasta como importável.
#    É como colocar uma placa "Esta é uma biblioteca" na porta.
#
# 2. DEFINE O QUE É EXPORTADO
#    Quando alguém faz "from src.agents import X", 
#    este arquivo controla o que "X" pode ser.
#
# 🎯 COMO USAR:
# -------------
# Depois de ter este arquivo, você pode fazer:
#
#     from src.agents import build_triage_graph
#     from src.agents import DatabricksCheckpointer
#
# Em vez do caminho completo:
#
#     from src.agents.graph import build_triage_graph
#     from src.agents.checkpointer import DatabricksCheckpointer
#
# ============================================================================

# Veja a documentação completa em README.md
# Acesse os componentes individuais:
#   - graph.py: Definição do grafo LangGraph
#   - checkpointer.py: Persistência de estado no Databricks
