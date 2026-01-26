"""
🏥 NeuroTriage-AI: Demo Interface Moderna
--------------------------------------------------
Uma interface demonstrativa focada em UX/UI premium para
explicar cada etapa do pipeline de triagem médica.
"""

import streamlit as st
import time
import requests
import json
from streamlit_lottie import st_lottie

# Configuração da Página
st.set_page_config(
    page_title="NeuroTriage-AI | Demonstração",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# ESTILOS E CSS (Moderno, Clean, Glassmorphism)
# ============================================================================
st.markdown("""
<style>
    /* Fontes e Cores Globais */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #1e293b;
    }
    
    .stApp {
        background-color: #f8fafc;
        background-image: 
            radial-gradient(at 0% 0%, hsla(210,100%,93%,1) 0, transparent 50%), 
            radial-gradient(at 100% 0%, hsla(160,100%,92%,1) 0, transparent 50%), 
            radial-gradient(at 100% 100%, hsla(210,100%,96%,1) 0, transparent 50%), 
            radial-gradient(at 0% 100%, hsla(160,100%,96%,1) 0, transparent 50%);
    }

    /* Cards de Pipeline */
    .pipeline-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
        margin-bottom: 20px;
    }
    
    .pipeline-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    .pipeline-active {
        border-left: 5px solid #3b82f6; /* Blue 500 */
        background: rgba(255, 255, 255, 0.95);
    }
    
    .pipeline-success {
        border-left: 5px solid #10b981; /* Emerald 500 */
    }
    
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge-processing { background: #e0f2fe; color: #0284c7; }
    .badge-done { background: #dcfce7; color: #16a34a; }
    
    /* Input Personalizado */
    .stTextArea textarea {
        border-radius: 12px;
        border: 1px solid #cbd5e1;
        padding: 15px;
        box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.06);
        transition: border-color 0.2s;
    }
    
    .stTextArea textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
    }
    
    /* Botão Principal */
    .stButton button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        font-weight: 600;
        border-radius: 9999px;
        padding: 0.75rem 2rem;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.4);
        transition: all 0.2s;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.5);
    }

    /* Result Cards */
    .result-emergency { border: 2px solid #ef4444; background: #fef2f2; }
    .result-urgent { border: 2px solid #f59e0b; background: #fffbeb; }
    .result-routine { border: 2px solid #10b981; background: #f0fdf4; }

</style>
""", unsafe_allow_html=True)


# ============================================================================
# LÓGICA DE ANIMAÇÕES
# ============================================================================

# URLs de animações Lottie (arquivos JSON)
LOTTIE_ASSETS = {
    "doctor_bot": "https://lottie.host/embed/9e5c4a75-1025-4c01-8b01-8e5f22e7d58a/lottie.json", # Exemplo genérico
    "medical_scan": "https://assets5.lottiefiles.com/packages/lf20_tutvdkg0.json",
    "brain_processing": "https://assets9.lottiefiles.com/packages/lf20_49rdyysj.json",
    "transcription": "https://assets2.lottiefiles.com/private_files/lf30_10z31wav.json",
    "success": "https://assets1.lottiefiles.com/packages/lf20_jbrw3hcz.json",
    "emergency_alert": "https://assets8.lottiefiles.com/packages/lf20_Tkwjw8.json",
}

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Carregar assets
ani_scan = load_lottieurl(LOTTIE_ASSETS["medical_scan"])
ani_brain = load_lottieurl(LOTTIE_ASSETS["brain_processing"])
ani_transcribe = load_lottieurl(LOTTIE_ASSETS["transcription"])
ani_success = load_lottieurl(LOTTIE_ASSETS["success"])

# ============================================================================
# TÍTULO E HERO SECTION
# ============================================================================

col1, col2 = st.columns([1, 4])

with col1:
    if ani_scan:
        st_lottie(ani_scan, height=120, key="hero_anim")
    else:
        st.image("https://img.icons8.com/color/144/heart-monitor.png", width=100)

with col2:
    st.title("NeuroTriage-AI 2.0")
    st.markdown("### Sistema de Triagem Médica Inteligente com **MedGemma 1.5**")
    st.markdown("""
    Este sistema ouve o paciente, transcreve com precisão médica, entende os sintomas
    e classifica a urgência em segundos.
    """)

st.divider()

# ============================================================================
# INPUT AREA
# ============================================================================

st.markdown("#### 🗣️ Descreva o caso clínico")

# Opções de exemplos rápidos
example_case = st.selectbox(
    "Escolha um caso para testar ou digite o seu:",
    [
        "Personalizado (Digite abaixo)",
        "Caso 1: Dor torácica (Emergência)",
        "Caso 2: Cefaleia leve (Rotina)",
        "Caso 3: Gestante Hipertense (Urgente)",
    ]
)

default_text = ""
if example_case == "Caso 1: Dor torácica (Emergência)":
    default_text = "Paciente masculino, 55 anos. Relata dor forte no peito há 30 minutos, irradiando para o braço esquerdo. Está suando frio e com falta de ar."
elif example_case == "Caso 2: Cefaleia leve (Rotina)":
    default_text = "Paciente feminina, 25 anos. Queixa de dor de cabeça leve na região frontal há 2 dias. Melhora com dipirona. Nega febre ou vômitos."
elif example_case == "Caso 3: Gestante Hipertense (Urgente)":
    default_text = "Gestante de 34 semanas. Chegou com pressão arterial de 160x100. Relata visão embaçada e dor na nuca. Edema em membros inferiores."

user_input = st.text_area(
    "Transcrição ou Input de Texto:",
    value=default_text,
    height=150,
    placeholder="Ex: Paciente sentindo dor no peito..."
)

start_btn = st.button("🚀 Iniciar Triagem Inteligente")

# ============================================================================
# PIPELINE VISUALIZATION
# ============================================================================

if start_btn and user_input:
    
    pipeline_col1, pipeline_col2 = st.columns([1, 1.5])
    
    with pipeline_col1:
        st.subheader("⚙️ Pipeline em Tempo Real")
        
        # --- ETAPA 1: Transcrição & Identificação ---
        step1 = st.empty()
        with step1.container():
            st.markdown("""
            <div class="pipeline-card pipeline-active">
                <div class="status-badge badge-processing">Processando</div>
                <h4>1. MedASR Transcriber</h4>
                <p>Identificando termos médicos e mascarando PII...</p>
            </div>
            """, unsafe_allow_html=True)
            if ani_transcribe:
                st_lottie(ani_transcribe, height=100, key="step1_ani")
        
        # Simula delay de rede/processamento
        time.sleep(1.5)
        
        step1.markdown("""
        <div class="pipeline-card pipeline-success">
            <div class="status-badge badge-done">Concluído</div>
            <h4>1. MedASR Transcriber</h4>
            <p>✅ Transcrição médica calibrada.</p>
        </div>
        """, unsafe_allow_html=True)

        # --- ETAPA 2: Extração de Sintomas (MedGemma) ---
        step2 = st.empty()
        with step2.container():
            st.markdown("""
            <div class="pipeline-card pipeline-active">
                <div class="status-badge badge-processing">Analisando</div>
                <h4>2. MedGemma 1.5 Analysis</h4>
                <p>Extraindo sintomas estruturados e CID-10...</p>
            </div>
            """, unsafe_allow_html=True)
            if ani_brain:
                st_lottie(ani_brain, height=100, key="step2_ani")

        # CHAMADA REAL A API
        api_url = "https://neurotriage-ai-10993113678.us-central1.run.app/triage"
        try:
            payload = {"transcription": user_input, "conversation_id": "demo_ui_v2"}
            response = requests.post(api_url, json=payload, timeout=30)
            data = response.json()
            
            # Formatar dados para exibição
            risk_level = data.get("risk_level", "routine").lower()
            confidence = data.get("confidence", 0.0)
            red_flags = data.get("red_flags", [])
            symptoms = data.get("symptoms", [])
            rationale = data.get("rationale", "")
            
        except Exception as e:
            st.error(f"Erro na conexão com API: {str(e)}")
            st.stop()

        step2.markdown(f"""
        <div class="pipeline-card pipeline-success">
            <div class="status-badge badge-done">Concluído</div>
            <h4>2. MedGemma 1.5 Analysis</h4>
            <p>✅ {len(symptoms)} sintomas identificados.</p>
        </div>
        """, unsafe_allow_html=True)

        # --- ETAPA 3: Risco & Decisão ---
        step3 = st.empty()
        with step3.container():
            st.markdown("""
            <div class="pipeline-card pipeline-active">
                <div class="status-badge badge-processing">Validando</div>
                <h4>3. Risk Guardrail</h4>
                <p>Aplicando Protocolo de Manchester...</p>
            </div>
            """, unsafe_allow_html=True)
        
        time.sleep(1)
        
        step3.markdown("""
        <div class="pipeline-card pipeline-success">
            <div class="status-badge badge-done">Decisão Tomada</div>
            <h4>3. Risk Guardrail</h4>
            <p>✅ Protocolo aplicado.</p>
        </div>
        """, unsafe_allow_html=True)

    # ========================================================================
    # RESULTADO FINAL
    # ========================================================================
    with pipeline_col2:
        st.subheader("📋 Resultado da Triagem")
        
        # Determinar cor e ícone baseados no risco
        triage_config = {
            "emergency": {"color": "#ef4444", "icon": "🔴", "title": "EMERGÊNCIA", "class": "result-emergency"},
            "urgent": {"color": "#f59e0b", "icon": "🟠", "title": "URGENTE", "class": "result-urgent"},
            "routine": {"color": "#10b981", "icon": "🟢", "title": "ROTINA", "class": "result-routine"},
        }
        
        tc = triage_config.get(risk_level, triage_config["routine"])
        
        st.markdown(f"""
        <div class="pipeline-card {tc['class']}" style="border-top: 8px solid {tc['color']}">
            <h1 style="color: {tc['color']}; margin-top:0;">{tc['icon']} {tc['title']}</h1>
            <p><strong>Confiança da IA:</strong> {confidence:.0%}</p>
            <p><strong>Raciocínio:</strong> {rationale}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detalhes dos Sintomas
        st.markdown("### 🔍 Sintomas Detectados")
        
        for s in symptoms:
            s_name = s.get('name', 'N/A').replace('_', ' ').title()
            s_severity = s.get('severity', 'low')
            
            badge_color = "#e2e8f0" # gray
            if s_severity == 'critical': badge_color = "#fecaca" # red
            elif s_severity == 'high': badge_color = "#fde68a" # yellow
            
            st.markdown(f"""
            <div style="background: white; padding: 10px; border-radius: 8px; margin-bottom: 5px; border: 1px solid #e2e8f0; display: flex; justify-content: space-between;">
                <strong>{s_name}</strong>
                <span style="background: {badge_color}; padding: 2px 8px; border-radius: 12px; font-size: 0.8em;">{s_severity.upper()}</span>
            </div>
            """, unsafe_allow_html=True)

        if red_flags:
            st.markdown("### ⚠️ Red Flags (Sinais de Alerta)")
            for rf in red_flags:
                 st.markdown(f"""
                <div style="background: #fee2e2; color: #991b1b; padding: 8px; border-radius: 6px; margin-bottom: 5px; font-weight: bold;">
                    ⚠️ {rf.replace('_', ' ').upper()}
                </div>
                """, unsafe_allow_html=True)

        # Download do SOAP
        st.markdown("### 📄 Prontuário Médico (SOAP)")
        st.info("O prontuário foi gerado e enviado para o EMR (Electronic Medical Record).")
        st.download_button(
            label="Baixar Relatório PDF (Simulado)",
            data=json.dumps(data, indent=2),
            file_name="relatorio_triagem.json",
            mime="application/json"
        )

else:
    # Se não processou, mostra info visual
    st.info("👆 Selecione um caso ou digite o texto e clique em 'Iniciar Triagem' para ver a mágica acontecer!")
    
    with st.expander("ℹ️ Como funciona o pipeline?"):
        st.markdown("""
        1.  **Entrada de Áudio/Texto**: O profissional fala ou digita o caso.
        2.  **MedASR**: AI transcreve termos médicos complexos.
        3.  **MedGemma 1.5**: LLM médico extrai sintomas e histórico.
        4.  **Risk Guardrail**: Aplica regras do Protocolo de Manchester.
        5.  **Output**: Gera classificação e prontuário.
        """)
