import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Activity, Mic, Search, ChevronRight, Zap, CheckCircle2,
    ShieldCheck, Terminal, Brain, Database, Github, Linkedin, Send, AlertTriangle
} from 'lucide-react';
import { cn } from './lib/utils';
import { Badge } from './components/ui/badge';

// --- COMPONENTS ---

const GlassCard = ({ children, className }: { children: React.ReactNode, className?: string }) => (
    <div className={cn("glass-card p-6", className)}>
        {children}
    </div>
);

// Terminal Log Simulation
const SystemLog = ({ logs }: { logs: string[] }) => {
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }, [logs]);

    return (
        <div className="font-mono text-xs bg-slate-900 text-green-400 p-4 rounded-xl border border-slate-700 shadow-inner h-full flex flex-col overflow-hidden">
            <div className="flex items-center gap-2 border-b border-slate-700 pb-2 mb-2 text-slate-400">
                <Terminal className="w-3 h-3" />
                <span className="uppercase tracking-wider text-[10px]">System Kernel Logs</span>
            </div>
            <div ref={scrollRef} className="overflow-y-auto flex-1 space-y-1 scrollbar-hide">
                {logs.map((log, i) => (
                    <div key={i} className="opacity-90 hover:opacity-100 transition-opacity whitespace-nowrap">
                        <span className="text-slate-500 mr-2">[{new Date().toISOString().split('T')[1].slice(0, 8)}]</span>
                        {log}
                    </div>
                ))}
                {logs.length === 0 && <span className="text-slate-600 italic">System idle... Waiting for input signal.</span>}
            </div>
        </div>
    );
};

// Glass Inspector for Pipeline Stages
const GlassInspector = ({ title, icon: Icon, data, active, done, step }: { title: string, icon: any, data: any, active: boolean, done: boolean, step: string }) => (
    <motion.div
        animate={{ opacity: active || done ? 1 : 0.6, scale: active ? 1.02 : 1 }}
        className={cn("glass-panel p-4 border transition-all duration-500 w-full relative",
            active ? "border-purple-400 shadow-[0_0_15px_rgba(168,85,247,0.2)] bg-white/70" :
                done ? "border-green-400/50 bg-white/50" : "border-white/20 bg-white/30"
        )}
    >
        {/* Step Label */}
        <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-white px-3 py-0.5 rounded-full text-[10px] font-bold text-slate-400 border border-slate-100 uppercase tracking-widest shadow-sm z-10">
            {step}
        </div>

        <div className="flex items-center justify-between mb-3 mt-1">
            <div className="flex items-center gap-2 font-bold text-slate-700">
                <div className={cn("p-1.5 rounded-lg", active ? "bg-purple-100 text-purple-600" : "bg-slate-100 text-slate-500")}>
                    <Icon className="w-4 h-4" />
                </div>
                {title}
            </div>
            {active && <Activity className="w-4 h-4 animate-spin text-purple-600" />}
            {done && <CheckCircle2 className="w-4 h-4 text-green-500" />}
        </div>

        <div className="bg-slate-50/80 rounded-lg p-2 font-mono text-[10px] text-slate-600 overflow-x-auto max-h-[120px] border border-white/50 scrollbar-hide">
            {data ? (
                <pre>{JSON.stringify(data, null, 2)}</pre>
            ) : (
                <span className="text-slate-400 italic">Waiting...</span>
            )}
        </div>
    </motion.div>
);

function App() {
    const [input, setInput] = useState('');
    const [logs, setLogs] = useState<string[]>([]);
    const [processState, setProcessState] = useState<'idle' | 'transcribing' | 'extracting' | 'validating' | 'finished'>('idle');
    const [result, setResult] = useState<any>(null);

    // Data States
    const [transcriptionData, setTranscriptionData] = useState<any>(null);
    const [extractionData, setExtractionData] = useState<any>(null);
    const [validationData, setValidationData] = useState<any>(null);

    const addLog = (msg: string) => setLogs(prev => [...prev.slice(-20), `> ${msg}`]);

    const runSimulation = async () => {
        if (!input) return;
        setProcessState('transcribing');
        setResult(null); setTranscriptionData(null); setExtractionData(null); setValidationData(null); setLogs([]);

        addLog("Initializing MedASR-1.0 Worker...");
        addLog("Audio Buffer: 256kb received via WebSocket");

        // Step 1: Transcribe (Mock)
        await new Promise(r => setTimeout(r, 1000));
        addLog("Diarization: Speaker 0 (Patient) identified");
        setTranscriptionData({ raw_audio_len: "4.2s", confidence: 0.98, detected_pii: ["***.***.***-**"] });
        setProcessState('extracting');

        // Step 2: Extract (Mock or Real)
        addLog("Dispatching to MedGemma-1.5 (Vertex AI)...");
        addLog("Context Window: 4096 tokens");
        await new Promise(r => setTimeout(r, 1500));

        try {
            const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8080/triage";
            const response = await axios.post(API_URL, {
                transcription: input,
                conversation_id: "hourglass-v2-" + Date.now()
            });
            const data = response.data;

            // Mocking intermediate data for visual feedback based on real response
            setExtractionData({
                model: "medgemma-1.5-27b",
                reasoning_steps: 4,
                symptoms_found: data.symptoms?.length,
                entities: data.symptoms?.map((s: any) => s.name)
            });
            addLog(`Extracted ${data.symptoms?.length} entities`);

            setProcessState('validating');
            addLog("Initializing Manchester Protocol Engine...");
            await new Promise(r => setTimeout(r, 1000));

            const isEmergency = data.risk_level === 'emergency';
            setValidationData([
                { text: isEmergency ? "Risco Iminente de Vida (Vermelho)" : "Baixa Complexidade (Verde)", status: true },
                { text: "Alteração Sinais Vitais (Laranja)", status: false },
                { text: "Dor Aguda / Instabilidade (Amarelo)", status: false }
            ]);
            addLog(`Validation complete. Risk Level: ${data.risk_level.toUpperCase()}`);

            setResult(data);
            setProcessState('finished');
            addLog("Process finished successfully. 200 OK.");
        } catch (error) {
            console.error(error);
            addLog(`ERROR: ${error}`);
            setProcessState('idle');
        }
    };

    return (

        <div className="flex min-h-screen overflow-y-auto lg:overflow-hidden p-4 lg:p-6 gap-6 relative font-sans text-slate-800 bg-[#eef2f6]">
            {/* Background elements */}
            <div className="absolute top-0 left-0 w-full h-full bg-soft-gradient -z-20" />
            <div className="absolute inset-0 bg-mesh-gradient opacity-30 -z-10 blur-3xl pointer-events-none" />

            {/* HEADER */}
            <header className="absolute top-4 left-6 flex items-center gap-3 z-50">
                <div className="bg-blue-600 p-2 rounded-lg shadow-lg"><Activity className="text-white w-5 h-5" /></div>
                <div>
                    <h1 className="text-lg lg:text-xl font-bold tracking-tight text-slate-900 leading-none">NeuroTriage OS</h1>
                    <p className="text-[10px] lg:text-xs text-slate-500 font-medium">Medical Intelligence Kernel v2.0.1</p>
                </div>
            </header>

            {/* MAIN GRID */}
            <div className="mt-14 w-full h-auto lg:h-full grid grid-cols-1 lg:grid-cols-12 gap-6 pb-2">

                {/* === COLUMN 1: INPUT CHANNEL (3 cols) === */}
                <div className="col-span-1 lg:col-span-3 flex flex-col gap-4 h-auto lg:h-[calc(100%-2rem)]">

                    {/* Input Card */}
                    <GlassCard className="flex flex-col gap-3 flex-shrink-0">
                        <div className="flex items-center gap-2 text-slate-500 uppercase text-[10px] font-bold tracking-wider mb-1">
                            <Mic className="w-3 h-3" /> Input Signal
                        </div>

                        <div className="bg-white/50 rounded-lg p-3 border border-slate-200/60 transition-all focus-within:ring-2 focus-within:ring-purple-200">
                            <div className="text-[10px] font-bold text-slate-400 mb-1">RAW CLINICAL DATA</div>
                            <textarea
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                placeholder="Narrate clinical case..."
                                className="w-full h-24 bg-transparent resize-none text-sm leading-relaxed focus:outline-none text-slate-700 placeholder:text-slate-400 font-medium"
                            />
                        </div>

                        <div className="flex gap-2 flex-col lg:flex-row">
                            <button onClick={() => setInput("Paciente com dor torácica intensa, irradiando para MSE, sudorese, náuseas.")} className="flex-1 text-[10px] bg-white border border-slate-200 text-slate-600 px-2 py-3 lg:py-2 rounded shadow-sm hover:bg-slate-50 transition-colors font-semibold active:scale-95 touch-manipulation">
                                Caso Emergência (IAM)
                            </button>
                            <button onClick={() => setInput("Criança com febre de 38 graus há 2 dias, coriza e tosse produtiva.")} className="flex-1 text-[10px] bg-white border border-slate-200 text-slate-600 px-2 py-3 lg:py-2 rounded shadow-sm hover:bg-slate-50 transition-colors font-semibold active:scale-95 touch-manipulation">
                                Caso Urgência (Infecção)
                            </button>
                        </div>

                        <button
                            onClick={runSimulation}
                            disabled={processState !== 'idle' && processState !== 'finished'}
                            className="w-full py-3 rounded-lg bg-blue-600 text-white text-xs font-bold hover:bg-blue-700 hover:shadow-lg hover:shadow-blue-500/20 transition-all flex items-center justify-center gap-2 uppercase tracking-wide disabled:opacity-70 disabled:cursor-not-allowed"
                        >
                            {processState === 'idle' || processState === 'finished' ? (
                                <>INICIAR PROCESSAMENTO <Send className="w-3 h-3" /></>
                            ) : (
                                <>PROCESSING... <Activity className="w-3 h-3 animate-spin" /></>
                            )}
                        </button>
                    </GlassCard>

                    {/* Terminal Logs */}
                    <div className="flex-1 min-h-[250px]">
                        <SystemLog logs={logs} />
                    </div>
                </div>

                {/* === COLUMN 2: PIPELINE VISUALIZATION (5 cols) === */}
                <div className="col-span-1 lg:col-span-5 flex flex-col items-center justify-center h-auto lg:h-[calc(100%-2rem)] relative px-0 lg:px-4 py-8 lg:py-0">
                    {/* Connecting Line */}
                    <div className="hidden lg:block absolute left-1/2 top-4 bottom-4 w-0.5 bg-slate-300/50 -z-10 -translate-x-1/2" />

                    <div className="w-full space-y-6">
                        <GlassInspector
                            step="Etapa 01: Ingestão"
                            title="MedASR Transcriber"
                            icon={Mic}
                            data={transcriptionData}
                            active={processState === 'transcribing'}
                            done={!!transcriptionData}
                        />

                        <GlassInspector
                            step="Etapa 02: Raciocinio (LLM)"
                            title="MedGemma Neural Core"
                            icon={Brain}
                            data={extractionData}
                            active={processState === 'extracting'}
                            done={!!extractionData}
                        />

                        <GlassInspector
                            step="Etapa 03: Segurança"
                            title="Risk Guardrail Engine"
                            icon={ShieldCheck}
                            data={validationData}
                            active={processState === 'validating'}
                            done={!!validationData}
                        />
                    </div>
                </div>

                {/* === COLUMN 3: OUTPUT RESULT (4 cols) === */}
                <div className="col-span-1 lg:col-span-4 flex flex-col gap-4 h-auto lg:h-[calc(100%-2rem)] justify-center">
                    <AnimatePresence mode="wait">
                        {result ? (
                            <motion.div
                                initial={{ opacity: 0, x: 20 }}
                                animate={{ opacity: 1, x: 0 }}
                                className="space-y-4 h-full flex flex-col justify-center"
                            >
                                <GlassCard className={cn("border-l-4 shadow-xl flex-1 flex flex-col justify-center max-h-[500px]",
                                    result.risk_level === 'emergency' ? "border-l-red-500 shadow-red-200/40" :
                                        result.risk_level === 'urgent' ? "border-l-yellow-500 shadow-yellow-200/40" : "border-l-green-500"
                                )}>
                                    <div className="flex items-center gap-2 mb-6">
                                        <Zap className={cn("w-5 h-5", result.risk_level === 'emergency' ? "text-red-500" : "text-green-500")} />
                                        <span className="text-xs font-bold uppercase text-slate-400">Classificação Final</span>
                                    </div>

                                    <h2 className="text-3xl lg:text-4xl font-black text-slate-800 mb-2 uppercase tracking-tight">{result.risk_level}</h2>
                                    <div className="text-xs font-mono font-bold text-slate-500 mb-8">Confiança AI: {(result.confidence * 100).toFixed(1)}%</div>

                                    <div className="space-y-6">
                                        <div className="bg-white/40 p-4 rounded-xl border border-white/50">
                                            <div className="text-[10px] font-bold uppercase text-slate-400 mb-2">Raciocínio Clínico</div>
                                            <p className="text-sm text-slate-700 leading-relaxed font-medium">
                                                {result.rationale}
                                            </p>
                                        </div>

                                        <div>
                                            <span className="text-[10px] font-bold uppercase text-slate-400 block mb-3">Sintomas Detectados</span>
                                            <div className="flex flex-wrap gap-2">
                                                {result.symptoms?.map((s: any, i: number) => (
                                                    <Badge key={i} variant="secondary" className="bg-slate-100 text-slate-600 hover:bg-slate-200 transition-colors px-3 py-1">
                                                        {s.name}
                                                    </Badge>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                </GlassCard>

                                <div className="bg-slate-900 text-slate-300 p-6 rounded-2xl shadow-xl relative overflow-hidden flex-shrink-0">
                                    <div className="absolute top-0 right-0 p-4 opacity-5"><Database className="w-24 h-24" /></div>
                                    <div className="relative z-10">
                                        <h3 className="font-bold text-white flex items-center gap-2 mb-4"><Database className="w-4 h-4" /> EMR Integration</h3>
                                        <div className="text-[10px] font-mono space-y-2 opacity-80">
                                            <div className="flex justify-between border-b border-slate-700 pb-1"><span>Wait time:</span> <span>&lt; 2s</span></div>
                                            <div className="flex justify-between border-b border-slate-700 pb-1"><span>Protocol:</span> <span>HL7 FHIR v4</span></div>
                                            <div className="flex justify-between border-b border-slate-700 pb-1"><span>Destination:</span> <span>Hospital_DB_Primary</span></div>
                                            <div className="flex justify-between text-green-400 font-bold pt-1"><span>Status:</span> <span>SYNC_COMPLETE</span></div>
                                        </div>
                                    </div>
                                </div>
                            </motion.div>
                        ) : (
                            <div className="h-full flex flex-col items-center justify-center opacity-40 gap-4">
                                <div className="w-32 h-32 bg-white/50 rounded-full flex items-center justify-center animate-pulse shadow-inner">
                                    <Search className="w-12 h-12 text-slate-400" />
                                </div>
                                <p className="text-sm font-medium text-slate-500">Aguardando processamento...</p>
                            </div>
                        )}
                    </AnimatePresence>
                </div>
            </div>
        </div>
    );
}

export default App;
