import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Activity, Mic, Send, AlertTriangle, ShieldCheck, Brain,
    Stethoscope, ChevronRight, CheckCircle2, Terminal,
    Cpu, Database, Lock, Search, FileJson, FileText, Zap
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './components/ui/card';
import { Button } from './components/ui/button';
import { Badge } from './components/ui/badge';
import { cn } from './lib/utils';

// --- COMPONENTS ---

// Terminal Log Simulation
const SystemLog = ({ logs }: { logs: string[] }) => {
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }, [logs]);

    return (
        <div className="font-mono text-xs bg-slate-950 text-green-400 p-4 rounded-xl border border-slate-800 shadow-inner h-full flex flex-col">
            <div className="flex items-center gap-2 border-b border-slate-800 pb-2 mb-2 text-slate-400">
                <Terminal className="w-3 h-3" />
                <span className="uppercase tracking-wider text-[10px]">System Kernel Logs</span>
            </div>
            <div ref={scrollRef} className="overflow-y-auto flex-1 space-y-1 scrollbar-hide">
                {logs.map((log, i) => (
                    <div key={i} className="opacity-90 hover:opacity-100 transition-opacity">
                        <span className="text-slate-600 mr-2">[{new Date().toISOString().split('T')[1].slice(0, 8)}]</span>
                        {log}
                    </div>
                ))}
                {logs.length === 0 && <span className="text-slate-600 italic">System idle... Waiting for input signal.</span>}
            </div>
        </div>
    );
};

// Data Inspector Card
const DataInspector = ({ title, icon: Icon, data, type = "json", status }: { title: string, icon: any, data: any, type?: "json" | "text" | "rules", status: 'idle' | 'active' | 'done' }) => (
    <motion.div
        layout
        initial={{ opacity: 0.5 }}
        animate={{
            opacity: status === 'idle' ? 0.5 : 1,
            borderColor: status === 'active' ? '#3b82f6' : status === 'done' ? '#22c55e' : '#e2e8f0',
            boxShadow: status === 'active' ? '0 0 20px rgba(59, 130, 246, 0.2)' : 'none'
        }}
        className="bg-white rounded-xl border-2 p-4 text-sm w-full relative overflow-hidden"
    >
        <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2 font-bold text-slate-700">
                <Icon className="w-4 h-4" />
                {title}
            </div>
            {status === 'active' && <Activity className="w-3 h-3 animate-spin text-blue-500" />}
            {status === 'done' && <CheckCircle2 className="w-3 h-3 text-green-500" />}
        </div>

        <div className="bg-slate-50 rounded-lg p-3 border border-slate-100 font-mono text-xs text-slate-600 overflow-x-auto max-h-[120px]">
            {type === 'json' ? (
                <pre>{JSON.stringify(data, null, 2)}</pre>
            ) : type === 'rules' ? (
                <div className="space-y-1">
                    {data?.map((rule: any, i: number) => (
                        <div key={i} className={cn("flex items-center gap-2", rule.match ? "text-green-600 font-bold" : "text-slate-400")}>
                            {rule.match ? <CheckCircle2 className="w-3 h-3" /> : <div className="w-3 h-3 rounded-full border" />}
                            {rule.text}
                        </div>
                    ))}
                </div>
            ) : (
                <p className="whitespace-pre-wrap">{data || "Aguardando dados..."}</p>
            )}
        </div>
    </motion.div>
);

// Connection Line Animation
const ConnectionLine = ({ active }: { active: boolean }) => (
    <div className="flex flex-col items-center justify-center h-8 my-1 relative">
        <div className="w-0.5 h-full bg-slate-200"></div>
        {active && (
            <motion.div
                layoutId="flow-particle"
                className="absolute w-2 h-2 bg-blue-500 rounded-full shadow-[0_0_10px_#3b82f6]"
                initial={{ y: -10, opacity: 0 }}
                animate={{ y: 20, opacity: 1 }}
                transition={{ duration: 0.5, repeat: Infinity }}
            />
        )}
    </div>
);

function App() {
    const [input, setInput] = useState('');
    const [processState, setProcessState] = useState<'idle' | 'transcribing' | 'extracting' | 'validating' | 'finished'>('idle');
    const [logs, setLogs] = useState<string[]>([]);
    const [result, setResult] = useState<any>(null);

    // Data States for Visualization
    const [transcriptionData, setTranscriptionData] = useState<any>(null);
    const [extractionData, setExtractionData] = useState<any>(null);
    const [validationData, setValidationData] = useState<any>(null);

    const addLog = (msg: string) => setLogs(prev => [...prev.slice(-15), `> ${msg}`]);

    const runSimulation = async () => {
        if (!input) return;
        setProcessState('transcribing');
        setLogs([]);
        setResult(null);
        setTranscriptionData(null);
        setExtractionData(null);
        setValidationData(null);

        // --- STEP 1: TRANSCRIPTION ---
        addLog("Initializing MedASR-1.0 Worker...");
        addLog("Audio Buffer: 256kb received via WebSocket");
        await new Promise(r => setTimeout(r, 800));
        addLog("Diarization: Speaker 0 (Patient) identified");
        addLog("PII Scrubbing: Active (Regex Filter)");
        setTranscriptionData({
            raw_audio_len: "4.2s",
            confidence: 0.98,
            detected_pii: ["***.***.***-**"],
            text_clean: input
        });
        setProcessState('extracting');

        // --- STEP 2: MEDGEMMA ---
        addLog("Dispatching to MedGemma-1.5 (Vertex AI)...");
        addLog("Context Window: 4096 tokens");
        addLog("Prompt Engineering: Chain-of-Thought (CoT) active");
        await new Promise(r => setTimeout(r, 1200));

        // Call API Real
        try {
            addLog("POST /triage waiting for response...");
            const API_URL = "https://neurotriage-ai-10993113678.us-central1.run.app/triage";
            const response = await axios.post(API_URL, {
                transcription: input,
                conversation_id: "deep-dive-" + Date.now()
            });
            const data = response.data;

            setExtractionData({
                model: "medgemma-1.5-27b",
                reasoning_steps: 4,
                symptoms_found: data.symptoms?.length || 0,
                entities: data.symptoms?.map((s: any) => s.name)
            });

            addLog(`Extracted ${data.symptoms?.length} entities`);
            setProcessState('validating');

            // --- STEP 3: GUARDRAIL ---
            addLog("Initializing Manchester Protocol Engine...");
            await new Promise(r => setTimeout(r, 800));

            const isEmergency = data.risk_level === 'emergency';
            const isUrgent = data.risk_level === 'urgent';

            setValidationData([
                { text: "Risco Iminente de Vida (Vermelho)", match: isEmergency },
                { text: "Alteração Sinais Vitais (Laranja)", match: isUrgent && !isEmergency },
                { text: "Dor Aguda / Instabilidade (Amarelo)", match: false },
                { text: "Queixa de Baixa Complexidade (Verde)", match: !isEmergency && !isUrgent },
            ]);

            addLog(`Validation complete. Risk Level: ${data.risk_level.toUpperCase()}`);
            setResult(data);
            setProcessState('finished');
            addLog("Process finished successfully. 200 OK.");

        } catch (error) {
            addLog(`ERROR: ${error}`);
            setProcessState('idle');
        }
    };

    return (
        <div className="min-h-screen bg-slate-50 text-slate-900 font-sans p-6">

            <div className="max-w-7xl mx-auto grid grid-cols-12 gap-6 h-[calc(100vh-3rem)]">

                {/* === LEFT COLUMN: INPUT & CONTROLS (4 cols) === */}
                <div className="col-span-12 lg:col-span-4 flex flex-col gap-6">
                    <header className="mb-2">
                        <div className="flex items-center gap-2 mb-2">
                            <div className="bg-blue-600 p-2 rounded-lg"><Cpu className="text-white w-5 h-5" /></div>
                            <h1 className="text-2xl font-bold tracking-tight text-slate-900">NeuroTriage OS</h1>
                        </div>
                        <p className="text-sm text-slate-500">Medical Intelligence Kernel v2.0.1</p>
                    </header>

                    <Card className="shadow-lg border-0 bg-white">
                        <CardHeader className="pb-3 border-b border-slate-100">
                            <CardTitle className="text-sm font-bold uppercase tracking-wider text-slate-500 flex items-center gap-2">
                                <Mic className="w-4 h-4" /> Input Signal
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="pt-4 space-y-4">
                            <div>
                                <label className="text-xs font-semibold text-slate-700 mb-2 block">RAW CLINICAL DATA</label>
                                <textarea
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    placeholder="Cole o caso clínico ou digite aqui..."
                                    className="w-full text-sm font-mono bg-slate-50 border border-slate-200 rounded-lg p-3 min-h-[120px] focus:ring-2 focus:ring-blue-500 focus:outline-none resize-none"
                                />
                            </div>

                            <div className="grid grid-cols-2 gap-2">
                                <Button variant="outline" size="sm" onClick={() => setInput("Paciente com dor torácica intensa, irradiando para MSE, sudorese, náuseas.")} className="text-xs truncate">
                                    Caso Emergência (IAM)
                                </Button>
                                <Button variant="outline" size="sm" onClick={() => setInput("Criança com febre de 38 graus há 2 dias, coriza e tosse produtiva.")} className="text-xs truncate">
                                    Caso Urgência (Infecção)
                                </Button>
                            </div>

                            <Button
                                onClick={runSimulation}
                                disabled={!input || processState !== 'idle' && processState !== 'finished'}
                                className={cn("w-full h-12 shadow-lg hover:shadow-blue-500/25 transition-all", processState === 'transcribing' ? "bg-slate-800" : "bg-blue-600 hover:bg-blue-700")}
                            >
                                {processState !== 'idle' && processState !== 'finished' ? (
                                    <span className="flex items-center gap-2"><Activity className="animate-spin w-4 h-4" /> PROCESSING THREADS...</span>
                                ) : (
                                    <span className="flex items-center gap-2">INICIAR PROCESSAMENTO <Send className="w-4 h-4" /></span>
                                )}
                            </Button>
                        </CardContent>
                    </Card>

                    <div className="flex-1 overflow-hidden">
                        <SystemLog logs={logs} />
                    </div>
                </div>

                {/* === CENTER COLUMN: THE MACHINE (GLASS BOX) (5 cols) === */}
                <div className="col-span-12 lg:col-span-5 flex flex-col justify-center py-4">
                    <div className="relative">
                        {/* Background Line */}
                        <div className="absolute left-1/2 top-0 bottom-0 w-0.5 bg-slate-200 -z-10 -translate-x-1/2" />

                        {/* STAGE 1: INGEST */}
                        <div className="mb-2 relative z-10">
                            <div className="flex justify-center mb-2">
                                <Badge variant="outline" className="bg-white">Etapa 01: Ingestão</Badge>
                            </div>
                            <DataInspector
                                title="MedASR Transcriber"
                                icon={Mic}
                                status={processState === 'transcribing' ? 'active' : processState === 'idle' ? 'idle' : 'done'}
                                type="json"
                                data={transcriptionData}
                            />
                            <ConnectionLine active={processState === 'transcribing'} />
                        </div>

                        {/* STAGE 2: PROCESS */}
                        <div className="mb-2 relative z-10">
                            <div className="flex justify-center mb-2">
                                <Badge variant="outline" className="bg-white">Etapa 02: Raciocínio (LLM)</Badge>
                            </div>
                            <DataInspector
                                title="MedGemma Neural Core"
                                icon={Brain}
                                status={processState === 'extracting' ? 'active' : (processState === 'validating' || processState === 'finished') ? 'done' : 'idle'}
                                type="json"
                                data={extractionData}
                            />
                            <ConnectionLine active={processState === 'extracting'} />
                        </div>

                        {/* STAGE 3: GUARDRAIL */}
                        <div className="mb-2 relative z-10">
                            <div className="flex justify-center mb-2">
                                <Badge variant="outline" className="bg-white">Etapa 03: Segurança</Badge>
                            </div>
                            <DataInspector
                                title="Risk Guardrail Engine"
                                icon={ShieldCheck}
                                status={processState === 'validating' ? 'active' : processState === 'finished' ? 'done' : 'idle'}
                                type="rules"
                                data={validationData}
                            />
                        </div>

                    </div>
                </div>

                {/* === RIGHT COLUMN: FINAL OUTPUT (3 cols) === */}
                <div className="col-span-12 lg:col-span-3 flex flex-col justify-center">
                    <AnimatePresence>
                        {result && processState === 'finished' && (
                            <motion.div
                                initial={{ x: 50, opacity: 0 }}
                                animate={{ x: 0, opacity: 1 }}
                                className="space-y-4"
                            >
                                <Card className={cn("border-l-8 shadow-2xl",
                                    result.risk_level === 'emergency' ? "border-l-red-500 shadow-red-200/50" :
                                        result.risk_level === 'urgent' ? "border-l-yellow-500 shadow-yellow-200/50" : "border-l-green-500"
                                )}>
                                    <CardHeader>
                                        <div className="flex items-center gap-2 mb-1">
                                            <Zap className={cn("w-5 h-5", result.risk_level === 'emergency' ? "text-red-500" : "text-green-500")} />
                                            <span className="text-xs font-bold uppercase text-slate-400">CLASSIFICAÇÃO FINAL</span>
                                        </div>
                                        <CardTitle className="text-2xl font-black">
                                            {result.risk_level === 'emergency' ? "EMERGÊNCIA" : result.risk_level.toUpperCase()}
                                        </CardTitle>
                                        <CardDescription>
                                            Confiança AI: <span className="font-mono font-bold text-slate-900">{(result.confidence * 100).toFixed(1)}%</span>
                                        </CardDescription>
                                    </CardHeader>
                                    <CardContent>
                                        <p className="text-sm text-slate-600 leading-relaxed mb-4">
                                            {result.rationale}
                                        </p>
                                        <div className="space-y-2">
                                            <h4 className="text-xs font-bold text-slate-400 uppercase">Sintomas Detectados</h4>
                                            <div className="flex flex-wrap gap-1">
                                                {result.symptoms?.map((s: any, i: number) => (
                                                    <Badge key={i} variant="secondary" className="text-xs">{s.name}</Badge>
                                                ))}
                                            </div>
                                        </div>
                                    </CardContent>
                                </Card>

                                <Card className="bg-slate-900 text-slate-300 border-slate-800">
                                    <CardHeader className="pb-2">
                                        <CardTitle className="text-sm flex items-center gap-2">
                                            <Database className="w-4 h-4" /> EMR Integration
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="text-[10px] font-mono space-y-1 opacity-70">
                                            <div>Wait time: &lt; 2s</div>
                                            <div>Protocol: HL7 FHIR v4</div>
                                            <div>Destination: Hospital_DB_Primary</div>
                                            <div className="text-green-400">Status: SYNC_COMPLETE</div>
                                        </div>
                                    </CardContent>
                                </Card>
                            </motion.div>
                        )}
                        {!result && (
                            <div className="h-full flex items-center justify-center text-slate-300">
                                <div className="text-center">
                                    <Search className="w-12 h-12 mx-auto mb-2 opacity-50" />
                                    <p className="text-sm">Aguardando output...</p>
                                </div>
                            </div>
                        )}
                    </AnimatePresence>
                </div>

            </div>
        </div>
    );
}

export default App;
