"use client";

import React, { useEffect, useState } from 'react';
import {
    Bot,
    Database,
    BarChart,
    LineChart,
    FileText,
    Search,
    ShieldCheck,
    Brain,
    Calculator,
    Workflow,
} from 'lucide-react';
import { StreamEvent, AgentState, AgentStatus } from '../types';
import { cn } from '@/lib/utils';

interface SwarmHUDProps {
    events: StreamEvent[];
    isProcessing: boolean;
    className?: string;
    results?: any;
}

// Map agent names to icons
const AgentIcon = ({ name, className }: { name: string, className?: string }) => {
    const n = name.toLowerCase();
    if (n.includes('orchestrat')) return <Workflow className={className} />;
    if (n.includes('analyst')) return <BarChart className={className} />;
    if (n.includes('sql')) return <Database className={className} />;
    if (n.includes('visual')) return <LineChart className={className} />;
    if (n.includes('report')) return <FileText className={className} />;
    if (n.includes('rag') || n.includes('search')) return <Search className={className} />;
    if (n.includes('review') || n.includes('security')) return <ShieldCheck className={className} />;
    if (n.includes('ml')) return <Brain className={className} />;
    if (n.includes('stat') || n.includes('math')) return <Calculator className={className} />;
    if (n.includes('time')) return <LineChart className={className} />;
    return <Bot className={className} />;
};

// Status visual config
const statusConfig: Record<AgentStatus | string, {
    glow: string;
    border: string;
    bg: string;
    text: string;
    iconColor: string;
    pulseColor: string;
    beamColor: string;
}> = {
    thinking: {
        glow: 'shadow-[0_0_20px_rgba(99,102,241,0.4)]',
        border: 'border-indigo-400/60',
        bg: 'bg-indigo-500/10',
        text: 'text-indigo-300',
        iconColor: 'text-indigo-400',
        pulseColor: 'bg-indigo-400',
        beamColor: 'from-indigo-500/60 via-indigo-400/30 to-transparent',
    },
    working: {
        glow: 'shadow-[0_0_24px_rgba(168,85,247,0.5)]',
        border: 'border-purple-400/60',
        bg: 'bg-purple-500/10',
        text: 'text-purple-300',
        iconColor: 'text-purple-400',
        pulseColor: 'bg-purple-400',
        beamColor: 'from-purple-500/60 via-purple-400/30 to-transparent',
    },
    delegating: {
        glow: 'shadow-[0_0_20px_rgba(250,204,21,0.4)]',
        border: 'border-yellow-400/60',
        bg: 'bg-yellow-500/10',
        text: 'text-yellow-300',
        iconColor: 'text-yellow-400',
        pulseColor: 'bg-yellow-400',
        beamColor: 'from-yellow-500/60 via-yellow-400/30 to-transparent',
    },
    complete: {
        glow: 'shadow-[0_0_16px_rgba(34,197,94,0.3)]',
        border: 'border-emerald-400/50',
        bg: 'bg-emerald-500/10',
        text: 'text-emerald-300',
        iconColor: 'text-emerald-400',
        pulseColor: 'bg-emerald-400',
        beamColor: 'from-emerald-500/40 via-emerald-400/20 to-transparent',
    },
    error: {
        glow: 'shadow-[0_0_16px_rgba(239,68,68,0.4)]',
        border: 'border-red-400/50',
        bg: 'bg-red-500/10',
        text: 'text-red-300',
        iconColor: 'text-red-400',
        pulseColor: 'bg-red-400',
        beamColor: 'from-red-500/40 via-red-400/20 to-transparent',
    },
    idle: {
        glow: '',
        border: 'border-slate-600/40',
        bg: 'bg-slate-800/30',
        text: 'text-slate-500',
        iconColor: 'text-slate-500',
        pulseColor: 'bg-slate-500',
        beamColor: 'from-slate-600/20 via-slate-600/10 to-transparent',
    },
};

const getConfig = (status: AgentStatus) => statusConfig[status] || statusConfig.idle;

// Connection beam between orchestrator and agent
const ConnectionBeam = ({ status, side }: { status: AgentStatus, side: 'left' | 'right' }) => {
    const config = getConfig(status);
    const isActive = status !== 'idle' && status !== 'complete';

    return (
        <div className={cn(
            "relative flex items-center",
            side === 'left' ? 'flex-row-reverse' : 'flex-row'
        )}>
            {/* Beam line */}
            <div className={cn(
                "h-[2px] w-12 md:w-20 transition-all duration-700",
                side === 'left'
                    ? `bg-gradient-to-l ${config.beamColor}`
                    : `bg-gradient-to-r ${config.beamColor}`,
                isActive ? 'opacity-100' : 'opacity-40'
            )} />
            {/* Traveling dot */}
            {isActive && (
                <div
                    className={cn(
                        "absolute w-1.5 h-1.5 rounded-full",
                        config.pulseColor,
                        side === 'left' ? 'right-0 animate-travel-dot-l' : 'left-0 animate-travel-dot-r'
                    )}
                />
            )}
        </div>
    );
};

// Individual agent node
const AgentNode = ({ agent, isOrchestrator = false }: { agent: AgentState, isOrchestrator?: boolean }) => {
    const config = getConfig(agent.status);
    const isActive = agent.status !== 'idle';
    const isPulsing = agent.status === 'thinking' || agent.status === 'working' || agent.status === 'delegating';
    const size = isOrchestrator ? 'w-14 h-14 md:w-16 md:h-16' : 'w-10 h-10 md:w-12 md:h-12';
    const iconSize = isOrchestrator ? 'w-6 h-6 md:w-7 md:h-7' : 'w-4 h-4 md:w-5 md:h-5';

    return (
        <div className="flex flex-col items-center gap-1.5 md:gap-2 group">
            {/* Status message tooltip */}
            {agent.message && isActive && agent.status !== 'complete' && (
                <div className={cn(
                    "absolute -top-8 md:-top-9 px-2.5 py-1 rounded-md text-[10px] font-medium whitespace-nowrap z-20",
                    "border backdrop-blur-md",
                    config.bg, config.border, config.text,
                    "animate-fade-slide-up"
                )}>
                    {agent.message}
                    <div className={cn(
                        "absolute left-1/2 -translate-x-1/2 -bottom-1 w-2 h-2 rotate-45 border-b border-r",
                        config.bg, config.border
                    )} />
                </div>
            )}

            {/* Node circle */}
            <div className="relative">
                {/* Outer pulse ring */}
                {isPulsing && (
                    <div className={cn(
                        "absolute inset-0 rounded-full animate-ping opacity-20",
                        config.pulseColor,
                        isOrchestrator ? 'scale-110' : ''
                    )} />
                )}
                {/* Secondary glow ring */}
                {isActive && (
                    <div className={cn(
                        "absolute -inset-1 rounded-full opacity-30 blur-sm transition-all duration-500",
                        config.pulseColor
                    )} />
                )}
                {/* Main circle */}
                <div className={cn(
                    "relative flex items-center justify-center rounded-full border-2 transition-all duration-500 backdrop-blur-sm",
                    size,
                    config.border,
                    config.bg,
                    isActive ? config.glow : '',
                    isOrchestrator ? 'ring-1 ring-white/5' : ''
                )}>
                    <AgentIcon name={agent.name} className={cn(iconSize, config.iconColor, "transition-colors duration-500")} />
                </div>
                {/* Status dot */}
                <div className={cn(
                    "absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 rounded-full border border-gray-900 transition-all duration-300",
                    config.pulseColor
                )} />
            </div>

            {/* Label */}
            <div className={cn(
                "px-2 py-0.5 rounded-md text-[10px] md:text-[11px] font-mono font-medium tracking-wide transition-all duration-300",
                "bg-gray-900/70 backdrop-blur-sm border",
                isActive ? `${config.border} ${config.text}` : 'border-gray-800/50 text-gray-500'
            )}>
                {agent.name.replace('Agent', '')}
            </div>
        </div>
    );
};


export function SwarmHUD({ events, isProcessing, className }: SwarmHUDProps) {
    // Initial Agent Configuration
    const initialAgents: AgentState[] = [
        { id: 'orchestrator', name: 'Orchestrator', status: 'idle' },
        { id: 'analyst', name: 'DataAnalyst', status: 'idle' },
        { id: 'visualizer', name: 'Visualizer', status: 'idle' },
    ];

    const [agents, setAgents] = useState<AgentState[]>(initialAgents);

    // Process Stream Events to update Agent Status
    useEffect(() => {
        if (!events || events.length === 0) {
            if (!isProcessing) {
                setAgents(initialAgents);
            }
            return;
        }

        const newAgents = [...agents];
        const ensureAgent = (name: string, defaultStatus: AgentStatus = 'idle') => {
            let agent = newAgents.find(a => a.name === name);
            if (!agent) {
                agent = { id: name.toLowerCase(), name, status: defaultStatus };
                newAgents.push(agent);
            }
            return agent;
        };

        // Reset loop
        newAgents.forEach(a => {
            if (isProcessing) {
                // Keep existing status if processing
            } else {
                a.status = 'complete'; // Mark all complete if done
            }
        });

        // Replay events to build current state
        events.forEach(e => {
            if (e.step === 'routing') {
                const orch = ensureAgent('Orchestrator');
                orch.status = 'thinking';
                orch.message = 'Routing query...';
            }
            else if (e.step === 'agent_start') {
                let agentName = 'Unknown';
                if (e.message?.includes('DataAnalyst')) agentName = 'DataAnalyst';
                else if (e.message?.includes('Visualizer')) agentName = 'Visualizer';
                else if (e.message?.includes('Reporter')) agentName = 'Reporter';
                else if (e.message?.includes('SQL')) agentName = 'SQLAgent';
                else if (e.message?.includes('TimeSeries')) agentName = 'TimeSeriesAgent';
                else if (e.message?.includes('Statistical')) agentName = 'StatisticalAgent';
                else if (e.message?.includes('ML')) agentName = 'MLInsightsAgent';
                else if (e.message?.includes('Financial')) agentName = 'FinancialAgent';
                else if (e.message?.includes('RAG')) agentName = 'RAGAgent';

                if (agentName !== 'Unknown') {
                    const agent = ensureAgent(agentName);
                    agent.status = 'working';
                    agent.message = 'Processing...';

                    // Orchestrator delegated to this agent
                    const orch = ensureAgent('Orchestrator');
                    orch.status = 'delegating';
                    orch.target = agentName;
                    orch.message = `→ ${agentName.replace('Agent', '')}`;
                }
            }
            else if (e.step === 'complete') {
                newAgents.forEach(a => a.status = 'complete');
            }
        });

        // Only update if changed
        if (JSON.stringify(newAgents) !== JSON.stringify(agents)) {
            setAgents(newAgents);
        }

    }, [events, isProcessing]);


    // Find the orchestrator
    const orchestrator = agents.find(a => a.name === 'Orchestrator') || agents[0];
    const leftAgents = agents.filter(a => a.id !== orchestrator?.id).filter((_, i) => i % 2 === 0);
    const rightAgents = agents.filter(a => a.id !== orchestrator?.id).filter((_, i) => i % 2 !== 0);

    return (
        <div className={cn(
            "relative w-full overflow-hidden rounded-2xl border border-white/[0.08]",
            "bg-gradient-to-br from-gray-950/80 via-gray-900/60 to-gray-950/80",
            "backdrop-blur-xl",
            className
        )}>
            {/* Animated CSS dot-grid background */}
            <div
                className="absolute inset-0 opacity-[0.04]"
                style={{
                    backgroundImage: `radial-gradient(circle, rgba(255,255,255,0.8) 1px, transparent 1px)`,
                    backgroundSize: '24px 24px',
                }}
            />

            {/* Subtle gradient glow behind orchestrator */}
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className={cn(
                    "w-40 h-40 rounded-full blur-3xl transition-all duration-1000",
                    orchestrator?.status === 'thinking' ? 'bg-indigo-500/15' :
                        orchestrator?.status === 'delegating' ? 'bg-yellow-500/10' :
                            orchestrator?.status === 'working' ? 'bg-purple-500/10' :
                                orchestrator?.status === 'complete' ? 'bg-emerald-500/10' :
                                    'bg-slate-500/5'
                )} />
            </div>

            {/* Main horizontal layout */}
            <div className="relative z-10 flex items-center justify-center gap-2 md:gap-4 px-4 md:px-8 py-6 md:py-8">
                {/* Left agents */}
                <div className="flex flex-col items-end gap-4 md:gap-5 flex-shrink-0">
                    {leftAgents.map(agent => (
                        <div key={agent.id} className="relative flex items-center gap-0">
                            <AgentNode agent={agent} />
                            <ConnectionBeam status={agent.status} side="left" />
                        </div>
                    ))}
                </div>

                {/* Central Orchestrator */}
                {orchestrator && (
                    <div className="relative flex-shrink-0">
                        <AgentNode agent={orchestrator} isOrchestrator />
                    </div>
                )}

                {/* Right agents */}
                <div className="flex flex-col items-start gap-4 md:gap-5 flex-shrink-0">
                    {rightAgents.map(agent => (
                        <div key={agent.id} className="relative flex items-center gap-0">
                            <ConnectionBeam status={agent.status} side="right" />
                            <AgentNode agent={agent} />
                        </div>
                    ))}
                </div>
            </div>

            {/* Bottom status bar */}
            {isProcessing && (
                <div className="absolute bottom-0 left-0 right-0 h-[2px]">
                    <div className="h-full bg-gradient-to-r from-transparent via-indigo-500/60 to-transparent animate-shimmer" />
                </div>
            )}


        </div>
    );
}
