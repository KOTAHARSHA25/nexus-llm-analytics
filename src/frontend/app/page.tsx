"use client";

import { useState, useEffect, useRef } from "react";
import { Header } from "@/components/header";
import { FileUpload } from "@/components/file-upload";
import { QueryInput } from "@/components/query-input";
import { ResultsDisplay } from "@/components/results-display";
import { AnalyticsSidebar } from "@/components/analytics-sidebar";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import ModelSettings from "@/components/model-settings";
import SetupWizard from "@/components/setup-wizard";
import {
  Download, Menu, X, Sparkles, BarChart3, Brain,
  RefreshCw, AlertCircle, Info, Wifi, WifiOff
} from "lucide-react";
import { FileInfo } from "@/hooks/useDashboardState";
import { getEndpoint } from "@/lib/config";
import { StreamEvent } from "@/types";
import { SwarmHUD } from "@/components/swarm-hud";

export default function AnalyticsDashboard() {
  const [uploadedFiles, setUploadedFiles] = useState<FileInfo[]>([]);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Record<string, unknown> | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [hasResults, setHasResults] = useState(false);
  const [queryHistory, setQueryHistory] = useState<string[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | { message: string; details?: string; retryPossible?: boolean; status?: string; suggestions?: string[] } | null>(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [showSetupWizard, setShowSetupWizard] = useState(false);
  const [selectedPlugin, setSelectedPlugin] = useState<string>("Auto-Select Agent");
  const [isFirstTime, setIsFirstTime] = useState(false);
  const [currentAnalysisId, setCurrentAnalysisId] = useState<string | undefined>(undefined);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [retryCount, setRetryCount] = useState(0);
  const [streamEvents, setStreamEvents] = useState<StreamEvent[]>([]);

  // Online/Offline mode toggle
  const [nexusMode, setNexusMode] = useState<"offline" | "online">("offline");
  const [modeToast, setModeToast] = useState<string | null>(null);
  const modeToastTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [ollamaStatus, setOllamaStatus] = useState<"running" | "stopped" | "starting" | "stopping" | "unknown">("unknown");

  // AbortController ref for cancelling in-flight streaming requests
  const abortControllerRef = useRef<AbortController | null>(null);

  // Check if this is a first-time user and load query history
  useEffect(() => {
    checkFirstTimeUser();
    loadQueryHistory();
  }, []);

  // Restore mode from localStorage and sync with backend on mount
  useEffect(() => {
    const saved = localStorage.getItem("nexus_mode");

    // Using TS type assertion below to ensure "online" or "offline" is passed if the string is valid
    const syncMode = async () => {
      try {
        if (saved === "online" || saved === "offline") {
          // If we have a saved preference, tell the backend to use it
          setNexusMode(saved as "online" | "offline");
          await fetch(getEndpoint('mode'), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode: saved }),
          });
        } else {
          // Otherwise, fetch whatever the backend is currently using
          const res = await fetch(getEndpoint('mode'));
          if (res.ok) {
            const data = await res.json();
            if (data.mode === "online" || data.mode === "offline") {
              setNexusMode(data.mode);
              localStorage.setItem("nexus_mode", data.mode);
            }
          }
        }
      } catch (e) {
        console.warn("Failed to sync mode with backend on mount:", e);
      }
    };

    syncMode();
  }, []);

  const showModeToast = (msg: string) => {
    setModeToast(msg);
    if (modeToastTimerRef.current) clearTimeout(modeToastTimerRef.current);
    modeToastTimerRef.current = setTimeout(() => setModeToast(null), 3500);
  };

  const handleModeToggle = async () => {
    const newMode = nexusMode === "offline" ? "online" : "offline";
    try {
      const res = await fetch(getEndpoint('mode'), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: newMode }),
      });
      if (!res.ok) throw new Error(`Server responded ${res.status}`);
      setNexusMode(newMode);
      localStorage.setItem("nexus_mode", newMode);
      showModeToast(
        newMode === "online"
          ? "Switched to Online Mode — Using cloud APIs"
          : "Switched to Offline Mode — Using local models"
      );
    } catch (e) {
      console.warn("Mode switch failed:", e);
      // Still update UI so user isn't blocked if backend is temporarily unavailable
      setNexusMode(newMode);
      localStorage.setItem("nexus_mode", newMode);
      showModeToast(
        newMode === "online"
          ? "Switched to Online Mode — Using cloud APIs"
          : "Switched to Offline Mode — Using local models"
      );
    }
  };

  const checkFirstTimeUser = async () => {
    try {
      const response = await fetch(getEndpoint('modelsPreferences'));
      if (!response.ok) return;
      const data = await response.json();
      if (data.is_first_time) {
        setIsFirstTime(true);
        setShowSetupWizard(true);
      }
    } catch {
      // Backend not reachable yet — silently ignore, user can still use the app
    }
  };

  const loadQueryHistory = async () => {
    try {
      const response = await fetch(getEndpoint('history'));
      if (response.ok) {
        const data = await response.json();
        // Extract just the query strings and reverse to show most recent first
        const queries = data.history.map((item: { query: string }) => item.query).reverse();
        setQueryHistory(queries.slice(0, 10)); // Keep only recent 10 for UI
      }
    } catch (error) {
      console.warn('Failed to load query history:', error);
      // Don't show error to user, just use empty history
    }
  };

  const handleSetupComplete = () => {
    setShowSetupWizard(false);
    setIsFirstTime(false);
  };

  const handleFileUpload = (files: FileInfo[]) => {
    setUploadedFiles(files);
  };

  // Enhanced query handling with true Streaming support
  const handleQuery = async (queryText: string) => {
    // Abort any in-flight request before starting a new one (prevents race condition)
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setQuery(queryText);
    setIsLoading(true);
    setErrorMsg(null);
    setAnalysisProgress(0);
    setRetryCount(0);
    setStatusMsg("Initializing analysis...");
    setQueryHistory((prev) => [queryText, ...prev.slice(0, 9)]);
    setResults(null);
    setHasResults(false);
    setStreamEvents([]); // Reset events

    // Set a temporary analysis ID immediately so cancel button works
    const tempId = `temp_${Date.now()}`;
    setCurrentAnalysisId(tempId);

    try {
      // Generate a session ID for this query
      const sessionId = crypto.randomUUID?.() ?? `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      const requestBody: Record<string, unknown> = {
        query: queryText,
        session_id: sessionId,
        max_retries: 2,
        ...(selectedPlugin && { preferred_plugin: selectedPlugin })
      };

      // Use uploaded file (including pasted text saved as file)
      if (uploadedFiles[0]) {
        requestBody.filename = uploadedFiles[0].name;
      }

      setStatusMsg("🚀 Connect to Stream...");

      // Use the Streaming Endpoint
      const response = await fetch(getEndpoint('stream'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
        signal: controller.signal,
      });

      if (!response.ok) {
        // Provide more detailed error information
        let errorDetail = `HTTP ${response.status}: ${response.statusText}`;
        try {
          const errorData = await response.json();
          errorDetail += ` - ${errorData.error || errorData.detail || JSON.stringify(errorData)}`;
        } catch {
          // Couldn't parse error response
        }
        throw new Error(errorDetail);
      }

      // Initialize result state structure
      setResults({
        result: "",
        metadata: {},
        query: queryText,
        status: "processing"
      });
      setHasResults(true);

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) throw new Error("Stream reader not available");

      let buffer = "";
      let accumulatedResult = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        // Split by double newline to get SSE messages
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || ""; // Keep incomplete line in buffer

        for (const line of lines) {
          // Skip empty or undefined lines
          if (!line || !line.trim()) continue;

          if (line.startsWith('data: ')) {
            const jsonStr = line.slice(6);
            if (!jsonStr.trim()) continue;

            try {
              const data = JSON.parse(jsonStr) as StreamEvent;
              setStreamEvents(prev => [...prev, data]); // Store event for HUD

              // Handle different event types
              if (data.step) {
                // Update progress message
                if (data.message) setStatusMsg(data.message);
                if (data.progress) setAnalysisProgress(data.progress);

                // Handle Content Tokens
                if (data.step === 'token' && data.token) {
                  accumulatedResult += data.token;
                  setResults((prev) => ({
                    ...prev,
                    result: accumulatedResult
                  }));
                }

                // Handle Plan event — store execution plan for display
                if (data.step === 'plan' && data.plan) {
                  setResults((prev) => ({
                    ...prev,
                    plan: data.plan
                  }));
                }

                // Handle Completion
                if (data.step === 'complete' && data.result) {
                  // Final update with all metadata (including plan)
                  setResults((prev) => ({
                    ...data.result!, // Assert non-null since checked above
                    plan: prev?.plan || data.result!.plan
                  }));
                  setCurrentAnalysisId(data.result.analysis_id || undefined);
                  setAnalysisProgress(100);
                  setStatusMsg("Analysis complete!");

                  // Save to history
                  try {
                    const historyItem = {
                      query: queryText,
                      results_summary: (data.result.result || "").slice(0, 100) + "...",
                      files_used: uploadedFiles.map(f => f.name)
                    };
                    await fetch(getEndpoint('historyAdd'), {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify(historyItem),
                    });
                  } catch (e) { console.warn("History save failed", e); }
                }

                // Handle Errors
                if (data.step === 'error') {
                  setErrorMsg(data.error || "Streaming error occurred");
                }
              }
            } catch (e) {
              console.warn("Error parsing SSE JSON", e);
            }
          }
        }
      }

    } catch (e: unknown) {
      // Don't show error if the request was intentionally aborted
      if (e instanceof DOMException && e.name === 'AbortError') {
        return;
      }
      console.error("Streaming error:", e);
      const errorMessage = e instanceof Error ? e.message : "Failed to connect to streaming endpoint";
      const isServiceDown = errorMessage.includes("503") || errorMessage.includes("fetch") || errorMessage.includes("Failed to fetch");
      const networkError = {
        message: "Connection Error",
        details: errorMessage,
        retryPossible: true,
        status: isServiceDown ? 'service_unavailable' : 'error',
        suggestions: [
          "Check that the backend server is running (start_backend.bat)",
          "Verify Ollama is running with a model loaded (ollama list)",
          "Try refreshing the page and re-uploading your file",
          ...(isServiceDown ? ["Check if port 8000 is available and not blocked"] : [])
        ]
      };
      setErrorMsg(networkError);
    } finally {
      // Clean up the controller ref
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
      setIsLoading(false);
      // Clean up status message after a delay
      setTimeout(() => {
        setStatusMsg(null);
        setAnalysisProgress(0);
      }, 3000);
    }
  };

  // Cancel analysis handler
  const handleCancelAnalysis = async () => {
    if (!currentAnalysisId) return;

    try {
      setStatusMsg("Cancelling analysis...");

      // Abort the in-flight fetch request immediately
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }

      // If it's a temporary ID (analysis hasn't started on backend yet), just stop locally
      if (currentAnalysisId.startsWith('temp_')) {
        setStatusMsg("Analysis cancelled successfully");
        setIsLoading(false);
        setCurrentAnalysisId(undefined);
        setErrorMsg("Analysis was cancelled by user");
        setTimeout(() => setStatusMsg(null), 2000);
        return;
      }

      // Cancel on the backend using configured endpoint
      const res = await fetch(`${getEndpoint('analyzeCancel')}/${currentAnalysisId}`, {
        method: "POST",
      });

      const data = await res.json();
      if (res.ok) {
        setStatusMsg("Analysis cancelled successfully");
        setIsLoading(false);
        setCurrentAnalysisId(undefined);
        setErrorMsg("Analysis was cancelled by user");
      } else {
        setStatusMsg("Failed to cancel analysis");
      }
    } catch (e) {
      setStatusMsg("Error cancelling analysis");
      console.error("Cancel error:", e);
    } finally {
      setTimeout(() => setStatusMsg(null), 2000);
    }
  };

  // Download report logic
  const handleDownloadReport = async () => {
    if (!results) return;
    setIsDownloading(true);
    setErrorMsg(null);
    setStatusMsg("Generating report in backend...");
    try {
      // 1. Generate report
      const genRes = await fetch(getEndpoint('generateReport'), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ results: [results] }),
      });
      const genData = await genRes.json();
      if (!genRes.ok || genData.error) {
        setErrorMsg(genData.error || "Failed to generate report");
        setIsDownloading(false);
        setStatusMsg(null);
        return;
      }
      setStatusMsg("Downloading report from backend...");
      // 2. Download report
      const dlRes = await fetch(getEndpoint('downloadReport'));
      if (!dlRes.ok) {
        setErrorMsg("Failed to download report");
        setIsDownloading(false);
        setStatusMsg(null);
        return;
      }
      const blob = await dlRes.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "generated_report.pdf";
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
      setStatusMsg("Report downloaded.");
    } catch (e) {
      setErrorMsg("Failed to download report");
      setStatusMsg(null);
    } finally {
      setIsDownloading(false);
      setTimeout(() => setStatusMsg(null), 2000);
    }
  };

  const handleHistoryClick = (historicalQuery: string) => {
    setQuery(historicalQuery);
  };

  const handleClearCache = async () => {
    try {
      setStatusMsg("Clearing analysis cache...");
      const res = await fetch(getEndpoint('clearCache'), {
        method: "POST"
      });
      const data = await res.json();

      if (res.ok) {
        setStatusMsg("Cache cleared successfully");
        // Also clear local results to reflect "fresh" state
        setResults(null);
        setHasResults(false);
      } else {
        setStatusMsg(`Failed to clear cache: ${data.error || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Failed to clear cache:', error);
      setStatusMsg("Error clearing cache");
    } finally {
      setTimeout(() => setStatusMsg(null), 2000);
    }
  };

  const handleClearHistory = async () => {
    try {
      // Clear backend history using configured endpoint
      await fetch(getEndpoint('historyClear'), {
        method: "DELETE"
      });
    } catch (error) {
      console.warn('Failed to clear backend history:', error);
    }

    // Always clear frontend history
    setQueryHistory([]);
  };

  return (
    <div className="min-h-screen gradient-bg dark">
      <Header />

      {/* Mobile menu toggle */}
      <Button
        variant="ghost"
        size="icon"
        className="fixed top-4 left-4 z-50 md:hidden glass-card text-white hover:bg-white/20"
        onClick={() => setSidebarOpen(!sidebarOpen)}
      >
        {sidebarOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
      </Button>

      <div className="flex">
        {/* Sidebar */}
        <div
          className={`fixed inset-y-0 left-0 z-40 w-80 transform transition-transform duration-300 ease-in-out md:relative md:translate-x-0 ${sidebarOpen ? "translate-x-0" : "-translate-x-full"
            }`}
        >
          <AnalyticsSidebar
            queryHistory={queryHistory}
            selectedPlugin={selectedPlugin}
            onPluginSelect={setSelectedPlugin}
            onHistoryClick={handleHistoryClick}
            onClearHistory={handleClearHistory}
            onOpenSettings={() => setShowSettings(true)}
            onClearCache={handleClearCache}
          />
        </div>

        {/* Overlay for mobile */}
        {sidebarOpen && (
          <div className="fixed inset-0 z-30 bg-black/50 md:hidden" onClick={() => setSidebarOpen(false)} />
        )}

        {/* Main content */}
        <main className="flex-1 p-4 md:p-6 max-w-full">
          <div className="max-w-7xl mx-auto space-y-6">

            {/* Mode Toast Notification */}
            {modeToast && (
              <div
                className="fixed top-6 right-6 z-50 flex items-center gap-3 px-5 py-3 rounded-xl
                           glass-card border border-white/30 text-white text-sm shadow-2xl
                           animate-in fade-in slide-in-from-top-3 duration-300"
              >
                {nexusMode === "online"
                  ? <Wifi className="h-4 w-4 text-green-400 flex-shrink-0" />
                  : <WifiOff className="h-4 w-4 text-blue-400 flex-shrink-0" />
                }
                {modeToast}
              </div>
            )}

            {/* Hero Section */}
            <div className="text-center space-y-4 py-8">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card text-white border border-white/30">
                <Sparkles className="h-4 w-4" />
                Advanced Multi-Agent AI Analytics Platform
              </div>
              <h1 className="text-4xl md:text-6xl font-bold text-white leading-tight">
                Nexus LLM Analytics
              </h1>
              <p className="text-xl text-white/80 max-w-3xl mx-auto leading-relaxed">
                Transform your data into insights with our advanced AI multi-agent system.
                Upload files or paste text, ask questions in natural language, and get comprehensive analysis.
              </p>

              {/* Online / Offline Mode Toggle */}
              <div className="inline-flex flex-col items-center gap-2 mt-2">
                <div className="inline-flex items-center gap-3">
                  <span className={`text-sm font-medium transition-colors ${nexusMode === "offline" ? "text-white" : "text-white/40"}`}>
                    Offline
                  </span>
                  <button
                    onClick={handleModeToggle}
                    aria-label={`Switch to ${nexusMode === "offline" ? "online" : "offline"} mode`}
                    className={`relative inline-flex h-7 w-14 items-center rounded-full border-2 transition-all duration-300 focus:outline-none focus-visible:ring-2 focus-visible:ring-white/50
                      ${nexusMode === "online"
                        ? "bg-gradient-to-r from-green-500 to-emerald-500 border-green-400/60"
                        : "bg-white/10 border-white/20 hover:border-white/40"
                      }`}
                  >
                    <span
                      className={`inline-block h-5 w-5 transform rounded-full bg-white shadow-lg transition-all duration-300
                        ${nexusMode === "online" ? "translate-x-[30px]" : "translate-x-[2px]"}`}
                    />
                  </button>
                  <span className={`text-sm font-medium transition-colors ${nexusMode === "online" ? "text-green-400" : "text-white/40"}`}>
                    Online
                  </span>
                  {nexusMode === "online" && (
                    <span className="flex items-center gap-1 text-xs text-green-400/80 bg-green-500/10 border border-green-500/20 rounded-full px-2.5 py-0.5">
                      <Wifi className="h-3 w-3" />
                      Cloud APIs active
                    </span>
                  )}
                </div>

                {/* Ollama process status indicator */}
                {nexusMode === "offline" && ollamaStatus === "running" && (
                  <div className="flex items-center gap-1.5 text-xs">
                    <span className="h-1.5 w-1.5 rounded-full bg-green-400 inline-block" />
                    <span className="text-green-400">Ollama running</span>
                  </div>
                )}
                {nexusMode === "offline" && ollamaStatus === "starting" && (
                  <div className="flex items-center gap-1.5 text-xs">
                    <span className="h-1.5 w-1.5 rounded-full bg-yellow-400 animate-pulse inline-block" />
                    <span className="text-yellow-300/80">Ollama starting\u2026</span>
                  </div>
                )}
                {nexusMode === "offline" && ollamaStatus === "stopped" && (
                  <div className="flex items-center gap-1.5 text-xs">
                    <span className="h-1.5 w-1.5 rounded-full bg-red-400 inline-block" />
                    <span className="text-red-400/80">Ollama not running</span>
                  </div>
                )}
                {nexusMode === "online" && (ollamaStatus === "stopped" || ollamaStatus === "stopping") && (
                  <div className="flex items-center gap-1.5 text-xs">
                    <span className="h-1.5 w-1.5 rounded-full bg-white/20 inline-block" />
                    <span className="text-white/40">Ollama idle</span>
                  </div>
                )}
              </div>
            </div>

            {/* Main Interface Grid */}
            <div className="grid gap-6 lg:grid-cols-12">
              {/* Left Column - Input & Plugins */}
              <div className="lg:col-span-5 space-y-6">
                {/* File/Text Input */}
                <FileUpload onFileUpload={handleFileUpload} uploadedFiles={uploadedFiles} />

                {/* Query Input */}
                <Card className="glass-card border-white/30">
                  <CardHeader className="pb-4">
                    <CardTitle className="text-white flex items-center gap-2">
                      <Brain className="h-5 w-5" />
                      AI Query Interface
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <QueryInput
                      onQuery={handleQuery}
                      isLoading={isLoading}
                      disabled={false}
                      uploadedFiles={uploadedFiles}
                      onCancel={handleCancelAnalysis}
                      currentAnalysisId={currentAnalysisId}
                    />

                    {/* Analysis Progress */}
                    {(isLoading || analysisProgress > 0) && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm text-white/80">
                          <span>Analysis Progress</span>
                          <span>{Math.round(analysisProgress)}%</span>
                        </div>
                        <Progress value={analysisProgress} className="bg-white/20" />
                      </div>
                    )}

                    {/* Status Messages */}
                    {statusMsg && (
                      <div className="flex items-center gap-2 p-3 rounded-lg bg-blue-500/20 border border-blue-500/30">
                        <Info className="h-4 w-4 text-blue-300" />
                        <span className="text-sm text-blue-300">{statusMsg}</span>
                      </div>
                    )}

                    {errorMsg && (
                      <div className="p-4 rounded-lg bg-red-500/20 border border-red-500/30 space-y-3">
                        <div className="flex items-center gap-2">
                          <AlertCircle className="h-4 w-4 text-red-300 flex-shrink-0" />
                          <span className="text-sm font-medium text-red-300">
                            {typeof errorMsg === 'string' ? errorMsg : errorMsg.message}
                          </span>
                        </div>

                        {typeof errorMsg === 'object' && errorMsg !== null && errorMsg.details && (
                          <div className="text-xs text-red-200/80 ml-6">
                            {errorMsg.details}
                          </div>
                        )}

                        {typeof errorMsg === 'object' && errorMsg !== null && Array.isArray(errorMsg.suggestions) && errorMsg.suggestions.length > 0 && (
                          <div className="ml-6 space-y-2">
                            <div className="text-xs font-medium text-red-200">Suggestions:</div>
                            <ul className="text-xs text-red-200/80 space-y-1">
                              {errorMsg.suggestions.map((suggestion: string, index: number) => (
                                <li key={index} className="flex items-start gap-2">
                                  <span className="text-red-300 mt-0.5">•</span>
                                  <span>{suggestion}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {typeof errorMsg === 'object' && errorMsg !== null && errorMsg.retryPossible && (
                          <div className="flex gap-2 ml-6 pt-2">
                            <button
                              onClick={() => setErrorMsg(null)}
                              className="px-3 py-1.5 text-xs bg-red-500/30 hover:bg-red-500/40 text-red-200 rounded-lg border border-red-500/50 transition-colors"
                            >
                              Dismiss
                            </button>
                            {errorMsg.status === 'service_unavailable' && (
                              <button
                                onClick={() => setShowSettings(true)}
                                className="px-3 py-1.5 text-xs bg-blue-500/30 hover:bg-blue-500/40 text-blue-200 rounded-lg border border-blue-500/50 transition-colors"
                              >
                                Open Model Settings
                              </button>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>

              {/* Right Column - Results & Actions */}
              <div className="lg:col-span-7 space-y-6">

                {/* Visual Swarm HUD */}
                {(isLoading || hasResults) && (
                  <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <SwarmHUD events={streamEvents} isProcessing={isLoading} />
                  </div>
                )}

                {/* Results Display */}
                <Card className="glass-card border-white/30 min-h-96">
                  <CardHeader className="pb-4">
                    <CardTitle className="text-white flex items-center gap-2">
                      <BarChart3 className="h-5 w-5" />
                      Analysis Results
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ResultsDisplay
                      results={results}
                      isLoading={isLoading}
                      filename={uploadedFiles[0]?.name || "No file"}
                    />
                  </CardContent>
                </Card>

                {/* Action Buttons */}
                {hasResults && (
                  <div className="flex gap-4 justify-center">
                    <Button
                      onClick={handleDownloadReport}
                      className="gap-2 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white shadow-lg hover:shadow-xl transition-all duration-200"
                      size="lg"
                      disabled={isDownloading}
                    >
                      <Download className="h-5 w-5" />
                      {isDownloading ? "Generating..." : "Download Report"}
                    </Button>

                    <Button
                      onClick={() => {
                        setResults(null);
                        setHasResults(false);
                        setQuery("");
                        setUploadedFiles([]);
                        setStreamEvents([]);
                      }}
                      variant="outline"
                      size="lg"
                      className="gap-2 border-white/30 text-white hover:bg-white/10"
                    >
                      <RefreshCw className="h-5 w-5" />
                      New Analysis
                    </Button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </main>
      </div>

      {/* Model Settings Modal */}
      <ModelSettings
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
      />

      {/* Setup Wizard */}
      <SetupWizard
        isOpen={showSetupWizard}
        onComplete={handleSetupComplete}
      />
    </div>
  );
}
