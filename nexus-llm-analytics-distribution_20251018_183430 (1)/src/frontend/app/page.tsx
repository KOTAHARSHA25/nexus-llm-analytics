"use client";

import { useState, useEffect } from "react";
import { Header } from "@/components/header";
import { FileUpload } from "@/components/file-upload";
import { QueryInput } from "@/components/query-input";
import { ResultsDisplay } from "@/components/results-display";
import { AnalyticsSidebar } from "@/components/analytics-sidebar";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import ModelSettings from "@/components/model-settings";
import SetupWizard from "@/components/setup-wizard";
import { 
  Download, Menu, X, Sparkles, Zap, Database, FileText, BarChart3, Brain, 
  TrendingUp, DollarSign, Activity, Cpu, Upload, MessageSquare, Settings,
  Play, Square, RefreshCw, CheckCircle, AlertCircle, Info, Clock
} from "lucide-react";
import { FileInfo, PluginInfo } from "@/hooks/useDashboardState";
import { apiUrl, getEndpoint } from "@/lib/config";

export default function AnalyticsDashboard() {
  const [uploadedFiles, setUploadedFiles] = useState<FileInfo[]>([]);
  const [query, setQuery] = useState("");
  const [textInput, setTextInput] = useState("");
  const [inputMode, setInputMode] = useState<"file" | "text">("file");
  const [results, setResults] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [hasResults, setHasResults] = useState(false);
  const [queryHistory, setQueryHistory] = useState<string[]>([]);
  const [errorMsg, setErrorMsg] = useState<any>(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [showSetupWizard, setShowSetupWizard] = useState(false);
  const [selectedPlugin, setSelectedPlugin] = useState<string>("Auto-Select Agent");
  const [isFirstTime, setIsFirstTime] = useState(false);
  const [currentAnalysisId, setCurrentAnalysisId] = useState<string | undefined>(undefined);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [retryCount, setRetryCount] = useState(0);

  // Check if this is a first-time user and load query history
  useEffect(() => {
    checkFirstTimeUser();
    loadQueryHistory();
  }, []);

  const checkFirstTimeUser = async () => {
    try {
      const response = await fetch(getEndpoint('modelsPreferences'));
      const data = await response.json();
      if (data.is_first_time) {
        setIsFirstTime(true);
        setShowSetupWizard(true);
      }
    } catch (error) {
      console.error('Failed to check first-time user status:', error);
    }
  };

  const loadQueryHistory = async () => {
    try {
      const response = await fetch(getEndpoint('history'));
      if (response.ok) {
        const data = await response.json();
        // Extract just the query strings and reverse to show most recent first
        const queries = data.history.map((item: any) => item.query).reverse();
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

  // Enhanced query handling with progress tracking and text support
  const handleQuery = async (queryText: string) => {
    setQuery(queryText);
    setIsLoading(true);
    setErrorMsg(null);
    setAnalysisProgress(0);
    setRetryCount(0);
    setStatusMsg("Initializing analysis...");
    setQueryHistory((prev) => [queryText, ...prev.slice(0, 9)]);
    setResults(null);
    setHasResults(false);
    setCurrentAnalysisId(undefined);
    
    // Progress simulation
    const progressInterval = setInterval(() => {
      setAnalysisProgress(prev => {
        if (prev < 85) return prev + Math.random() * 15;
        return prev;
      });
    }, 500);
    
    try {
      // Generate a session ID for this query
      const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      // Set a temporary analysis ID immediately so stop button appears
      const tempAnalysisId = `temp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      setCurrentAnalysisId(tempAnalysisId);
      
      setStatusMsg("🤖 AI agents are analyzing your query...");
      setAnalysisProgress(20);
      
      const requestBody: any = {
        query: queryText,
        session_id: sessionId,
        max_retries: 2,
        ...(selectedPlugin && { preferred_plugin: selectedPlugin })
      };

      // Handle different input modes
      if (inputMode === "file" && uploadedFiles[0]) {
        requestBody.filename = uploadedFiles[0].name;
      } else if (inputMode === "text" && textInput.trim()) {
        requestBody.text_data = textInput.trim();
      }

      setStatusMsg("🔍 Plugin system routing your query...");
      setAnalysisProgress(40);
      
      const res = await fetch(getEndpoint('analyze'), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });
      
      setStatusMsg("⚡ Processing with specialized agents...");
      setAnalysisProgress(70);
      
      const data = await res.json();
      
      // Update with the real analysis ID for cancellation
      if (data.analysis_id) {
        setCurrentAnalysisId(data.analysis_id);
      }

      // Track retry attempts
      if (data.retry_attempt) {
        setRetryCount(data.retry_attempt);
      }
      
      if (!res.ok || data.error) {
        // Enhanced error handling with structured error information
        const errorInfo = {
          message: data.error || "Analysis failed",
          details: data.details || null,
          suggestions: data.suggestions || [],
          status: data.status || "error",
          retryPossible: data.retry_possible !== false
        };
        
        setErrorMsg(errorInfo);
        setIsLoading(false);
        setStatusMsg(null);
        setCurrentAnalysisId(undefined);
        clearInterval(progressInterval);
        setAnalysisProgress(0);
        return;
      }
      
      setAnalysisProgress(100);
      setResults(data);
      setHasResults(true);
      setStatusMsg("✅ Analysis complete!");
      
      // Save successful query to history
      try {
        const historyItem = {
          query: queryText,
          results_summary: data.summary || `Analysis completed successfully`,
          files_used: inputMode === "file" ? uploadedFiles.map(f => f.name) : []
        };
        
        await fetch(getEndpoint('historyAdd'), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(historyItem),
        });
      } catch (historyError) {
        console.warn("Failed to save query to history:", historyError);
        // Don't fail the main operation if history saving fails
      }
      
    } catch (e) {
      const networkError = {
        message: "❌ Network error - please check your connection",
        details: "Unable to connect to the analysis server. The server may be down or there may be a network connectivity issue.",
        suggestions: [
          "Check if the backend server is running on port 8000",
          "Verify your internet connection",
          "Try refreshing the page"
        ],
        status: "network_error",
        retryPossible: true
      };
      
      setErrorMsg(networkError);
      setStatusMsg(null);
      clearInterval(progressInterval);
      setAnalysisProgress(0);
    } finally {
      setIsLoading(false);
      setCurrentAnalysisId(undefined);
      clearInterval(progressInterval);
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
      
      // If it's a temporary ID (analysis hasn't started on backend yet), just stop locally
      if (currentAnalysisId.startsWith('temp_')) {
        setStatusMsg("Analysis cancelled successfully");
        setIsLoading(false);
        setCurrentAnalysisId(undefined);
        setErrorMsg("Analysis was cancelled by user");
        setTimeout(() => setStatusMsg(null), 2000);
        return;
      }
      
      // Otherwise, try to cancel on the backend
      const res = await fetch(apiUrl(`analyze/cancel/${currentAnalysisId}`), {
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
    if (inputMode === "text") {
      setTextInput(historicalQuery);
    }
  };

  const handleClearHistory = async () => {
    try {
      // Clear backend history
      await fetch("http://127.0.0.1:8000/history/clear", {
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
          className={`fixed inset-y-0 left-0 z-40 w-80 transform transition-transform duration-300 ease-in-out md:relative md:translate-x-0 ${
            sidebarOpen ? "translate-x-0" : "-translate-x-full"
          }`}
        >
          <AnalyticsSidebar
            queryHistory={queryHistory}
            selectedPlugin={selectedPlugin}
            onPluginSelect={setSelectedPlugin}
            onHistoryClick={handleHistoryClick}
            onClearHistory={handleClearHistory}
            onOpenSettings={() => setShowSettings(true)}
          />
        </div>

        {/* Overlay for mobile */}
        {sidebarOpen && (
          <div className="fixed inset-0 z-30 bg-black/50 md:hidden" onClick={() => setSidebarOpen(false)} />
        )}

        {/* Main content */}
        <main className="flex-1 p-4 md:p-6 max-w-full">
          <div className="max-w-7xl mx-auto space-y-6">
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
            </div>

            {/* Main Interface Grid */}
            <div className="grid gap-6 lg:grid-cols-12">
              {/* Left Column - Input & Plugins */}
              <div className="lg:col-span-5 space-y-6">
                {/* Input Method Selection */}
                <Card className="glass-card border-white/30">
                  <CardHeader className="pb-4">
                    <CardTitle className="text-white flex items-center gap-2">
                      <MessageSquare className="h-5 w-5" />
                      Data Input
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <Tabs value={inputMode} onValueChange={(v: any) => setInputMode(v)}>
                      <TabsList className="grid w-full grid-cols-2 bg-white/10">
                        <TabsTrigger value="file" className="data-[state=active]:bg-white/20">
                          <Upload className="h-4 w-4 mr-2" />
                          File Upload
                        </TabsTrigger>
                        <TabsTrigger value="text" className="data-[state=active]:bg-white/20">
                          <FileText className="h-4 w-4 mr-2" />
                          Text Input
                        </TabsTrigger>
                      </TabsList>
                      
                      <TabsContent value="file" className="space-y-4">
                        <FileUpload onFileUpload={handleFileUpload} uploadedFiles={uploadedFiles} />
                      </TabsContent>
                      
                      <TabsContent value="text" className="space-y-4">
                        <div className="space-y-2">
                          <label className="text-sm font-medium text-white/90">
                            Paste your data or text for analysis:
                          </label>
                          <Textarea
                            placeholder="Paste CSV data, JSON, or any text content you want to analyze..."
                            value={textInput}
                            onChange={(e) => setTextInput(e.target.value)}
                            className="min-h-32 bg-white/10 border-white/30 text-white placeholder:text-white/50 resize-none"
                          />
                          <p className="text-xs text-white/60">
                            Tip: You can paste CSV data directly or any text content for analysis
                          </p>
                        </div>
                      </TabsContent>
                    </Tabs>
                  </CardContent>
                </Card>

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
                      disabled={inputMode === "file" ? uploadedFiles.length === 0 : !textInput.trim()}
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
                        
                        {typeof errorMsg === 'object' && errorMsg.details && (
                          <div className="text-xs text-red-200/80 ml-6">
                            {errorMsg.details}
                          </div>
                        )}
                        
                        {typeof errorMsg === 'object' && errorMsg.suggestions?.length > 0 && (
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
                        
                        {typeof errorMsg === 'object' && errorMsg.retryPossible && (
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
                      filename={inputMode === "file" ? uploadedFiles[0]?.name : "Text Input"} 
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
                        setTextInput("");
                        setUploadedFiles([]);
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
