"use client";

import { useEffect, useRef, useCallback, useMemo, startTransition, Suspense, lazy } from "react";
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
import { 
  Download, Menu, X, Sparkles, Zap, Database, FileText, BarChart3, Brain, 
  TrendingUp, DollarSign, Activity, Cpu, Upload, MessageSquare, Settings,
  Play, Square, RefreshCw, CheckCircle, AlertCircle, Info, Clock
} from "lucide-react";

// Performance-optimized components and hooks
import { useDashboardState, usePerformanceMonitor, FileInfo, PluginInfo } from "../hooks/useDashboardState";
import {
  FileList,
  PluginSelector,
  QueryHistory,
  ProgressIndicator,
} from "../components/OptimizedComponents";

// Lazy load heavy components
const ModelSettings = lazy(() => import("@/components/model-settings"));
const SetupWizard = lazy(() => import("@/components/setup-wizard"));

// Memoized plugin configuration
const PLUGINS: PluginInfo[] = [
  {
    id: "auto-select",
    name: "Auto-Select Agent",
    description: "Automatically chooses the best agent for your task",
    icon: Sparkles,
    color: "#8B5CF6",
    capabilities: ["Intelligent routing", "Multi-agent coordination", "Optimal performance"]
  },
  {
    id: "data-analyst",
    name: "Data Analyst Agent",
    description: "Specialized in data analysis, statistics, and insights",
    icon: BarChart3,
    color: "#10B981",
    capabilities: ["Statistical analysis", "Data visualization", "Pattern recognition", "Trend analysis"]
  },
  {
    id: "business-intelligence",
    name: "Business Intelligence Agent",
    description: "Business metrics, KPIs, and strategic insights",
    icon: TrendingUp,
    color: "#3B82F6",
    capabilities: ["KPI tracking", "Business metrics", "Strategic analysis", "Performance monitoring"]
  },
  {
    id: "financial-analyst",
    name: "Financial Analyst Agent",
    description: "Financial modeling, forecasting, and analysis",
    icon: DollarSign,
    color: "#F59E0B",
    capabilities: ["Financial modeling", "Risk analysis", "Forecasting", "Investment analysis"]
  },
  {
    id: "performance-monitor",
    name: "Performance Monitor Agent",
    description: "System performance, monitoring, and optimization",
    icon: Activity,
    color: "#EF4444",
    capabilities: ["Performance tracking", "System monitoring", "Optimization", "Alerting"]
  },
  {
    id: "ml-engineer",
    name: "ML Engineer Agent",
    description: "Machine learning, model development, and AI insights",
    icon: Brain,
    color: "#8B5CF6",
    capabilities: ["Model development", "Feature engineering", "ML optimization", "AI insights"]
  }
];

// Performance-optimized component with error boundary
export default function AnalyticsDashboard() {
  // Use optimized state management
  const {
    state,
    derivedState,
    dispatch,
    batchUpdate,
    handleFileUpload,
    addFileWithId,
    removeFileById,
    setQuery,
    setTextInput,
    addToHistory,
    startProgressSimulation,
    stopProgressSimulation,
    cleanup,
    apiCallsRef,
    progressIntervalRef
  } = useDashboardState();

  // Performance monitoring (development only)
  const performanceData = usePerformanceMonitor();

  // Refs for DOM elements and stable references
  const fileInputRef = useRef<HTMLInputElement>(null);
  const retryTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Memoized computed values
  const computedState = useMemo(() => ({
    canSubmitQuery: derivedState.canSubmitQuery && !state.isLoading,
    currentInput: state.inputMode === "file" ? state.currentQuery : state.textInput,
    analysisButtonText: state.isLoading ? "Analyzing..." : "Start Analysis",
    progressPercentage: Math.min(state.analysisProgress, 100),
  }), [
    derivedState.canSubmitQuery, 
    state.isLoading, 
    state.inputMode, 
    state.currentQuery, 
    state.textInput, 
    state.analysisProgress
  ]);

  // Optimized file upload handler with batch processing
  const handleFilesUploaded = useCallback(async (files: FileList) => {
    const fileArray = Array.from(files);
    const MAX_FILES = 10; // Limit for performance
    const validFiles = fileArray.slice(0, MAX_FILES);

    try {
      dispatch({ type: 'BATCH_UPDATE', payload: { 
        fileUploadProgress: 0,
        errorMsg: null 
      }});

      const processedFiles = await Promise.all(
        validFiles.map(async (file, index) => {
          // Simulate progress
          const progress = ((index + 1) / validFiles.length) * 100;
          dispatch({ type: 'SET_FILE_UPLOAD_PROGRESS', payload: progress });

          // Process file metadata
          const fileInfo: FileInfo = {
            id: `${file.name}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            name: file.name,
            type: file.name.split('.').pop()?.toLowerCase() || 'unknown',
            uploadedAt: Date.now(),
          };

          // Extract columns for CSV files (performance optimized)
          if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
            try {
              const text = await file.text();
              const firstLine = text.split('\n')[0];
              const columns = firstLine.split(',').map(col => col.trim().replace(/"/g, ''));
              fileInfo.columns = columns.slice(0, 20); // Limit columns for performance
            } catch (error) {
              console.warn(`Failed to extract columns from ${file.name}:`, error);
            }
          }

          return fileInfo;
        })
      );

      // Batch update all files
      startTransition(() => {
        handleFileUpload(processedFiles);
        dispatch({ type: 'SET_FILE_UPLOAD_PROGRESS', payload: 100 });
      });

    } catch (error) {
      dispatch({ type: 'SET_ERROR_MSG', payload: `Failed to upload files: ${error}` });
    }
  }, [handleFileUpload, dispatch]);

  // Optimized query submission with request deduplication
  const handleQuerySubmit = useCallback(async () => {
    const queryToSubmit = computedState.currentInput.trim();
    if (!queryToSubmit || state.isLoading) return;

    // Generate unique request ID for deduplication
    const requestId = `${queryToSubmit}_${Date.now()}`;
    
    // Cancel any existing requests
    apiCallsRef.current.forEach(controller => controller.abort());
    apiCallsRef.current.clear();

    // Create new abort controller
    const abortController = new AbortController();
    apiCallsRef.current.set(requestId, abortController);

    try {
      // Batch state updates for analysis start
      batchUpdate({
        isLoading: true,
        analysisProgress: 0,
        currentAnalysisId: requestId,
        retryCount: 0,
        errorMsg: null,
        statusMsg: "Initializing analysis..."
      });

      // Add to history (debounced and deduplicated)
      addToHistory(queryToSubmit);

      // Start progress simulation
      startProgressSimulation();

      // Prepare request data
      const requestData = {
        query: queryToSubmit,
        mode: state.inputMode,
        files: state.inputMode === "file" ? derivedState.uploadedFilesList : [],
        selectedPlugin: state.selectedPlugin,
        requestId
      };

      // Make API request with timeout and retry logic
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestData),
        signal: abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      
      // Stop progress simulation and update results
      stopProgressSimulation();
      
      batchUpdate({
        isLoading: false,
        analysisProgress: 100,
        results: result,
        hasResults: true,
        statusMsg: "Analysis complete!"
      });

    } catch (error: any) {
      stopProgressSimulation();
      
      if (error.name === 'AbortError') {
        console.log('Request was cancelled');
        return;
      }

      // Implement exponential backoff for retries
      const shouldRetry = state.retryCount < 3 && !error.message.includes('400');
      
      if (shouldRetry) {
        const retryDelay = Math.pow(2, state.retryCount) * 1000; // Exponential backoff
        
        dispatch({ type: 'SET_STATUS_MSG', payload: `Retrying in ${retryDelay/1000}s...` });
        dispatch({ type: 'SET_RETRY_COUNT', payload: state.retryCount + 1 });

        retryTimeoutRef.current = setTimeout(() => {
          handleQuerySubmit();
        }, retryDelay);
        
      } else {
        batchUpdate({
          isLoading: false,
          analysisProgress: 0,
          errorMsg: error.message || "Analysis failed. Please try again.",
          statusMsg: undefined
        });
      }
    } finally {
      // Cleanup request tracking
      apiCallsRef.current.delete(requestId);
    }
  }, [
    computedState.currentInput,
    state.isLoading,
    state.inputMode,
    state.selectedPlugin,
    state.retryCount,
    derivedState.uploadedFilesList,
    batchUpdate,
    addToHistory,
    startProgressSimulation,
    stopProgressSimulation,
    apiCallsRef
  ]);

  // Optimized download handler with progress tracking
  const handleDownloadReport = useCallback(async () => {
    if (!state.results) return;

    try {
      dispatch({ type: 'SET_IS_DOWNLOADING', payload: true });

      const response = await fetch("/api/download-report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ results: state.results }),
      });

      if (!response.ok) throw new Error("Download failed");

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `nexus-report-${Date.now()}.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

    } catch (error) {
      dispatch({ type: 'SET_ERROR_MSG', payload: "Failed to download report" });
    } finally {
      dispatch({ type: 'SET_IS_DOWNLOADING', payload: false });
    }
  }, [state.results, dispatch]);

  // Memoized event handlers to prevent recreation
  const memoizedHandlers = useMemo(() => ({
    toggleSidebar: () => dispatch({ type: 'TOGGLE_SIDEBAR' }),
    setInputMode: (mode: "file" | "text") => dispatch({ type: 'SET_INPUT_MODE', payload: mode }),
    setSelectedPlugin: (plugin: string) => dispatch({ type: 'SET_SELECTED_PLUGIN', payload: plugin }),
    showSettings: () => dispatch({ type: 'SET_SHOW_SETTINGS', payload: true }),
    hideSettings: () => dispatch({ type: 'SET_SHOW_SETTINGS', payload: false }),
    showSetupWizard: () => dispatch({ type: 'SET_SHOW_SETUP_WIZARD', payload: true }),
    hideSetupWizard: () => dispatch({ type: 'SET_SHOW_SETUP_WIZARD', payload: false }),
    clearResults: () => batchUpdate({ results: null, hasResults: false, errorMsg: null }),
    clearHistory: () => dispatch({ type: 'CLEAR_HISTORY' }),
    cancelAnalysis: () => {
      apiCallsRef.current.forEach(controller => controller.abort());
      apiCallsRef.current.clear();
      stopProgressSimulation();
      dispatch({ type: 'RESET_ANALYSIS_STATE' });
    }
  }), [dispatch, batchUpdate, apiCallsRef, stopProgressSimulation]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanup();
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
      }
    };
  }, [cleanup]);

  // Performance logging in development
  useEffect(() => {
    if (process.env.NODE_ENV === 'development') {
      console.log('Performance metrics:', performanceData);
    }
  }, [performanceData]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Optimized Header */}
      <Header />

      {/* Main Content */}
      <div className="flex">
        {/* Optimized Sidebar */}
        <AnalyticsSidebar
          queryHistory={state.queryHistory}
          selectedPlugin={state.selectedPlugin}
          onPluginSelect={memoizedHandlers.setSelectedPlugin}
          onHistoryClick={(query: string) => setQuery(query)}
          onClearHistory={memoizedHandlers.clearHistory}
          onOpenSettings={memoizedHandlers.showSettings}
        />

        {/* Main Dashboard */}
        <main className={`flex-1 transition-all duration-300 ${
          state.sidebarOpen ? 'ml-64' : 'ml-0'
        }`}>
          <div className="p-6 max-w-7xl mx-auto">
            
            {/* Welcome Section */}
            <div className="mb-8">
              <h1 className="text-4xl font-bold text-gray-900 mb-2">
                Welcome to Nexus LLM Analytics
              </h1>
              <p className="text-lg text-gray-600">
                Advanced AI-powered analytics platform with intelligent agent routing
              </p>
            </div>

            {/* Plugin Selection */}
            <Card className="mb-6">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="w-5 h-5" />
                  Select AI Agent
                </CardTitle>
                <CardDescription>
                  Choose the specialized agent for your analysis task
                </CardDescription>
              </CardHeader>
              <CardContent>
                <PluginSelector
                  plugins={PLUGINS}
                  selectedPlugin={state.selectedPlugin}
                  onSelectPlugin={memoizedHandlers.setSelectedPlugin}
                  isLoading={state.isLoading}
                />
              </CardContent>
            </Card>

            {/* Input Mode Selection */}
            <Card className="mb-6">
              <CardHeader>
                <CardTitle>Input Method</CardTitle>
                <CardDescription>
                  Choose how you want to provide data for analysis
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Tabs 
                  value={state.inputMode} 
                  onValueChange={memoizedHandlers.setInputMode as any}
                  className="w-full"
                >
                  <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="file" className="flex items-center gap-2">
                      <Upload className="w-4 h-4" />
                      File Upload
                    </TabsTrigger>
                    <TabsTrigger value="text" className="flex items-center gap-2">
                      <MessageSquare className="w-4 h-4" />
                      Text Input
                    </TabsTrigger>
                  </TabsList>

                  <TabsContent value="file" className="space-y-4">
                    <FileUpload 
                      onFileUpload={handleFileUpload}
                      uploadedFiles={derivedState.uploadedFilesList}
                    />
                    
                    {derivedState.hasUploadedFiles && (
                      <Card>
                        <CardHeader>
                          <CardTitle className="text-lg">
                            Uploaded Files ({state.uploadedFiles.size})
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <FileList
                            files={derivedState.uploadedFilesList}
                            onRemoveFile={removeFileById}
                            isLoading={state.isLoading}
                          />
                        </CardContent>
                      </Card>
                    )}

                    <div className="space-y-4">
                      <QueryInput
                        onQuery={setQuery}
                        isLoading={state.isLoading}
                        disabled={state.isLoading}
                        uploadedFiles={derivedState.uploadedFilesList}
                      />
                    </div>
                  </TabsContent>

                  <TabsContent value="text" className="space-y-4">
                    <div className="space-y-4">
                      <Textarea
                        value={state.textInput}
                        onChange={(e) => setTextInput(e.target.value)}
                        placeholder="Paste your text data here or describe what you'd like to analyze..."
                        className="min-h-[200px] resize-vertical"
                        disabled={state.isLoading}
                      />
                    </div>
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>

            {/* Query History */}
            {state.queryHistory.length > 0 && (
              <Card className="mb-6">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Clock className="w-5 h-5" />
                    Recent Queries
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <QueryHistory
                    history={state.queryHistory}
                    onSelectQuery={(query) => {
                      if (state.inputMode === "file") {
                        setQuery(query);
                      } else {
                        setTextInput(query);
                      }
                    }}
                    onClearHistory={memoizedHandlers.clearHistory}
                  />
                </CardContent>
              </Card>
            )}

            {/* Analysis Controls */}
            <Card className="mb-6">
              <CardContent className="pt-6">
                <div className="flex flex-col sm:flex-row gap-4 items-center justify-between">
                  <div className="flex-1">
                    {state.isLoading && (
                      <ProgressIndicator
                        progress={computedState.progressPercentage}
                        isLoading={state.isLoading}
                        statusMsg={state.statusMsg}
                      />
                    )}
                  </div>
                  
                  <div className="flex gap-2">
                    {state.isLoading ? (
                      <Button
                        onClick={memoizedHandlers.cancelAnalysis}
                        variant="destructive"
                        size="lg"
                        className="flex items-center gap-2"
                      >
                        <Square className="w-4 h-4" />
                        Cancel
                      </Button>
                    ) : (
                      <Button
                        onClick={handleQuerySubmit}
                        disabled={!computedState.canSubmitQuery}
                        size="lg"
                        className="flex items-center gap-2"
                      >
                        <Play className="w-4 h-4" />
                        {computedState.analysisButtonText}
                      </Button>
                    )}
                    
                    {state.hasResults && (
                      <Button
                        onClick={handleDownloadReport}
                        disabled={state.isDownloading}
                        variant="outline"
                        size="lg"
                        className="flex items-center gap-2"
                      >
                        <Download className="w-4 h-4" />
                        {state.isDownloading ? "Downloading..." : "Download Report"}
                      </Button>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Error Display */}
            {state.errorMsg && (
              <Card className="mb-6 border-red-200 bg-red-50">
                <CardContent className="pt-6">
                  <div className="flex items-center gap-2 text-red-800">
                    <AlertCircle className="w-5 h-5" />
                    <span className="font-medium">Error</span>
                  </div>
                  <p className="mt-2 text-red-700">{state.errorMsg}</p>
                  <Button
                    onClick={() => dispatch({ type: 'SET_ERROR_MSG', payload: null })}
                    variant="outline"
                    size="sm"
                    className="mt-3"
                  >
                    Dismiss
                  </Button>
                </CardContent>
              </Card>
            )}

            {/* Results Display */}
            {state.hasResults && state.results && (
              <ResultsDisplay
                results={state.results}
                isLoading={false}
                filename={derivedState.uploadedFilesList[0]?.name}
              />
            )}

          </div>
        </main>
      </div>

      {/* Lazy-loaded Modal Components */}
      <Suspense fallback={<div>Loading...</div>}>
        {state.showSettings && (
          <ModelSettings 
            isOpen={state.showSettings}
            onClose={memoizedHandlers.hideSettings} 
          />
        )}
        
        {state.showSetupWizard && (
          <SetupWizard 
            isOpen={state.showSetupWizard}
            onComplete={memoizedHandlers.hideSetupWizard} 
          />
        )}
      </Suspense>
    </div>
  );
}