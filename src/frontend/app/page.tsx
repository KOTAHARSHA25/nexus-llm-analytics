"use client";

import { useState, useEffect } from "react";
import { Header } from "@/components/header";
import { FileUpload } from "@/components/file-upload";
import { QueryInput } from "@/components/query-input";
import { ResultsDisplay } from "@/components/results-display";
import { Sidebar } from "@/components/sidebar";
import { Button } from "@/components/ui/button";
import ModelSettings from "@/components/model-settings";
import SetupWizard from "@/components/setup-wizard";
import { Download, Menu, X, Sparkles, Zap } from "lucide-react";

interface FileInfo {
  name: string;
  type: string;
  columns?: string[];
}

export default function AnalyticsDashboard() {
  const [uploadedFiles, setUploadedFiles] = useState<FileInfo[]>([]);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [hasResults, setHasResults] = useState(false);
  const [queryHistory, setQueryHistory] = useState<string[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [showSetupWizard, setShowSetupWizard] = useState(false);
  const [isFirstTime, setIsFirstTime] = useState(false);

  // Check if this is a first-time user
  useEffect(() => {
    checkFirstTimeUser();
  }, []);

  const checkFirstTimeUser = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/models/preferences');
      const data = await response.json();
      if (data.is_first_time) {
        setIsFirstTime(true);
        setShowSetupWizard(true);
      }
    } catch (error) {
      console.error('Failed to check first-time user status:', error);
    }
  };

  const handleSetupComplete = () => {
    setShowSetupWizard(false);
    setIsFirstTime(false);
  };

  const handleFileUpload = (files: FileInfo[]) => {
    setUploadedFiles(files);
  };

  // Enhanced query handling with natural language processing
  const handleQuery = async (queryText: string) => {
    setQuery(queryText);
    setIsLoading(true);
    setErrorMsg(null);
    setStatusMsg("Processing your natural language query...");
    setQueryHistory((prev) => [queryText, ...prev.slice(0, 4)]);
    setResults(null);
    setHasResults(false);
    try {
      // The backend now handles natural language queries automatically
      setStatusMsg("AI agents are analyzing your query...");
      const res = await fetch("http://127.0.0.1:8000/analyze/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: queryText,
          filename: uploadedFiles[0]?.name,
        }),
      });
      setStatusMsg("Waiting for backend response...");
      const data = await res.json();
      if (!res.ok || data.error) {
        setErrorMsg(data.error || "Analysis failed");
        setIsLoading(false);
        setStatusMsg(null);
        return;
      }
      setResults(data);
      setHasResults(true);
      setStatusMsg("Analysis complete.");
    } catch (e) {
      setErrorMsg("Network error");
      setStatusMsg(null);
    } finally {
      setIsLoading(false);
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
      const genRes = await fetch("http://127.0.0.1:8000/generate-report/", {
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
      const dlRes = await fetch("http://127.0.0.1:8000/download-report/");
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

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
      <Header />
      <div className="flex">
        {/* Mobile sidebar toggle */}
        <Button
          variant="ghost"
          size="icon"
          className="fixed top-4 left-4 z-50 md:hidden backdrop-blur-sm bg-background/80"
          onClick={() => setSidebarOpen(!sidebarOpen)}
        >
          {sidebarOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </Button>

        {/* Sidebar */}
        <div
          className={`fixed inset-y-0 left-0 z-40 w-64 transform transition-transform duration-300 ease-in-out md:relative md:translate-x-0 ${
            sidebarOpen ? "translate-x-0" : "-translate-x-full"
          }`}
        >
          <Sidebar queryHistory={queryHistory} onSettingsClick={() => setShowSettings(true)} />
        </div>

        {/* Overlay for mobile */}
        {sidebarOpen && (
          <div className="fixed inset-0 z-30 bg-black/50 md:hidden" onClick={() => setSidebarOpen(false)} />
        )}

        {/* Main content */}
        <main className="flex-1 p-6 md:p-8 max-w-7xl mx-auto">
          <div className="space-y-8">
            {/* Enhanced welcome section */}
            <div className="text-center space-y-6">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-accent/10 text-foreground border border-accent/20 text-sm font-medium">
                <Sparkles className="h-4 w-4 text-accent" />
                AI-Powered Analytics Platform
              </div>
              <h1 className="text-4xl md:text-5xl font-bold text-balance bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
                Welcome to Nexus LLM Analytics
              </h1>
              <p className="text-xl text-muted-foreground text-pretty max-w-2xl mx-auto leading-relaxed">
                Upload your data files and ask questions in natural language. Get instant insights with interactive
                visualizations powered by advanced AI.
              </p>
              <div className="flex items-center justify-center gap-8 text-sm text-muted-foreground">
                <div className="flex items-center gap-2">
                  <Zap className="h-4 w-4 text-accent" />
                  <span>Lightning Fast</span>
                </div>
                <div className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-accent" />
                  <span>AI-Powered</span>
                </div>
                <div className="flex items-center gap-2">
                  <Download className="h-4 w-4 text-accent" />
                  <span>Export Ready</span>
                </div>
              </div>
            </div>

            {/* Main dashboard grid */}
            <div className="grid gap-8 lg:grid-cols-2">
              {/* File Upload */}
              <div className="space-y-6">
                <FileUpload onFileUpload={handleFileUpload} uploadedFiles={uploadedFiles} />

                {/* Query Input */}
                <QueryInput 
                  onQuery={handleQuery} 
                  isLoading={isLoading} 
                  disabled={uploadedFiles.length === 0}
                  uploadedFiles={uploadedFiles}
                />
                {errorMsg && <div className="text-red-600 text-sm">{errorMsg}</div>}
                {statusMsg && <div className="text-blue-600 text-sm">{statusMsg}</div>}
              </div>

              {/* Results Display */}
              <div className="space-y-6">
                <ResultsDisplay 
                  results={results} 
                  isLoading={isLoading} 
                  filename={uploadedFiles[0]?.name} 
                />

                {/* Download Report Button (only if backend supports) */}
                {hasResults && (
                  <div className="flex justify-center">
                    <Button
                      onClick={handleDownloadReport}
                      className="gap-2 bg-gradient-to-r from-accent to-accent/80 hover:from-accent/90 hover:to-accent/70 shadow-lg hover:shadow-xl transition-all duration-200"
                      size="lg"
                      data-download-btn
                      disabled={isDownloading}
                    >
                      <Download className="h-5 w-5" />
                      {isDownloading ? "Downloading..." : "Download Report"}
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
