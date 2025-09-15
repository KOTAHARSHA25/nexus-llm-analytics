"use client";

import { useState } from "react";
import { Header } from "@/components/header";
import { FileUpload } from "@/components/file-upload";
import { QueryInput } from "@/components/query-input";
import { ResultsDisplay } from "@/components/results-display";
import { Sidebar } from "@/components/sidebar";
import { Button } from "@/components/ui/button";
import { Download, Menu, X, Sparkles, Zap } from "lucide-react";

export default function AnalyticsDashboard() {
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [hasResults, setHasResults] = useState(false);
  const [queryHistory, setQueryHistory] = useState<string[]>([]);

  const handleFileUpload = (files: File[]) => {
    setUploadedFiles(files);
  };

  const handleQuery = async (queryText: string) => {
    setQuery(queryText);
    setIsLoading(true);
    setQueryHistory((prev) => [queryText, ...prev.slice(0, 4)]);

    // Simulate API call with more realistic data
    setTimeout(() => {
      setResults({
        summary:
          "Analysis complete! Found 3 key insights from your data with 94% confidence. The dataset shows strong performance trends with notable growth patterns in Q2.",
        insights: [
          {
            title: "Revenue Growth",
            description: "12% increase in total revenue compared to last quarter",
          },
          {
            title: "Customer Retention",
            description: "94.1% retention rate, up 2.1% from previous period",
          },
          { title: "Performance Metrics", description: "Average score improved to 87.3 points" },
        ],
      });
      setIsLoading(false);
      setHasResults(true);
    }, 2000);
  };

  const handleDownloadReport = () => {
    console.log("Downloading report...");
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
          <Sidebar queryHistory={queryHistory} />
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
                <QueryInput onQuery={handleQuery} isLoading={isLoading} disabled={uploadedFiles.length === 0} />
              </div>

              {/* Results Display */}
              <div className="space-y-6">
                <ResultsDisplay results={results} isLoading={isLoading} />

                {/* Enhanced Download Report Button */}
                {hasResults && (
                  <div className="flex justify-center">
                    <Button
                      onClick={handleDownloadReport}
                      className="gap-2 bg-gradient-to-r from-accent to-accent/80 hover:from-accent/90 hover:to-accent/70 shadow-lg hover:shadow-xl transition-all duration-200"
                      size="lg"
                      data-download-btn
                    >
                      <Download className="h-5 w-5" />
                      Download Report
                    </Button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
