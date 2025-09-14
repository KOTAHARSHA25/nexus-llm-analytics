"use client";
import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import FileUpload from "../components/file-upload";
import QueryInput from "../components/query-input";
import ResultsDisplay from "../components/results-display";
import { Zap, Bot, Download, Sparkles } from "lucide-react";

const HomePage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [status, setStatus] = useState("idle");
  const [message, setMessage] = useState<string | undefined>(undefined);

  // Simulate file upload
  const handleFileSelect = async (f: File) => {
    setUploading(true);
    setFile(f);
    setFileName(f.name);
    setStatus("loading");
    setMessage("Uploading file...");
    // TODO: Replace with real upload logic
    setTimeout(() => {
      setUploading(false);
      setStatus("success");
      setMessage("File uploaded successfully.");
    }, 1000);
  };

  // Simulate query send
  const handleSend = async () => {
    setLoading(true);
    setStatus("loading");
    setMessage("Processing query...");
    // TODO: Replace with real query logic
    setTimeout(() => {
      setResults({ query, file: fileName, data: "Sample result data." });
      setLoading(false);
      setStatus("success");
      setMessage("Query processed successfully.");
    }, 1200);
  };

  return (
    <div className="container mx-auto max-w-7xl space-y-8">
      {/* Hero */}
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Badge variant="secondary" className="rounded-full px-3 py-1">
            <Sparkles className="h-3 w-3 mr-1" /> AI-Powered Analytics Platform
          </Badge>
        </div>
        <h1 className="text-3xl md:text-5xl font-bold tracking-tight">
          Welcome to Nexus LLM Analytics
        </h1>
        <p className="text-muted-foreground text-base md:text-lg max-w-3xl">
          Upload your data files and ask questions in natural language. Get instant
          insights with interactive visualizations powered by advanced AI.
        </p>
        <div className="flex flex-wrap gap-6 text-sm text-muted-foreground">
          <div className="flex items-center gap-2"><Zap className="h-4 w-4" /> Lightning Fast</div>
          <div className="flex items-center gap-2"><Bot className="h-4 w-4" /> AI-Powered</div>
          <div className="flex items-center gap-2"><Download className="h-4 w-4" /> Export Ready</div>
        </div>
      </div>

      {/* Content grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>File Upload</CardTitle>
            <div className="text-muted-foreground">Upload PDF, CSV, JSON, or TXT files for analysis (Max 10MB per file)</div>
          </CardHeader>
          <CardContent>
            <FileUpload onFileSelect={handleFileSelect} uploading={uploading} fileName={fileName} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Analysis Results</CardTitle>
            <div className="text-muted-foreground">Your analysis results will appear here</div>
          </CardHeader>
          <CardContent>
            <ResultsDisplay status={status as any} message={message} results={results} />
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Ask a Question</CardTitle>
          <div className="text-muted-foreground">Describe what you want to analyze in plain English</div>
        </CardHeader>
        <CardContent>
          <QueryInput value={query} onChange={setQuery} onSend={handleSend} loading={loading} />
        </CardContent>
      </Card>
    </div>
  );
};

export default HomePage;
