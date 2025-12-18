"use client";

import type React from "react";
import { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { MessageSquare, Send, Lightbulb, X } from "lucide-react";

interface QueryInputProps {
  onQuery: (query: string) => void;
  isLoading: boolean;
  disabled: boolean;
  uploadedFiles?: Array<{
    name: string;
    type: string;
    columns?: string[];
  }>;
  onCancel?: () => void;
  currentAnalysisId?: string;
}

// Generate completely generic suggestions based only on file type
const generateGenericSuggestions = (files: Array<{name: string; type: string; columns?: string[]}> = []) => {
  if (!files.length) {
    return [
      "Analyze this data and show me key insights",
      "Create visualizations for the main patterns",
      "What are the statistical summaries?",
      "Show me the data structure and types"
    ];
  }

  const suggestions: string[] = [];
  const hasCSV = files.some(f => f.name.toLowerCase().includes('.csv'));
  const hasPDF = files.some(f => f.name.toLowerCase().includes('.pdf'));
  const hasJSON = files.some(f => f.name.toLowerCase().includes('.json'));
  const hasTXT = files.some(f => f.name.toLowerCase().includes('.txt'));
  
  // Generic suggestions based only on file types, not content
  if (hasCSV) {
    suggestions.push(
      "Analyze this data and show me key insights",
      "Create visualizations for the main patterns",
      "What are the statistical summaries?",
      "Show me correlations between variables"
    );
  }
  
  if (hasPDF) {
    suggestions.push(
      "Summarize the content of this document",
      "Extract the key information",
      "What are the main topics discussed?"
    );
  }
  
  if (hasJSON) {
    suggestions.push(
      "Show me the structure of this data",
      "Convert to table format for analysis"
    );
  }
  
  if (hasTXT) {
    suggestions.push(
      "Analyze the text content",
      "Extract key themes and insights"
    );
  }
  
  // Always include these generic ones
  suggestions.push("Generate a comprehensive report");
  
  // Remove duplicates and limit to 6 suggestions
  const uniqueSuggestions = Array.from(new Set(suggestions));
  return uniqueSuggestions.slice(0, 6);
};

export function QueryInput({ onQuery, isLoading, disabled, uploadedFiles, onCancel, currentAnalysisId }: QueryInputProps) {
  const [query, setQuery] = useState("");
  
  // Generate suggestions based on uploaded files
  const suggestions = generateGenericSuggestions(uploadedFiles);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !disabled) {
      onQuery(query.trim());
      setQuery("");
    }
  };

  const handleExampleClick = (example: string) => {
    setQuery(example);
  };

  return (
    <div className="space-y-4">
      <form onSubmit={handleSubmit} className="space-y-4">
        <Textarea
          placeholder={
            disabled
              ? "Please add data first to start querying..."
              : "Ask anything about your data in natural language - our AI agents will understand and analyze it for you!"
          }
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={disabled || isLoading}
          className="min-h-[120px] resize-none bg-white/10 border-white/30 text-white placeholder:text-white/50"
        />

        <div className="flex justify-between items-center">
          <p className="text-xs text-white/60">
            {query.length}/1000 characters
          </p>
          <div className="flex gap-2">
            {isLoading && onCancel && (
              <Button
                type="button"
                variant="outline"
                onClick={onCancel}
                className="gap-2 border-red-400/50 text-red-300 hover:bg-red-500/20"
              >
                <X className="h-4 w-4" />
                Stop Analysis
              </Button>
            )}
            <Button
              type="submit"
              disabled={!query.trim() || disabled || isLoading}
              className="gap-2 bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white shadow-lg hover:shadow-xl transition-all duration-200"
            >
              {isLoading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current"></div>
                  Analyzing...
                </>
              ) : (
                <>
                  <Send className="h-4 w-4" />
                  Send Query
                </>
              )}
            </Button>
          </div>
        </div>
      </form>

      {!disabled && suggestions.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Lightbulb className="h-4 w-4 text-yellow-400" />
            <span className="text-sm font-medium text-white/90">Quick Examples:</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {suggestions.slice(0, 4).map((example, index) => (
              <Badge
                key={index}
                variant="outline"
                className="cursor-pointer hover:bg-white/20 bg-white/10 border-white/30 text-white/90 hover:text-white transition-all"
                onClick={() => handleExampleClick(example)}
              >
                {example}
              </Badge>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
