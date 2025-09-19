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
import { MessageSquare, Send, Lightbulb } from "lucide-react";

interface QueryInputProps {
  onQuery: (query: string) => void;
  isLoading: boolean;
  disabled: boolean;
  uploadedFiles?: Array<{
    name: string;
    type: string;
    columns?: string[];
  }>;
}

// Generate dynamic suggestions based on file type and content
const generateDynamicSuggestions = (files: Array<{name: string; type: string; columns?: string[]}> = []) => {
  if (!files.length) {
    return [
      "Show me a summary of the first 10 rows",
      "What are the statistical patterns in this data?",
      "Create a visualization showing the distribution",
      "Find outliers in the dataset",
    ];
  }

  const suggestions: string[] = [];
  
  files.forEach(file => {
    const fileName = file.name.toLowerCase();
    const fileType = file.type?.toLowerCase() || fileName.split('.').pop() || '';
    
    // CSV/Structured data suggestions
    if (fileName.includes('.csv') || fileType === 'text/csv') {
      if (fileName.includes('sales') || fileName.includes('revenue')) {
        suggestions.push(
          "Show me total sales by month",
          "What are the top performing products?",
          "Analyze revenue trends over time",
          "Find the highest revenue transactions"
        );
      } else if (fileName.includes('stress') || fileName.includes('health')) {
        suggestions.push(
          "Analyze stress levels by age group",
          "Show correlation between working hours and stress",
          "What factors contribute most to stress?",
          "Compare stress levels across demographics"
        );
      } else if (fileName.includes('customer') || fileName.includes('user')) {
        suggestions.push(
          "Segment customers by behavior",
          "What's the customer age distribution?",
          "Analyze customer lifetime value",
          "Find patterns in customer demographics"
        );
      } else if (fileName.includes('employee') || fileName.includes('hr')) {
        suggestions.push(
          "Analyze employee satisfaction scores",
          "Show salary distribution by department",
          "What's the average tenure by role?",
          "Identify retention patterns"
        );
      } else {
        // Generic CSV suggestions with column awareness
        if (file.columns && file.columns.length > 0) {
          const cols = file.columns.slice(0, 3).join(', ');
          suggestions.push(
            `Show me statistics for ${cols}`,
            `Create a distribution chart for the main columns`,
            `Find correlations between ${file.columns[0]} and other variables`,
            `Analyze patterns in ${file.columns[0] || 'the data'}`
          );
        } else {
          suggestions.push(
            "Show me a summary of all columns",
            "What are the data types and distributions?",
            "Find missing values and data quality issues",
            "Create visualizations for key metrics"
          );
        }
      }
    }
    
    // PDF/Document suggestions
    else if (fileName.includes('.pdf') || fileType === 'application/pdf') {
      if (fileName.includes('report') || fileName.includes('analysis')) {
        suggestions.push(
          "Summarize the key findings from this report",
          "What are the main conclusions?",
          "Extract the most important statistics",
          "What recommendations are made?"
        );
      } else if (fileName.includes('resume') || fileName.includes('cv')) {
        suggestions.push(
          "Summarize the candidate's experience",
          "What are their key skills?",
          "Extract education and work history",
          "What makes this candidate unique?"
        );
      } else if (fileName.includes('contract') || fileName.includes('agreement')) {
        suggestions.push(
          "What are the key terms and conditions?",
          "Summarize the main obligations",
          "What are the important dates?",
          "Extract payment and delivery terms"
        );
      } else {
        suggestions.push(
          "Summarize the main content of this document",
          "What are the key topics discussed?",
          "Extract the most important information",
          "What are the main takeaways?"
        );
      }
    }
    
    // JSON suggestions
    else if (fileName.includes('.json') || fileType === 'application/json') {
      suggestions.push(
        "Show me the structure of this JSON data",
        "What are the main data fields?",
        "Extract and analyze nested objects",
        "Convert to tabular format for analysis"
      );
    }
    
    // Text file suggestions
    else if (fileName.includes('.txt') || fileType === 'text/plain') {
      suggestions.push(
        "Summarize this text document",
        "What are the main themes?",
        "Extract key information and insights",
        "Analyze the content structure"
      );
    }
  });
  
  // Remove duplicates and limit to 8 suggestions
  const uniqueSuggestions = [...new Set(suggestions)];
  return uniqueSuggestions.slice(0, 8);
};

export function QueryInput({ onQuery, isLoading, disabled, uploadedFiles }: QueryInputProps) {
  const [query, setQuery] = useState("");
  
  // Generate suggestions based on uploaded files
  const exampleQueries = generateDynamicSuggestions(uploadedFiles);

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
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <MessageSquare className="h-5 w-5" />
          Natural Language Query
        </CardTitle>
        <CardDescription>
          Ask questions about your data in plain English
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <form onSubmit={handleSubmit} className="space-y-4">
          <Textarea
            placeholder={
              disabled
                ? "Please upload files first to start querying..."
                : "Ask anything about your data in natural language - our AI agents will understand and analyze it for you!"
            }
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={disabled || isLoading}
            className="min-h-[100px] resize-none"
          />

          <div className="flex justify-between items-center">
            <p className="text-xs text-muted-foreground">
              {query.length}/500
            </p>
            <Button
              type="submit"
              disabled={!query.trim() || disabled || isLoading}
              className="gap-2"
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
        </form>

        {!disabled && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Lightbulb className="h-4 w-4 text-primary" />
              <span className="text-sm font-medium">Try these examples:</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {exampleQueries.map((example, index) => (
                <Badge
                  key={index}
                  variant="outline"
                  className="cursor-pointer hover:bg-accent hover:text-accent-foreground"
                  onClick={() => handleExampleClick(example)}
                >
                  {example}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
