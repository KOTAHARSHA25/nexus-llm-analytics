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
}

const exampleQueries = [
  "Show me a summary of the first 10 rows",
  "What are the statistical patterns in this data?",
  "Create a bar chart showing the distribution",
  "Find all records where revenue is greater than 1000",
  "Show me the correlation between age and income",
  "What are the top 5 categories by count?",
  "Generate a trend analysis over time",
  "Identify any outliers in the dataset",
];

export function QueryInput({ onQuery, isLoading, disabled }: QueryInputProps) {
  const [query, setQuery] = useState("");

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
