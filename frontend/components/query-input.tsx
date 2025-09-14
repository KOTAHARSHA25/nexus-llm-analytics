"use client";
import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Textarea } from "./ui/textarea";
import { MessageSquare, Send, Lightbulb } from "lucide-react";

interface QueryInputProps {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  loading: boolean;
}

const exampleQueries = [
  "What are the key trends in this data?",
  "Show me a summary of the main findings",
  "Create a visualization of the performance metrics",
  "What insights can you extract from this dataset?",
];

const QueryInput: React.FC<QueryInputProps> = ({ value, onChange, onSend, loading }) => {
  const [localValue, setLocalValue] = useState(value);

  React.useEffect(() => {
    setLocalValue(value);
  }, [value]);

  const handleExampleClick = (example: string) => {
    setLocalValue(example);
    onChange(example);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setLocalValue(e.target.value);
    onChange(e.target.value);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (localValue.trim() && !loading) {
      onSend();
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <MessageSquare className="h-5 w-5" />
          Natural Language Query
        </CardTitle>
        <div className="text-sm text-muted-foreground">Ask questions about your data in plain English</div>
      </CardHeader>
      <CardContent className="space-y-4">
        <form onSubmit={handleSubmit} className="space-y-4">
          <Textarea
            placeholder={
              loading
                ? "Please wait for the current query to finish..."
                : "What would you like to know about your data? e.g., 'Show me the top 10 customers by revenue'"
            }
            value={localValue}
            onChange={handleInputChange}
            disabled={loading}
            className="min-h-[100px] resize-none"
          />
          <div className="flex justify-between items-center">
            <p className="text-xs text-muted-foreground">{localValue.length}/500 characters</p>
            <Button type="submit" disabled={!localValue.trim() || loading} className="gap-2">
              {loading ? (
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
        {/* Example queries */}
        {!loading && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Lightbulb className="h-4 w-4 text-accent" />
              <span className="text-sm font-medium">Try these examples:</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {exampleQueries.map((example, index) => (
                <Badge
                  key={index}
                  variant="outline"
                  className="cursor-pointer hover:bg-accent hover:text-accent-foreground transition-colors"
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
};

export default QueryInput;
