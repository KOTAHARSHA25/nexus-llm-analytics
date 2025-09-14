"use client";
import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "./ui/table";
import { BarChart3, FileText, TrendingUp, AlertCircle } from "lucide-react";

interface ResultsDisplayProps {
  status: string;
  message?: string;
  results?: any;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ status, message, results }) => {
  if (status === "loading") {
    return (
      <Card className="mt-4">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Analysis Results
          </CardTitle>
          <div className="text-muted-foreground">Processing your query and generating insights...</div>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center py-12 space-y-4">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent"></div>
            <div className="text-center space-y-2">
              <p className="font-medium">Analyzing your data...</p>
              <p className="text-sm text-muted-foreground">This may take a few moments depending on file size</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!results) {
    return (
      <Card className="mt-4">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Analysis Results
          </CardTitle>
          <div className="text-muted-foreground">Your analysis results will appear here</div>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center py-12 space-y-4 text-center">
            <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center">
              <FileText className="h-8 w-8 text-muted-foreground" />
            </div>
            <div className="space-y-2">
              <p className="font-medium">Ready for Analysis</p>
              <p className="text-sm text-muted-foreground max-w-sm">
                Upload your files and ask a question to get started with AI-powered insights
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Example: results.data.summary, results.data.table, results.data.chart
  return (
    <div className="space-y-6 mt-4">
      {/* Summary Card */}
      {results.data?.summary && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Analysis Summary
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 rounded-full bg-accent mt-2 flex-shrink-0"></div>
                <p className="text-sm leading-relaxed">{results.data.summary}</p>
              </div>
              <Badge variant="secondary" className="gap-1">
                <AlertCircle className="h-3 w-3" />
                High Confidence
              </Badge>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Data Table */}
      {results.data?.table && (
        <Card>
          <CardHeader>
            <CardTitle>Key Metrics</CardTitle>
            <div className="text-muted-foreground">Overview of important data points from your analysis</div>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Metric</TableHead>
                  <TableHead>Value</TableHead>
                  <TableHead>Change</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {results.data.table.map((row: any, index: number) => (
                  <TableRow key={index}>
                    <TableCell className="font-medium">{row.metric}</TableCell>
                    <TableCell>{row.value}</TableCell>
                    <TableCell>
                      <Badge variant={row.change?.startsWith("+") ? "default" : "secondary"} className="text-xs">
                        {row.change}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}

      {/* Chart Visualization (placeholder) */}
      {results.data?.chart && (
        <Card>
          <CardHeader>
            <CardTitle>Performance Trend</CardTitle>
            <div className="text-muted-foreground">Visual representation of your data over time</div>
          </CardHeader>
          <CardContent>
            <div className="h-[300px] w-full flex items-center justify-center text-muted-foreground">
              {/* Integrate chart library here, e.g., recharts or plotly */}
              <span>Chart visualization coming soon...</span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Fallback for raw results */}
      {!results.data?.summary && !results.data?.table && !results.data?.chart && (
        <Card className="p-4 overflow-x-auto text-sm">
          <pre className="bg-transparent p-0 m-0">
            {JSON.stringify(results, null, 2)}
          </pre>
        </Card>
      )}
    </div>
  );
};

export default ResultsDisplay;
