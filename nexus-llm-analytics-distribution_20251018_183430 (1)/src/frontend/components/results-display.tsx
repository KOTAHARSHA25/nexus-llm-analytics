"use client";

import { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { getEndpoint } from "@/lib/config";
import { ChartViewer } from "@/components/chart-viewer";
import { 
  BarChart3, 
  FileText, 
  Code, 
  TrendingUp, 
  Database,
  Eye,
  BarChart,
  ChevronDown,
  ChevronRight,
  Download,
  Settings,
  RefreshCw,
  Bot,
  Lightbulb,
  CheckCircle,
  AlertCircle,
  Activity
} from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";

// Helper function to format analysis results with proper structure
const formatAnalysisResult = (result: string) => {
  if (!result) return <p>No result available</p>;

  // Split by numbered sections or bullet points
  const sections = result.split(/(?=\d+\.\s)|(?=\n[•\-\*]\s)/);
  
  return (
    <div className="space-y-4">
      {sections.map((section, index) => {
        if (!section.trim()) return null;
        
        const lines = section.trim().split('\n');
        const isNumberedSection = /^\d+\.\s/.test(section.trim());
        const isBulletPoint = /^[•\-\*]\s/.test(section.trim());
        
        if (isNumberedSection || isBulletPoint) {
          return (
            <div key={index} className="border-l-2 border-purple-400/30 pl-4 py-2">
              <div className="space-y-2">
                {lines.map((line, lineIndex) => {
                  if (!line.trim()) return null;
                  
                  // Check if it's a code block
                  if (line.includes('```') || line.includes('import ') || line.includes('pd.')) {
                    return (
                      <pre key={lineIndex} className="bg-gray-800/50 rounded p-2 text-sm overflow-x-auto text-green-400">
                        <code>{line.trim()}</code>
                      </pre>
                    );
                  }
                  
                  // Check if it's a header line (starts with number or bullet)
                  if (lineIndex === 0 && (isNumberedSection || isBulletPoint)) {
                    return (
                      <h4 key={lineIndex} className="font-semibold text-purple-300">
                        {line.trim()}
                      </h4>
                    );
                  }
                  
                  return (
                    <p key={lineIndex} className="text-sm text-gray-300">
                      {line.trim()}
                    </p>
                  );
                })}
              </div>
            </div>
          );
        }
        
        // Regular paragraph
        return (
          <div key={index} className="space-y-2">
            {lines.map((line, lineIndex) => {
              if (!line.trim()) return null;
              
              // Check if it's a code block
              if (line.includes('```') || line.includes('import ') || line.includes('pd.')) {
                return (
                  <pre key={lineIndex} className="bg-gray-800/50 rounded p-2 text-sm overflow-x-auto text-green-400">
                    <code>{line.trim()}</code>
                  </pre>
                );
              }
              
              return (
                <p key={lineIndex} className="text-sm text-gray-300">
                  {line.trim()}
                </p>
              );
            })}
          </div>
        );
      })}
    </div>
  );
};

interface ResultsDisplayProps {
  results: any;
  isLoading: boolean;
  filename?: string;
}

interface CollapsibleSectionProps {
  title: string;
  children: React.ReactNode;
  defaultExpanded?: boolean;
  icon?: React.ReactNode;
  badge?: React.ReactNode;
}

function CollapsibleSection({ 
  title, 
  children, 
  defaultExpanded = false, 
  icon, 
  badge 
}: CollapsibleSectionProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  return (
    <Card className="glass-card hover:glow-card transition-all duration-300">
      <CardHeader 
        className="cursor-pointer hover:bg-gradient-to-r hover:from-purple-500/10 hover:to-blue-500/10 transition-all duration-300"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-purple-300">{icon}</span>
            <CardTitle className="text-base text-white">{title}</CardTitle>
            {badge}
          </div>
          <div className="flex items-center gap-2">
            {isExpanded ? (
              <ChevronDown className="h-4 w-4 text-purple-300" />
            ) : (
              <ChevronRight className="h-4 w-4 text-purple-300" />
            )}
          </div>
        </div>
      </CardHeader>
      {isExpanded && (
        <CardContent>
          {children}
        </CardContent>
      )}
    </Card>
  );
}

export function ResultsDisplay({ results, isLoading, filename }: ResultsDisplayProps) {
  const [chartData, setChartData] = useState<any>(null);
  const [chartLoading, setChartLoading] = useState(false);
  const [showCode, setShowCode] = useState(false);
  const [reviewInsights, setReviewInsights] = useState<any>(null);
  const [reviewLoading, setReviewLoading] = useState(false);
  const [modelSettings, setModelSettings] = useState<any>(null);
  const [activeTab, setActiveTab] = useState("analysis");

  // Load model settings and auto-generate content when results are available
  useEffect(() => {
    if (results && results.success) {
      loadModelSettings();
      if (filename && !chartData) {
        generateVisualization();
      }
      if (!reviewInsights) {
        generateReviewInsights();
      }
    }
  }, [results, filename]);

  const loadModelSettings = async () => {
    try {
      const response = await fetch(getEndpoint("modelsPreferences"));
      const data = await response.json();
      if (data.preferences) {
        setModelSettings(data.preferences);
      }
    } catch (error) {
      console.error("Failed to load model settings:", error);
    }
  };

  const generateVisualization = async () => {
    if (!filename) return;

    setChartLoading(true);
    try {
      const response = await fetch(getEndpoint("visualizeGenerate"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          data_summary: JSON.stringify(results),
          chart_type: "auto",
          filename: filename
        }),
      });

      const data = await response.json();
      setChartData(data);
    } catch (error) {
      console.error("Visualization generation failed:", error);
      setChartData({ error: "Failed to generate visualization" });
    } finally {
      setChartLoading(false);
    }
  };

  const generateReviewInsights = async () => {
    if (!results || !results.success || !modelSettings) return;

    setReviewLoading(true);
    try {
      const response = await fetch(getEndpoint("analyzeReview"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          original_results: results,
          review_model: modelSettings.review_model || "phi3:latest",
          analysis_type: "quality_review"
        }),
      });

      const data = await response.json();
      setReviewInsights(data);
    } catch (error) {
      console.error("Review insights generation failed:", error);
      setReviewInsights({ error: "Failed to generate review insights" });
    } finally {
      setReviewLoading(false);
    }
  };

  const downloadReport = () => {
    const reportData = {
      timestamp: new Date().toISOString(),
      filename: filename,
      query: results.query || "N/A",
      results: results,
      review_insights: reviewInsights,
      chart_data: chartData
    };

    const dataStr = JSON.stringify(reportData, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `analysis_report_${filename || 'results'}_${new Date().toISOString().slice(0,10)}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  if (isLoading) {
    return (
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-white">
            <BarChart3 className="h-5 w-5 text-purple-300" />
            Analysis Results
          </CardTitle>
          <CardDescription className="text-gray-300">
            Processing your query with AI agents...
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center py-12 space-y-4">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-400"></div>
            <p className="font-medium text-white">AI agents are analyzing your data...</p>
            <p className="text-sm text-gray-300">This may take a moment</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!results) {
    return (
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-white">
            <BarChart3 className="h-5 w-5 text-purple-300" />
            Analysis Results
          </CardTitle>
          <CardDescription className="text-gray-300">
            Your AI-powered analysis results will appear here
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center py-12 space-y-4 text-center">
            <FileText className="h-8 w-8 text-purple-300" />
            <p className="font-medium text-white">Ready for Analysis</p>
            <p className="text-sm text-gray-300">
              Upload a file and ask a question to get started
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Handle error results
  if (results.error) {
    return (
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="text-red-400">Analysis Error</CardTitle>
          <CardDescription className="text-gray-300">
            The AI agents encountered an issue
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4">
            <p className="text-sm text-red-400">{results.error}</p>
            {results.suggestion && (
              <p className="text-sm text-gray-300 mt-2">
                Suggestion: {results.suggestion}
              </p>
            )}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header with Actions */}
      <Card className="glass-card">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2 text-white">
                <Activity className="h-5 w-5 text-green-400" />
                Analysis Dashboard
                {results.type && (
                  <Badge className="text-xs bg-gradient-to-r from-green-500 to-blue-500 text-white border-0">
                    {results.type}
                  </Badge>
                )}
              </CardTitle>
              <CardDescription className="text-gray-300">
                Generated by {modelSettings?.primary_model || 'AI'} with multi-agent analysis
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              {results.execution_time && (
                <Badge className="text-xs bg-gradient-to-r from-purple-500 to-blue-500 text-white border-0">
                  {results.execution_time.toFixed(2)}s
                </Badge>
              )}
              <Button
                size="sm"
                onClick={downloadReport}
                className="flex items-center gap-1 bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 text-white border-0 shadow-lg hover:shadow-purple-500/25 transition-all duration-300"
              >
                <Download className="h-4 w-4" />
                Export Report
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Main Content Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4 bg-gradient-to-r from-gray-800/50 to-gray-700/50 border border-gray-600/50">
          <TabsTrigger value="analysis" className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-purple-500 data-[state=active]:to-blue-500 data-[state=active]:text-white text-gray-300 hover:text-white transition-all duration-300">Analysis</TabsTrigger>
          <TabsTrigger value="insights" className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-purple-500 data-[state=active]:to-blue-500 data-[state=active]:text-white text-gray-300 hover:text-white transition-all duration-300">
            {modelSettings?.review_model ? 
              `${modelSettings.review_model.split(':')[0]} Insights` : 
              'Review Insights'
            }
          </TabsTrigger>
          <TabsTrigger value="visualization" className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-purple-500 data-[state=active]:to-blue-500 data-[state=active]:text-white text-gray-300 hover:text-white transition-all duration-300">Charts</TabsTrigger>
          <TabsTrigger value="details" className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-purple-500 data-[state=active]:to-blue-500 data-[state=active]:text-white text-gray-300 hover:text-white transition-all duration-300">Details</TabsTrigger>
        </TabsList>

        <TabsContent value="analysis" className="space-y-4 mt-6">
          {/* Primary Analysis Results */}
          <CollapsibleSection
            title="Analysis Results"
            defaultExpanded={true}
            icon={<TrendingUp className="h-4 w-4" />}
            badge={
              <Badge className="text-xs bg-gradient-to-r from-green-500 to-blue-500 text-white border-0">
                Primary Analysis
              </Badge>
            }
          >
            <div className="prose prose-sm max-w-none">
              <div className="text-gray-100 leading-relaxed space-y-4">
                {typeof results.result === 'string' 
                  ? formatAnalysisResult(results.result)
                  : <pre className="whitespace-pre-wrap bg-gray-800/50 rounded p-3 text-sm text-gray-300">
                      {JSON.stringify(results.result, null, 2)}
                    </pre>}
              </div>
            </div>
          </CollapsibleSection>

          {/* Data Preview */}
          {results.preview && (
            <CollapsibleSection
              title="Data Preview"
              icon={<Eye className="h-4 w-4" />}
              badge={
                <Badge variant="outline" className="text-xs">
                  {results.preview.length} rows
                </Badge>
              }
            >
              <ScrollArea className="h-[300px] w-full border rounded-md">
                <Table className="text-xs">
                  <TableHeader className="sticky top-0 bg-background">
                    <TableRow>
                      {Object.keys(results.preview[0] || {}).map((key) => (
                        <TableHead key={key} className="text-xs whitespace-nowrap">
                          {key}
                        </TableHead>
                      ))}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {results.preview.slice(0, 100).map((row: any, index: number) => (
                      <TableRow key={index}>
                        {Object.values(row).map((value: any, i: number) => (
                          <TableCell key={i} className="text-xs whitespace-nowrap">
                            {String(value)}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </ScrollArea>
            </CollapsibleSection>
          )}

          {/* Generated Code */}
          {(results.code || results.generated_code) && (
            <CollapsibleSection
              title="Generated Code"
              icon={<Code className="h-4 w-4" />}
              badge={
                <Badge variant="outline" className="text-xs">
                  Python
                </Badge>
              }
            >
              <ScrollArea className="h-[300px] w-full">
                <pre className="bg-muted rounded-lg p-3 text-sm">
                  <code>{results.code || results.generated_code}</code>
                </pre>
              </ScrollArea>
            </CollapsibleSection>
          )}
        </TabsContent>

        <TabsContent value="insights" className="space-y-4 mt-6">
          {/* Review Model Insights */}
          <CollapsibleSection
            title={`${modelSettings?.review_model?.split(':')[0] || 'Review'} Model Insights`}
            defaultExpanded={true}
            icon={<Bot className="h-4 w-4" />}
            badge={
              reviewLoading ? (
                <Badge variant="outline" className="text-xs">
                  <RefreshCw className="h-3 w-3 animate-spin mr-1" />
                  Generating...
                </Badge>
              ) : reviewInsights ? (
                <Badge variant="outline" className="text-xs">
                  <CheckCircle className="h-3 w-3 mr-1" />
                  Ready
                </Badge>
              ) : (
                <Badge variant="outline" className="text-xs">
                  <AlertCircle className="h-3 w-3 mr-1" />
                  Pending
                </Badge>
              )
            }
          >
            {reviewLoading ? (
              <div className="flex items-center justify-center py-8">
                <div className="flex items-center gap-2">
                  <RefreshCw className="h-5 w-5 animate-spin" />
                  <span>Generating review insights...</span>
                </div>
              </div>
            ) : reviewInsights ? (
              <div className="space-y-4">
                {reviewInsights.error ? (
                  <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4">
                    <p className="text-sm text-destructive">{reviewInsights.error}</p>
                  </div>
                ) : (
                  <div className="prose prose-sm max-w-none">
                    <div className="text-foreground leading-relaxed space-y-4">
                      {typeof reviewInsights.insights === 'string' 
                        ? formatAnalysisResult(reviewInsights.insights)
                        : <pre className="whitespace-pre-wrap bg-muted/50 rounded p-3 text-sm">
                            {JSON.stringify(reviewInsights.insights, null, 2)}
                          </pre>}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex items-center justify-center py-8">
                <div className="text-center">
                  <Lightbulb className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
                  <p className="text-sm text-muted-foreground">
                    Review insights will be generated automatically
                  </p>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={generateReviewInsights}
                    className="mt-2"
                  >
                    Generate Now
                  </Button>
                </div>
              </div>
            )}
          </CollapsibleSection>

          {/* Quality Metrics */}
          {reviewInsights && reviewInsights.quality_metrics && (
            <CollapsibleSection
              title="Quality Metrics"
              icon={<CheckCircle className="h-4 w-4" />}
            >
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(reviewInsights.quality_metrics).map(([metric, value]: [string, any]) => (
                  <div key={metric} className="text-center p-3 bg-muted/50 rounded-lg">
                    <div className="text-2xl font-bold">{String(value)}</div>
                    <div className="text-xs text-muted-foreground capitalize">
                      {metric.replace('_', ' ')}
                    </div>
                  </div>
                ))}
              </div>
            </CollapsibleSection>
          )}
        </TabsContent>

        <TabsContent value="visualization" className="space-y-4 mt-6">
          {/* Visualization Section */}
          <CollapsibleSection
            title="Data Visualization"
            defaultExpanded={true}
            icon={<BarChart className="h-4 w-4" />}
            badge={
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={generateVisualization}
                  disabled={chartLoading || !filename}
                >
                  {chartLoading ? (
                    <>
                      <RefreshCw className="h-3 w-3 animate-spin mr-1" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <RefreshCw className="h-3 w-3 mr-1" />
                      Regenerate
                    </>
                  )}
                </Button>
              </div>
            }
          >
            <ChartViewer
              chartData={chartData}
              isLoading={chartLoading}
              error={chartData?.error}
            />
          </CollapsibleSection>
        </TabsContent>

        <TabsContent value="details" className="space-y-4 mt-6">
          {/* Data Statistics */}
          {(results.describe || results.value_counts || results.filtered_count !== undefined) && (
            <CollapsibleSection
              title="Statistical Summary"
              defaultExpanded={true}
              icon={<Database className="h-4 w-4" />}
            >
              {results.describe && (
                <div className="space-y-4">
                  <h4 className="text-sm font-medium">Descriptive Statistics</h4>
                  <ScrollArea className="h-[300px] w-full border rounded-md">
                    <Table className="text-xs">
                      <TableHeader className="sticky top-0 bg-background">
                        <TableRow>
                          <TableHead className="text-xs">Statistic</TableHead>
                          {Object.keys(Object.values(results.describe)[0] || {}).map((col) => (
                            <TableHead key={col} className="text-xs whitespace-nowrap">{col}</TableHead>
                          ))}
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {Object.entries(results.describe).map(([stat, values]: [string, any]) => (
                          <TableRow key={stat}>
                            <TableCell className="font-medium text-xs">{stat}</TableCell>
                            {Object.values(values).map((value: any, i: number) => (
                              <TableCell key={i} className="text-xs">
                                {typeof value === 'number' ? value.toFixed(2) : String(value)}
                              </TableCell>
                            ))}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </ScrollArea>
                </div>
              )}

              {results.value_counts && (
                <div className="space-y-4">
                  <h4 className="text-sm font-medium">Value Distribution</h4>
                  <ScrollArea className="h-[200px] w-full border rounded-md">
                    <Table className="text-xs">
                      <TableHeader className="sticky top-0 bg-background">
                        <TableRow>
                          <TableHead className="text-xs">Value</TableHead>
                          <TableHead className="text-xs">Count</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {Object.entries(results.value_counts).map(([value, count]) => (
                          <TableRow key={value}>
                            <TableCell className="text-xs">{value}</TableCell>
                            <TableCell className="text-xs">{String(count)}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </ScrollArea>
                </div>
              )}

              {results.filtered_count !== undefined && (
                <div className="space-y-2">
                  <h4 className="text-sm font-medium">Filter Results</h4>
                  <Badge variant="outline">
                    {results.filtered_count} rows match the filter
                  </Badge>
                </div>
              )}
            </CollapsibleSection>
          )}

          {/* Technical Details */}
          <CollapsibleSection
            title="Technical Details"
            icon={<Settings className="h-4 w-4" />}
          >
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h4 className="text-sm font-medium">Processing Information</h4>
                  <div className="text-xs text-muted-foreground space-y-1">
                    {results.execution_time && (
                      <div>Execution Time: {results.execution_time.toFixed(2)}s</div>
                    )}
                    {filename && (
                      <div>Source File: {filename}</div>
                    )}
                    {results.agent && (
                      <div>Agent: {results.agent}</div>
                    )}
                    {modelSettings && (
                      <div>Primary Model: {modelSettings.primary_model}</div>
                    )}
                  </div>
                </div>
                
                <div>
                  <h4 className="text-sm font-medium">Analysis Metadata</h4>
                  <div className="text-xs text-muted-foreground space-y-1">
                    <div>Generated: {new Date().toLocaleString()}</div>
                    <div>Type: {results.type || 'Standard Analysis'}</div>
                    {results.confidence && (
                      <div>Confidence: {(results.confidence * 100).toFixed(1)}%</div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </CollapsibleSection>
        </TabsContent>
      </Tabs>
    </div>
  );
}
