"use client";

import React, { useEffect, useRef, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Loader2, Download, Maximize2 } from "lucide-react";

interface ChartViewerProps {
  chartData?: any;
  isLoading?: boolean;
  error?: string;
}

declare global {
  interface Window {
    Plotly: any;
  }
}

export function ChartViewer({ chartData, isLoading, error }: ChartViewerProps) {
  const plotRef = useRef<HTMLDivElement>(null);
  const [plotlyLoaded, setPlotlyLoaded] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Load Plotly.js dynamically
  useEffect(() => {
    const loadPlotly = async () => {
      if (typeof window !== "undefined" && !window.Plotly) {
        try {
          // Load Plotly from CDN
          const script = document.createElement("script");
          script.src = "https://cdn.plot.ly/plotly-2.26.0.min.js";
          script.onload = () => {
            setPlotlyLoaded(true);
          };
          document.head.appendChild(script);
        } catch (error) {
          console.error("Failed to load Plotly.js:", error);
        }
      } else if (window.Plotly) {
        setPlotlyLoaded(true);
      }
    };

    loadPlotly();
  }, []);

  // Render chart when data is available and Plotly is loaded
  useEffect(() => {
    if (plotlyLoaded && chartData && chartData.success && plotRef.current) {
      try {
        let figureData;
        
        if (typeof chartData.figure_json === "string") {
          figureData = JSON.parse(chartData.figure_json);
        } else {
          figureData = chartData.figure_json;
        }

        // Configure the plot
        const config = {
          responsive: true,
          displayModeBar: true,
          modeBarButtonsToRemove: ["pan2d", "lasso2d"],
          displaylogo: false,
          toImageButtonOptions: {
            format: "png",
            filename: "nexus_chart",
            height: 600,
            width: 800,
            scale: 2,
          },
        };

        // Create the plot
        window.Plotly.newPlot(
          plotRef.current,
          figureData.data || [],
          figureData.layout || {},
          config
        );

      } catch (err) {
        console.error("Error rendering chart:", err);
      }
    }
  }, [plotlyLoaded, chartData]);

  const handleDownloadChart = () => {
    if (plotRef.current && window.Plotly) {
      window.Plotly.downloadImage(plotRef.current, {
        format: "png",
        width: 1200,
        height: 800,
        filename: "nexus_chart",
      });
    }
  };

  const toggleFullscreen = () => {
    if (plotRef.current) {
      if (!isFullscreen) {
        plotRef.current.requestFullscreen?.();
        setIsFullscreen(true);
      } else {
        document.exitFullscreen?.();
        setIsFullscreen(false);
      }
    }
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Loader2 className="h-5 w-5 animate-spin" />
            Generating Visualization
          </CardTitle>
          <CardDescription>
            Creating your chart using AI-powered analysis...
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-12">
            <div className="text-center space-y-4">
              <div className="animate-pulse bg-muted rounded-lg h-64 w-full"></div>
              <p className="text-sm text-muted-foreground">
                Processing data and generating visualization...
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-destructive">Visualization Error</CardTitle>
          <CardDescription>
            Failed to generate the chart
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4">
            <p className="text-sm text-destructive">{error}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!chartData || !chartData.success) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Data Visualization</CardTitle>
          <CardDescription>
            Your charts will appear here once data is analyzed
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-12">
            <div className="text-center space-y-4">
              <div className="bg-muted rounded-lg h-64 w-full flex items-center justify-center">
                <p className="text-muted-foreground">No visualization available</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="overflow-hidden">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              Data Visualization
              {chartData.chart_type && (
                <Badge variant="secondary" className="text-xs">
                  {chartData.chart_type}
                </Badge>
              )}
            </CardTitle>
            <CardDescription>
              Interactive chart generated from your data analysis
            </CardDescription>
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleDownloadChart}
              disabled={!plotlyLoaded}
            >
              <Download className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={toggleFullscreen}
              disabled={!plotlyLoaded}
            >
              <Maximize2 className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="relative">
          {!plotlyLoaded && (
            <div className="absolute inset-0 flex items-center justify-center bg-background/80">
              <div className="flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="text-sm">Loading chart library...</span>
              </div>
            </div>
          )}
          <div
            ref={plotRef}
            className="w-full"
            style={{ minHeight: "400px" }}
          />
        </div>
        
        {chartData.generated_code && (
          <div className="p-4 border-t bg-muted/30">
            <details className="group">
              <summary className="cursor-pointer text-sm font-medium text-muted-foreground hover:text-foreground">
                View Generated Code
              </summary>
              <pre className="mt-2 p-3 bg-background rounded border text-xs overflow-x-auto">
                <code>{chartData.generated_code}</code>
              </pre>
            </details>
          </div>
        )}
      </CardContent>
    </Card>
  );
}