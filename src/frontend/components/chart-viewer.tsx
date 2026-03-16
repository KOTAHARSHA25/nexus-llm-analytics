"use client";

import React, { useEffect, useRef, useState } from "react";
import { useTheme } from "next-themes";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Loader2, Download, Maximize2, ZoomIn, ZoomOut, Move } from "lucide-react";

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
  const { theme, systemTheme } = useTheme();

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
    // Backward compatibility: check both success formats
    const isSuccessful = chartData && (chartData.success === true || chartData.status === "success");

    if (plotlyLoaded && isSuccessful && plotRef.current) {
      try {
        let figureData;

        // Support both old and new response formats
        const figureJson = chartData.visualization?.figure_json || chartData.figure_json;

        if (typeof figureJson === "string") {
          figureData = JSON.parse(figureJson);
        } else {
          figureData = figureJson;
        }

        // --- Theme Adaptation ---
        const currentTheme = theme === 'system' ? systemTheme : theme;
        const isDark = currentTheme === 'dark';

        const darkLayout = {
          paper_bgcolor: 'rgba(0,0,0,0)', // Transparent
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: { color: '#e2e8f0' }, // Slate-200
          xaxis: {
            gridcolor: '#334155', // Slate-700
            zerolinecolor: '#334155'
          },
          yaxis: {
            gridcolor: '#334155',
            zerolinecolor: '#334155'
          }
        };

        const finalLayout = {
          ...figureData.layout,
          ...(isDark ? darkLayout : {}),
          autosize: true,
          margin: { l: 50, r: 20, t: 30, b: 50 }, // Tighter margins
          height: isFullscreen ? window.innerHeight - 100 : 450
        };

        // Configure the plot
        const config = {
          responsive: true,
          displayModeBar: true,
          displaylogo: false,
          toImageButtonOptions: {
            format: "png",
            filename: "nexus_chart",
            height: 600,
            width: 800,
            scale: 2,
          },
          // Enable all interaction modes
          modeBarButtonsToAdd: ['pan2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
        };

        // Create the plot
        window.Plotly.newPlot(
          plotRef.current,
          figureData.data || [],
          finalLayout,
          config
        );

        // Add resize listener
        const handleResize = () => {
          if (plotRef.current && window.Plotly) {
            window.Plotly.Plots.resize(plotRef.current);
          }
        };
        window.addEventListener('resize', handleResize);

        return () => window.removeEventListener('resize', handleResize);

      } catch (err) {
        console.error("Error rendering chart:", err);
      }
    }
  }, [plotlyLoaded, chartData, theme, systemTheme, isFullscreen]);

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
      // Simple toggle state - actual fullscreen logic is complex with React component structure
      // relying on layout height update in useEffect
      setIsFullscreen(!isFullscreen);

      // Trigger resize after state change gives layout time to propagate
      setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
      }, 100);
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

  if (!chartData || (!chartData.success && chartData.status !== "success")) {
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
    <Card className={`overflow-hidden transition-all duration-300 ${isFullscreen ? 'fixed inset-4 z-50 shadow-2xl' : ''}`}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              Data Visualization
              {(chartData.visualization?.chart_type || chartData.chart_type) && (
                <Badge variant="secondary" className="text-xs">
                  {chartData.visualization?.chart_type || chartData.chart_type}
                </Badge>
              )}
            </CardTitle>
            <CardDescription>
              Interactive chart generated from your data analysis
              <span className="ml-2 text-xs text-muted-foreground">(Scroll to zoom, Drag to pan)</span>
            </CardDescription>
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleDownloadChart}
              disabled={!plotlyLoaded}
              title="Download PNG"
            >
              <Download className="h-4 w-4" />
            </Button>
            <Button
              variant={isFullscreen ? "default" : "outline"}
              size="sm"
              onClick={toggleFullscreen}
              disabled={!plotlyLoaded}
              title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
            >
              <Maximize2 className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="relative bg-background">
          {!plotlyLoaded && (
            <div className="absolute inset-0 flex items-center justify-center bg-background/80 z-10">
              <div className="flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="text-sm">Loading chart library...</span>
              </div>
            </div>
          )}
          <div
            ref={plotRef}
            className="w-full"
            style={{ minHeight: isFullscreen ? "80vh" : "450px" }}
          />
        </div>

        {chartData.generated_code && !isFullscreen && ( // Hide code in fullscreen
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