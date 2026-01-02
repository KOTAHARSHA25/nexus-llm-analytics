"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { getEndpoint } from "@/lib/config";
import { 
  History, Settings, BarChart3, TrendingUp, DollarSign, Activity, Cpu, 
  Sparkles, Clock, MessageSquare, Brain, Database, Target, Zap, Bot,
  FileText, Download
} from "lucide-react";

interface PluginInfo {
  name: string;
  description: string;
  icon: React.ComponentType<any>;
  color: string;
  capabilities: string[];
}

interface AnalyticsSidebarProps {
  queryHistory: string[];
  selectedPlugin: string;
  onPluginSelect: (plugin: string) => void;
  onHistoryClick: (query: string) => void;
  onClearHistory: () => void;
  onOpenSettings: () => void;
}

const plugins: PluginInfo[] = [
  {
    name: "Auto-Select Agent",
    description: "AI chooses the best agent automatically",
    icon: Bot,
    color: "purple",
    capabilities: ["Smart Selection", "Adaptive", "Multi-Modal"]
  },
  {
    name: "Statistical Agent", 
    description: "Advanced statistical analysis, hypothesis testing, correlation analysis",
    icon: BarChart3,
    color: "blue",
    capabilities: ["T-tests", "ANOVA", "Correlation"]
  },
  {
    name: "Time Series Agent",
    description: "ARIMA forecasting, trend analysis, seasonality detection", 
    icon: TrendingUp,
    color: "green",
    capabilities: ["ARIMA", "Forecasting", "Trend Analysis"]
  },
  {
    name: "Financial Agent",
    description: "Business metrics, profitability analysis, financial health assessment",
    icon: DollarSign,
    color: "emerald", 
    capabilities: ["ROI", "Profitability", "Financial Ratios"]
  },
  {
    name: "ML Insights Agent",
    description: "Machine learning analysis, clustering, anomaly detection, PCA",
    icon: Brain,
    color: "violet",
    capabilities: ["Clustering", "PCA", "Anomaly Detection"]
  },
  {
    name: "SQL Agent",
    description: "Multi-database support, query generation, schema analysis",
    icon: Database,
    color: "orange",
    capabilities: ["Query Generation", "Schema Analysis", "Multi-DB"]
  }
];

export function AnalyticsSidebar({ 
  queryHistory, 
  selectedPlugin, 
  onPluginSelect, 
  onHistoryClick, 
  onClearHistory,
  onOpenSettings 
}: AnalyticsSidebarProps) {
  const [showHistory, setShowHistory] = useState(true);
  const [showAgents, setShowAgents] = useState(true);

  return (
    <div className="glass-card h-full bg-gray-900/20 backdrop-blur-xl border-r border-white/10">
      <ScrollArea className="h-full p-4">
        <div className="space-y-6">
          {/* Quick Actions */}
          <Card className="glass-card border-white/10">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg text-white flex items-center gap-2">
                <Settings className="h-5 w-5 text-purple-300" />
                Quick Actions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Button
                onClick={onOpenSettings}
                variant="ghost"
                className="w-full justify-start bg-white/10 border border-white/20 text-white hover:bg-white/20 transition-all duration-300"
              >
                <Settings className="h-4 w-4 mr-2" />
                Model Settings
              </Button>
            </CardContent>
          </Card>

          {/* Recent Queries History */}
          <Card className="glass-card border-white/10">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg text-white flex items-center gap-2">
                  <History className="h-5 w-5 text-purple-300" />
                  Recent Queries
                </CardTitle>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowHistory(!showHistory)}
                  className="text-white/70 hover:text-white"
                >
                  {showHistory ? "Hide" : "Show"}
                </Button>
              </div>
            </CardHeader>
            {showHistory && (
              <CardContent className="space-y-2">
                {queryHistory.length === 0 ? (
                  <p className="text-sm text-white/60 text-center py-4">
                    No recent queries
                  </p>
                ) : (
                  <>
                    <div className="space-y-2">
                      {queryHistory.slice(-5).reverse().map((query, index) => (
                        <div
                          key={index}
                          className="p-3 rounded-lg bg-white/5 border border-white/10 text-sm text-white/90 cursor-pointer hover:bg-white/10 transition-all duration-200 group"
                          onClick={() => onHistoryClick(query)}
                        >
                          <div className="flex items-start gap-2">
                            <Clock className="h-4 w-4 text-purple-300 mt-0.5 flex-shrink-0" />
                            <p className="line-clamp-2 group-hover:text-white transition-colors">
                              {query}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                    {queryHistory.length > 0 && (
                      <Button
                        variant="ghost" 
                        size="sm"
                        onClick={onClearHistory}
                        className="w-full text-white/60 hover:text-white/80"
                      >
                        Clear History
                      </Button>
                    )}
                  </>
                )}
              </CardContent>
            )}
          </Card>

          <Separator className="bg-white/10" />

          {/* Specialized AI Agents */}
          <Card className="glass-card border-white/10">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-lg text-white flex items-center gap-2">
                    <Sparkles className="h-5 w-5 text-purple-300" />
                    AI Agents
                  </CardTitle>
                  <CardDescription className="text-white/60 text-sm">
                    Choose a specialized agent
                  </CardDescription>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowAgents(!showAgents)}
                  className="text-white/70 hover:text-white"
                >
                  {showAgents ? "Hide" : "Show"}
                </Button>
              </div>
            </CardHeader>
            {showAgents && (
              <CardContent className="space-y-3">
                {plugins.map((plugin) => {
                  const IconComponent = plugin.icon;
                  const isSelected = selectedPlugin === plugin.name;
                  
                  return (
                    <div
                      key={plugin.name}
                      className={`plugin-card p-3 rounded-lg cursor-pointer transition-all duration-300 ${
                        isSelected 
                          ? 'bg-gradient-to-r from-purple-500/20 to-blue-500/20 border-purple-400/50 shadow-lg shadow-purple-500/25' 
                          : 'bg-white/5 border-white/10 hover:border-white/30 hover:bg-white/10'
                      }`}
                      onClick={() => onPluginSelect(plugin.name)}
                    >
                      <div className="flex items-start gap-3">
                        <div className={`p-2 rounded-lg bg-${plugin.color}-500/20`}>
                          <IconComponent className={`h-5 w-5 text-${plugin.color}-400`} />
                        </div>
                        <div className="flex-1 min-w-0">
                          <h4 className="font-medium text-white text-sm">
                            {plugin.name}
                          </h4>
                          <p className="text-xs text-white/70 line-clamp-2 mt-1">
                            {plugin.description}
                          </p>
                          <div className="flex flex-wrap gap-1 mt-2">
                            {plugin.capabilities.slice(0, 2).map((cap) => (
                              <Badge 
                                key={cap} 
                                variant="secondary" 
                                className="text-xs bg-white/10 text-white/70 border-0"
                              >
                                {cap}
                              </Badge>
                            ))}
                            {plugin.capabilities.length > 2 && (
                              <Badge 
                                variant="secondary" 
                                className="text-xs bg-white/10 text-white/70 border-0"
                              >
                                +{plugin.capabilities.length - 2}
                              </Badge>
                            )}
                          </div>
                        </div>
                        {isSelected && (
                          <div className="flex-shrink-0">
                            <div className="w-2 h-2 rounded-full bg-purple-400"></div>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </CardContent>
            )}
          </Card>

          {/* Recent Reports */}
          <Card className="glass-card border-white/10">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm text-white flex items-center gap-2">
                <FileText className="h-4 w-4 text-blue-400" />
                Recent Reports
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start text-white/70 hover:text-white hover:bg-white/10"
                onClick={() => {
                  // Download the most recent report
                  const link = document.createElement("a");
                  link.href = getEndpoint("downloadReport");
                  link.download = "latest_report.pdf";
                  document.body.appendChild(link);
                  link.click();
                  document.body.removeChild(link);
                }}
              >
                <Download className="h-4 w-4 mr-2" />
                Download Latest Report
              </Button>
              <div className="text-xs text-white/60 text-center py-2">
                Generate reports from analysis results
              </div>
            </CardContent>
          </Card>

          {/* Quick Stats */}
          <Card className="glass-card border-white/10">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm text-white flex items-center gap-2">
                <Activity className="h-4 w-4 text-green-400" />
                Session Stats
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-white/70">Queries Run</span>
                <span className="text-white font-medium">{queryHistory.length}</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-white/70">Active Agent</span>
                <span className="text-white font-medium text-xs">
                  {selectedPlugin.split(' ')[0]}
                </span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-white/70">Status</span>
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 rounded-full bg-green-400"></div>
                  <span className="text-green-400 text-xs">Ready</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </ScrollArea>
    </div>
  );
}