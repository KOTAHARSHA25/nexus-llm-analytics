"use client";

import { useState, useEffect } from "react";
import { BarChart3, Wifi, WifiOff, AlertCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { getEndpoint } from "@/lib/config";

interface SystemStatus {
  status: string;
  services?: {
    ollama?: { status: string };
    chromadb?: { status: string };
  };
}

export function Header() {
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    checkSystemHealth();
    const interval = setInterval(checkSystemHealth, 300000);
    return () => clearInterval(interval);
  }, []);

  const checkSystemHealth = async () => {
    try {
      const response = await fetch(getEndpoint('health'));
      if (response.ok) {
        setSystemStatus({ status: 'healthy' });
      } else {
        setSystemStatus({ status: 'degraded' });
      }
    } catch (error) {
      setSystemStatus({ status: 'unhealthy' });
    } finally {
      setIsLoading(false);
    }
  };

  const getStatusBadge = () => {
    if (isLoading) {
      return (
        <Badge variant="outline" className="text-xs">
          <Wifi className="h-3 w-3 mr-1" />
          Checking...
        </Badge>
      );
    }

    if (!systemStatus) {
      return (
        <Badge variant="destructive" className="text-xs">
          <WifiOff className="h-3 w-3 mr-1" />
          Offline
        </Badge>
      );
    }

    switch (systemStatus.status) {
      case 'healthy':
        return (
          <Badge variant="default" className="text-xs bg-green-600 hover:bg-green-700">
            <Wifi className="h-3 w-3 mr-1" />
            Online
          </Badge>
        );
      case 'degraded':
        return (
          <Badge variant="secondary" className="text-xs bg-yellow-600 hover:bg-yellow-700">
            <AlertCircle className="h-3 w-3 mr-1" />
            Limited
          </Badge>
        );
      default:
        return (
          <Badge variant="destructive" className="text-xs">
            <WifiOff className="h-3 w-3 mr-1" />
            Offline
          </Badge>
        );
    }
  };

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between px-6">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-accent">
            <BarChart3 className="h-6 w-6 text-accent-foreground" />
          </div>
          <div>
            <h1 className="text-xl font-bold">Nexus LLM Analytics</h1>
            <p className="text-sm text-muted-foreground">
              AI-Powered Data Insights
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          {getStatusBadge()}
        </div>
      </div>
    </header>
  );
}
