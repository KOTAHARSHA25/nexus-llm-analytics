"use client";

import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { CheckCircle2, XCircle, Loader2, Globe, Laptop } from "lucide-react";

interface BackendHealth {
  status: string;
  timestamp: string;
  version?: string;
  ollama_running?: boolean;
}

export function BackendUrlSettings() {
  const [backendUrl, setBackendUrl] = useState<string>("");
  const [customUrl, setCustomUrl] = useState<string>("");
  const [isTestingConnection, setIsTestingConnection] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<{
    connected: boolean;
    message: string;
    health?: BackendHealth;
  } | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  // Load saved backend URL on mount
  useEffect(() => {
    const savedUrl = localStorage.getItem("nexus_backend_url") || "http://localhost:8000";
    setBackendUrl(savedUrl);
    setCustomUrl(savedUrl);
  }, []);

  const testConnection = async (url: string) => {
    setIsTestingConnection(true);
    setConnectionStatus(null);

    try {
      const response = await fetch(`${url}/health`, {
        method: "GET",
        headers: {
          "Accept": "application/json",
        },
      });

      if (response.ok) {
        const health: BackendHealth = await response.json();
        setConnectionStatus({
          connected: true,
          message: "Successfully connected to backend!",
          health,
        });
      } else {
        setConnectionStatus({
          connected: false,
          message: `Backend returned status ${response.status}`,
        });
      }
    } catch (error) {
      setConnectionStatus({
        connected: false,
        message: error instanceof Error ? error.message : "Failed to connect to backend",
      });
    } finally {
      setIsTestingConnection(false);
    }
  };

  const saveBackendUrl = async () => {
    setIsSaving(true);
    
    // Test connection first
    await testConnection(customUrl);
    
    // Save to localStorage
    localStorage.setItem("nexus_backend_url", customUrl);
    setBackendUrl(customUrl);
    
    // Update environment variable (requires page reload)
    if (typeof window !== "undefined") {
      (window as any).NEXT_PUBLIC_BACKEND_URL = customUrl;
    }
    
    setIsSaving(false);
  };

  const usePreset = (url: string) => {
    setCustomUrl(url);
  };

  const isLocal = backendUrl.includes("localhost") || backendUrl.includes("127.0.0.1");

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Globe className="h-5 w-5" />
              Backend Connection
            </CardTitle>
            <CardDescription>
              Configure where your backend is running (local or remote)
            </CardDescription>
          </div>
          <Badge variant={isLocal ? "secondary" : "default"}>
            {isLocal ? (
              <>
                <Laptop className="mr-1 h-3 w-3" />
                Local
              </>
            ) : (
              <>
                <Globe className="mr-1 h-3 w-3" />
                Remote
              </>
            )}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Current Backend URL */}
        <div className="space-y-2">
          <Label>Current Backend URL</Label>
          <div className="flex items-center gap-2">
            <Input
              value={backendUrl}
              disabled
              className="font-mono text-sm"
            />
            <Button
              variant="outline"
              size="sm"
              onClick={() => testConnection(backendUrl)}
              disabled={isTestingConnection}
            >
              {isTestingConnection ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                "Test"
              )}
            </Button>
          </div>
        </div>

        {/* Connection Status */}
        {connectionStatus && (
          <Alert variant={connectionStatus.connected ? "default" : "destructive"}>
            <div className="flex items-start gap-2">
              {connectionStatus.connected ? (
                <CheckCircle2 className="h-4 w-4 mt-0.5" />
              ) : (
                <XCircle className="h-4 w-4 mt-0.5" />
              )}
              <div className="flex-1">
                <AlertDescription>{connectionStatus.message}</AlertDescription>
                {connectionStatus.health && (
                  <div className="mt-2 text-xs space-y-1">
                    <div>Status: {connectionStatus.health.status}</div>
                    {connectionStatus.health.version && (
                      <div>Version: {connectionStatus.health.version}</div>
                    )}
                    {connectionStatus.health.ollama_running !== undefined && (
                      <div>
                        Ollama: {connectionStatus.health.ollama_running ? "✅ Running" : "❌ Not Running"}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </Alert>
        )}

        {/* Quick Presets */}
        <div className="space-y-2">
          <Label>Quick Presets</Label>
          <div className="grid grid-cols-2 gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => usePreset("http://localhost:8000")}
              className="justify-start"
            >
              <Laptop className="mr-2 h-4 w-4" />
              Local (Port 8000)
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => usePreset("http://localhost:8080")}
              className="justify-start"
            >
              <Laptop className="mr-2 h-4 w-4" />
              Local (Port 8080)
            </Button>
          </div>
        </div>

        {/* Custom URL */}
        <div className="space-y-2">
          <Label htmlFor="custom-url">Custom Backend URL</Label>
          <div className="flex gap-2">
            <Input
              id="custom-url"
              value={customUrl}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setCustomUrl(e.target.value)}
              placeholder="http://your-custom-url:8000"
              className="font-mono text-sm"
            />
            <Button
              onClick={saveBackendUrl}
              disabled={isSaving || !customUrl}
            >
              {isSaving ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                "Save"
              )}
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            💡 Enter your backend URL (e.g., http://192.168.1.100:8000 for network access)
          </p>
        </div>

        {/* Instructions */}
        <div className="rounded-lg border p-4 space-y-3 text-sm">
          <div className="font-semibold">📋 Setup Instructions:</div>
          
          <div>
            <div className="font-medium mb-1">🖥️ Local Backend:</div>
            <ol className="list-decimal list-inside space-y-1 text-xs text-muted-foreground">
              <li>Make sure Ollama is running: <code className="bg-muted px-1 rounded">ollama serve</code></li>
              <li>Start backend: <code className="bg-muted px-1 rounded">python scripts/launch.py --backend-only</code></li>
              <li>Use URL: <code className="bg-muted px-1 rounded">http://localhost:8000</code></li>
            </ol>
          </div>

          <div>
            <div className="font-medium mb-1">🌐 Network Access:</div>
            <ol className="list-decimal list-inside space-y-1 text-xs text-muted-foreground">
              <li>Find your IP address: <code className="bg-muted px-1 rounded">ipconfig</code> (Windows) or <code className="bg-muted px-1 rounded">ifconfig</code> (Linux/Mac)</li>
              <li>Use: <code className="bg-muted px-1 rounded">http://YOUR_IP:8000</code></li>
              <li>Ensure firewall allows port 8000</li>
            </ol>
          </div>
        </div>

        {/* Warning */}
        <Alert>
          <AlertDescription className="text-xs">
            ⚠️ <strong>Note:</strong> Changing the backend URL requires a page reload to take effect.
            Click "Save" and then refresh your browser.
          </AlertDescription>
        </Alert>
      </CardContent>
    </Card>
  );
}
