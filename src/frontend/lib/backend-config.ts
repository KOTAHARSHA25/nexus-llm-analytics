/**
 * Backend Configuration Utility
 * 
 * Manages backend URL configuration for local and remote backends
 */

const DEFAULT_LOCAL_URL = "http://localhost:8000";
const STORAGE_KEY = "nexus_backend_url";

/**
 * Get the current backend URL
 * Priority: localStorage > environment variable > default
 */
export function getBackendUrl(): string {
  // Check localStorage first (user preference)
  if (typeof window !== "undefined") {
    const savedUrl = localStorage.getItem(STORAGE_KEY);
    if (savedUrl) {
      return savedUrl;
    }
  }

  // Check environment variable
  const envUrl = process.env.NEXT_PUBLIC_BACKEND_URL;
  if (envUrl) {
    return envUrl;
  }

  // Fall back to default
  return DEFAULT_LOCAL_URL;
}

/**
 * Set the backend URL in localStorage
 */
export function setBackendUrl(url: string): void {
  if (typeof window !== "undefined") {
    localStorage.setItem(STORAGE_KEY, url);
  }
}

/**
 * Reset backend URL to default
 */
export function resetBackendUrl(): void {
  if (typeof window !== "undefined") {
    localStorage.removeItem(STORAGE_KEY);
  }
}

/**
 * Check if backend is running locally or remotely
 */
export function isLocalBackend(url?: string): boolean {
  const backendUrl = url || getBackendUrl();
  return (
    backendUrl.includes("localhost") ||
    backendUrl.includes("127.0.0.1") ||
    backendUrl.includes("0.0.0.0")
  );
}

/**
 * Build a full API endpoint URL
 */
export function buildApiUrl(endpoint: string): string {
  const baseUrl = getBackendUrl();
  const cleanEndpoint = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;
  return `${baseUrl}${cleanEndpoint}`;
}

/**
 * Fetch wrapper that uses configured backend URL
 */
export async function fetchFromBackend(
  endpoint: string,
  options?: RequestInit
): Promise<Response> {
  const url = buildApiUrl(endpoint);
  
  const defaultOptions: RequestInit = {
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  };

  return fetch(url, { ...defaultOptions, ...options });
}

/**
 * Test backend connection
 */
export async function testBackendConnection(url?: string): Promise<{
  success: boolean;
  message: string;
  data?: any;
}> {
  const backendUrl = url || getBackendUrl();
  
  try {
    const response = await fetch(`${backendUrl}/health`, {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    });

    if (response.ok) {
      const data = await response.json();
      return {
        success: true,
        message: "Backend is reachable",
        data,
      };
    } else {
      return {
        success: false,
        message: `Backend returned status ${response.status}`,
      };
    }
  } catch (error) {
    return {
      success: false,
      message: error instanceof Error ? error.message : "Failed to connect",
    };
  }
}

/**
 * Get backend connection status
 */
export async function getBackendStatus(): Promise<{
  url: string;
  isLocal: boolean;
  connected: boolean;
  health?: any;
}> {
  const url = getBackendUrl();
  const isLocal = isLocalBackend(url);
  const connectionTest = await testBackendConnection(url);

  return {
    url,
    isLocal,
    connected: connectionTest.success,
    health: connectionTest.data,
  };
}

/**
 * Validate URL format
 */
export function isValidUrl(url: string): boolean {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
}

/**
 * Get suggested URLs based on common patterns
 */
export function getSuggestedUrls(): Array<{
  label: string;
  url: string;
  description: string;
}> {
  return [
    {
      label: "Local (Port 8000)",
      url: "http://localhost:8000",
      description: "Default local development",
    },
    {
      label: "Local (Port 8080)",
      url: "http://localhost:8080",
      description: "Alternative local port",
    },
    {
      label: "Local (All interfaces)",
      url: "http://0.0.0.0:8000",
      description: "Listen on all network interfaces",
    },
  ];
}
