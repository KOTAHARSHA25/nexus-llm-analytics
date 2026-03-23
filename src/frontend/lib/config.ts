/**
 * Application Configuration
 * Centralized configuration for backend URLs and environment settings
 * 
 * URL resolution priority: localStorage > env variable > hostname detection > default
 */

const STORAGE_KEY = "nexus_backend_url";

/**
 * Get the backend API base URL based on environment
 * Supports development, production, localStorage overrides, and custom configurations
 */
export function getBackendUrl(): string {
  // Check if running in browser
  if (typeof window !== 'undefined') {
    // 1. Check localStorage first (user preference from Settings UI)
    const savedUrl = localStorage.getItem(STORAGE_KEY);
    if (savedUrl) {
      return savedUrl;
    }

    // 2. Check for environment-specific override
    if (process.env.NEXT_PUBLIC_BACKEND_URL) {
      return process.env.NEXT_PUBLIC_BACKEND_URL;
    }

    // 3. Production check (if hosted on non-localhost)
    if (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
      return `${window.location.protocol}//${window.location.hostname}:8000`;
    }
  }

  // 4. Development default
  return 'http://localhost:8000';
}

/**
 * Set the backend URL in localStorage (called from Settings UI)
 */
export function setBackendUrl(url: string): void {
  if (typeof window !== 'undefined') {
    localStorage.setItem(STORAGE_KEY, url);
  }
}

/**
 * Reset backend URL to default (removes localStorage override)
 */
export function resetBackendUrl(): void {
  if (typeof window !== 'undefined') {
    localStorage.removeItem(STORAGE_KEY);
  }
}

/**
 * API endpoint builder helper
 * Calls getBackendUrl() dynamically to pick up localStorage changes
 */
export function apiUrl(endpoint: string): string {
  const baseUrl = getBackendUrl();
  if (!endpoint) return baseUrl;
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint.slice(1) : endpoint;
  return `${baseUrl}/${cleanEndpoint}`;
}

/**
 * WebSocket URL for real-time features (if needed in future)
 */
export function getWebSocketUrl(): string {
  const baseUrl = getBackendUrl();
  return baseUrl.replace('http://', 'ws://').replace('https://', 'wss://');
}

/**
 * Configuration object for easy access
 */
export const config = {
  backend: {
    get baseUrl() { return getBackendUrl(); },
    apiUrl: apiUrl,
    get wsUrl() { return getWebSocketUrl(); },
  },
  endpoints: {
    // ========== HEALTH & SYSTEM ==========
    health: '/api/health/status',
    healthStatus: '/api/health/status',
    networkInfo: '/api/health/network-info',
    cacheInfo: '/api/health/cache-info',
    clearCache: '/api/health/clear-cache',
    mode: '/api/mode',

    // ========== ANALYSIS (CORE FLOW) ==========
    analyze: '/api/analyze/',
    stream: '/api/analyze/stream',                 // Streaming analysis with SSE
    analyzeCancel: '/api/analyze/cancel',          // Note: Use with /{analysis_id}
    analyzeStatus: '/api/analyze/status',          // Note: Use with /{analysis_id}
    analyzeReview: '/api/analyze/review-insights',
    analyzeRoutingStats: '/api/analyze/routing-stats',

    // ========== MODELS ==========
    modelsPreferences: '/api/models/preferences',
    modelsAvailable: '/api/models/available',
    modelsTestResults: '/api/models/test-results',
    modelsTestModel: '/api/models/test-model',
    modelsRecommendations: '/api/models/recommendations',
    modelsSetupComplete: '/api/models/setup-complete',

    // ========== FILE UPLOAD ==========
    uploadDocuments: '/api/upload/',
    uploadRawText: '/api/upload/raw-text',         // For direct text analysis
    previewFile: '/api/upload/preview-file',       // Note: Use with /{filename}
    downloadFile: '/api/upload/download-file',     // Note: Use with /{filename}

    // ========== VISUALIZATION ==========
    visualizeGoalBased: '/api/visualize/goal-based',
    visualizeSuggestions: '/api/visualize/suggestions',
    visualizeTypes: '/api/visualize/types',

    // ========== VISUALIZATION ENHANCEMENT (LIDA-inspired) ==========
    vizEdit: '/api/viz/edit',
    vizExplain: '/api/viz/explain',
    vizRecommend: '/api/viz/recommend',

    // ========== HISTORY ==========
    history: '/api/history/',
    historyAdd: '/api/history/add',
    historyClear: '/api/history/clear',
    historySearch: '/api/history/search',
    historyStats: '/api/history/stats',

    // ========== REPORTS ==========
    generateReport: '/api/report/',
    downloadReport: '/api/report/download-report/',
    downloadLog: '/api/report/download-log',
    downloadAudit: '/api/report/download-audit',

    // ========== FEEDBACK ==========
    feedback: '/api/feedback/',
    feedbackStats: '/api/feedback/stats',
    feedbackExport: '/api/feedback/export',
    feedbackReset: '/api/feedback/reset',

    // ========== RUNNING ANALYSES ==========
    analyzeRunning: '/api/analyze/running',
  },
} as const;

/**
 * Helper to get full endpoint URL
 * @param endpointKey - Key from config.endpoints
 * @returns Full URL for the endpoint
 */
export function getEndpoint(endpointKey: keyof typeof config.endpoints): string {
  return apiUrl(config.endpoints[endpointKey]);
}
