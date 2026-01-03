/**
 * Application Configuration
 * Centralized configuration for backend URLs and environment settings
 */

/**
 * Get the backend API base URL based on environment
 * Supports development, production, and custom configurations
 */
export function getBackendUrl(): string {
  // Check if running in browser
  if (typeof window !== 'undefined') {
    // Check for environment-specific override
    if (process.env.NEXT_PUBLIC_BACKEND_URL) {
      return process.env.NEXT_PUBLIC_BACKEND_URL;
    }

    // Production check (if hosted)
    if (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
      // In production, backend might be on same host different port or subdomain
      return `${window.location.protocol}//${window.location.hostname}:8000`;
    }
  }

  // Development default
  return 'http://127.0.0.1:8000';
}

/**
 * Backend API base URL
 * Use this constant throughout the application
 */
export const BACKEND_URL = getBackendUrl();

/**
 * API endpoint builder helper
 * Ensures consistent URL construction
 */
export function apiUrl(endpoint: string): string {
  // Remove leading slash if present to avoid double slashes
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint.slice(1) : endpoint;
  return `${BACKEND_URL}/${cleanEndpoint}`;
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
    baseUrl: BACKEND_URL,
    apiUrl: apiUrl,
    wsUrl: getWebSocketUrl(),
  },
  endpoints: {
    // ========== HEALTH & SYSTEM ==========
    health: '/api/health/health',
    healthStatus: '/api/health/status',
    networkInfo: '/api/health/network-info',
    cacheInfo: '/api/health/cache-info',
    clearCache: '/api/health/clear-cache',

    // ========== ANALYSIS (CORE FLOW) ==========
    analyze: '/api/analyze/',
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
