// Optimized State Management with useReducer and Performance Hooks
// BEFORE: Multiple useState hooks causing excessive re-renders (O(n) state updates)
// AFTER: Centralized state with useReducer, memoized components (O(1) state updates)

"use client";

import { useReducer, useMemo, useCallback, useRef, startTransition } from "react";
import { debounce } from "lodash";

// Performance-optimized interfaces
export interface FileInfo {
  name: string;
  type: string;
  columns?: string[];
  id: string; // Added for O(1) lookups
  uploadedAt: number; // For efficient sorting
}

export interface PluginInfo {
  name: string;
  description: string;
  icon: React.ComponentType<any>;
  color: string;
  capabilities: string[];
  id: string; // Added for O(1) lookups
}

// Centralized state interface (reduces memory fragmentation)
interface DashboardState {
  // File management
  uploadedFiles: Map<string, FileInfo>; // O(1) lookups instead of O(n) array operations
  fileUploadProgress: number;
  
  // Query management
  currentQuery: string;
  textInput: string;
  queryHistory: string[]; // Keep as array for display order
  queryHistoryMap: Set<string>; // O(1) duplicate checking
  
  // UI state
  inputMode: "file" | "text";
  sidebarOpen: boolean;
  showSettings: boolean;
  showSetupWizard: boolean;
  selectedPlugin: string;
  
  // Analysis state
  isLoading: boolean;
  analysisProgress: number;
  currentAnalysisId?: string;
  retryCount: number;
  statusMsg?: string;
  
  // Results and errors
  results: any;
  hasResults: boolean;
  errorMsg: any;
  isDownloading: boolean;
  
  // First-time user
  isFirstTime: boolean;
}

// Action types for useReducer (type-safe and performant)
type DashboardAction =
  | { type: 'SET_UPLOADED_FILES'; payload: FileInfo[] }
  | { type: 'ADD_UPLOADED_FILE'; payload: FileInfo }
  | { type: 'REMOVE_UPLOADED_FILE'; payload: string }
  | { type: 'SET_QUERY'; payload: string }
  | { type: 'SET_TEXT_INPUT'; payload: string }
  | { type: 'ADD_TO_HISTORY'; payload: string }
  | { type: 'CLEAR_HISTORY' }
  | { type: 'SET_INPUT_MODE'; payload: "file" | "text" }
  | { type: 'TOGGLE_SIDEBAR' }
  | { type: 'SET_SIDEBAR_OPEN'; payload: boolean }
  | { type: 'SET_SHOW_SETTINGS'; payload: boolean }
  | { type: 'SET_SHOW_SETUP_WIZARD'; payload: boolean }
  | { type: 'SET_SELECTED_PLUGIN'; payload: string }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ANALYSIS_PROGRESS'; payload: number }
  | { type: 'SET_CURRENT_ANALYSIS_ID'; payload: string | undefined }
  | { type: 'SET_RETRY_COUNT'; payload: number }
  | { type: 'SET_STATUS_MSG'; payload: string | undefined }
  | { type: 'SET_RESULTS'; payload: any }
  | { type: 'SET_HAS_RESULTS'; payload: boolean }
  | { type: 'SET_ERROR_MSG'; payload: any }
  | { type: 'SET_IS_DOWNLOADING'; payload: boolean }
  | { type: 'SET_IS_FIRST_TIME'; payload: boolean }
  | { type: 'SET_FILE_UPLOAD_PROGRESS'; payload: number }
  | { type: 'RESET_ANALYSIS_STATE' }
  | { type: 'BATCH_UPDATE'; payload: Partial<DashboardState> }; // For atomic updates

// Performance-optimized reducer with batched updates
function dashboardReducer(state: DashboardState, action: DashboardAction): DashboardState {
  switch (action.type) {
    case 'SET_UPLOADED_FILES': {
      const filesMap = new Map<string, FileInfo>();
      action.payload.forEach(file => {
        const fileWithId = { ...file, id: file.id || `${file.name}_${Date.now()}` };
        filesMap.set(fileWithId.id, fileWithId);
      });
      return { ...state, uploadedFiles: filesMap };
    }
    
    case 'ADD_UPLOADED_FILE': {
      const newFile = { ...action.payload, id: action.payload.id || `${action.payload.name}_${Date.now()}` };
      const newMap = new Map(state.uploadedFiles);
      newMap.set(newFile.id, newFile);
      return { ...state, uploadedFiles: newMap };
    }
    
    case 'REMOVE_UPLOADED_FILE': {
      const newMap = new Map(state.uploadedFiles);
      newMap.delete(action.payload);
      return { ...state, uploadedFiles: newMap };
    }
    
    case 'SET_QUERY':
      return { ...state, currentQuery: action.payload };
    
    case 'SET_TEXT_INPUT':
      return { ...state, textInput: action.payload };
    
    case 'ADD_TO_HISTORY': {
      if (state.queryHistoryMap.has(action.payload)) {
        return state; // Prevent duplicates with O(1) check
      }
      const newHistorySet = new Set(state.queryHistoryMap);
      newHistorySet.add(action.payload);
      return {
        ...state,
        queryHistory: [action.payload, ...state.queryHistory.slice(0, 9)],
        queryHistoryMap: newHistorySet
      };
    }
    
    case 'CLEAR_HISTORY':
      return {
        ...state,
        queryHistory: [],
        queryHistoryMap: new Set()
      };
    
    case 'SET_INPUT_MODE':
      return { ...state, inputMode: action.payload };
    
    case 'TOGGLE_SIDEBAR':
      return { ...state, sidebarOpen: !state.sidebarOpen };
    
    case 'SET_SIDEBAR_OPEN':
      return { ...state, sidebarOpen: action.payload };
    
    case 'SET_SHOW_SETTINGS':
      return { ...state, showSettings: action.payload };
    
    case 'SET_SHOW_SETUP_WIZARD':
      return { ...state, showSetupWizard: action.payload };
    
    case 'SET_SELECTED_PLUGIN':
      return { ...state, selectedPlugin: action.payload };
    
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };
    
    case 'SET_ANALYSIS_PROGRESS':
      return { ...state, analysisProgress: action.payload };
    
    case 'SET_CURRENT_ANALYSIS_ID':
      return { ...state, currentAnalysisId: action.payload };
    
    case 'SET_RETRY_COUNT':
      return { ...state, retryCount: action.payload };
    
    case 'SET_STATUS_MSG':
      return { ...state, statusMsg: action.payload };
    
    case 'SET_RESULTS':
      return { ...state, results: action.payload };
    
    case 'SET_HAS_RESULTS':
      return { ...state, hasResults: action.payload };
    
    case 'SET_ERROR_MSG':
      return { ...state, errorMsg: action.payload };
    
    case 'SET_IS_DOWNLOADING':
      return { ...state, isDownloading: action.payload };
    
    case 'SET_IS_FIRST_TIME':
      return { ...state, isFirstTime: action.payload };
    
    case 'SET_FILE_UPLOAD_PROGRESS':
      return { ...state, fileUploadProgress: action.payload };
    
    case 'RESET_ANALYSIS_STATE':
      return {
        ...state,
        isLoading: false,
        analysisProgress: 0,
        currentAnalysisId: undefined,
        retryCount: 0,
        statusMsg: undefined,
        errorMsg: null
      };
    
    case 'BATCH_UPDATE':
      return { ...state, ...action.payload };
    
    default:
      return state;
  }
}

// Initial state factory (prevents recreation on each render)
const createInitialState = (): DashboardState => ({
  uploadedFiles: new Map<string, FileInfo>(),
  fileUploadProgress: 0,
  currentQuery: "",
  textInput: "",
  queryHistory: [],
  queryHistoryMap: new Set<string>(),
  inputMode: "file" as const,
  sidebarOpen: false,
  showSettings: false,
  showSetupWizard: false,
  selectedPlugin: "Auto-Select Agent",
  isLoading: false,
  analysisProgress: 0,
  currentAnalysisId: undefined,
  retryCount: 0,
  statusMsg: undefined,
  results: null,
  hasResults: false,
  errorMsg: null,
  isDownloading: false,
  isFirstTime: false
});

// Performance-optimized custom hook
export function useDashboardState() {
  const [state, dispatch] = useReducer(dashboardReducer, null, createInitialState);
  
  // Refs for stable references (prevent unnecessary re-renders)
  const apiCallsRef = useRef(new Map<string, AbortController>());
  const progressIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  // Memoized derived state (computed once per state change)
  const derivedState = useMemo(() => ({
    uploadedFilesList: Array.from(state.uploadedFiles.values()).sort((a, b) => b.uploadedAt - a.uploadedAt),
    hasUploadedFiles: state.uploadedFiles.size > 0,
    canSubmitQuery: state.inputMode === "file" 
      ? state.uploadedFiles.size > 0 
      : state.textInput.trim().length > 0,
    isAnalysisActive: state.isLoading || state.analysisProgress > 0,
  }), [state.uploadedFiles, state.inputMode, state.textInput, state.isLoading, state.analysisProgress]);
  
  // Debounced input handlers (prevent excessive API calls)
  const debouncedSetQuery = useCallback(
    debounce((query: string) => {
      dispatch({ type: 'SET_QUERY', payload: query });
    }, 300),
    []
  );
  
  const debouncedSetTextInput = useCallback(
    debounce((text: string) => {
      dispatch({ type: 'SET_TEXT_INPUT', payload: text });
    }, 300),
    []
  );
  
  // Optimized file operations with batch updates
  const handleFileUpload = useCallback((files: FileInfo[]) => {
    startTransition(() => {
      dispatch({ type: 'SET_UPLOADED_FILES', payload: files });
    });
  }, []);
  
  const addFileWithId = useCallback((file: Omit<FileInfo, 'id' | 'uploadedAt'>) => {
    const fileWithMetadata: FileInfo = {
      ...file,
      id: `${file.name}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      uploadedAt: Date.now()
    };
    dispatch({ type: 'ADD_UPLOADED_FILE', payload: fileWithMetadata });
  }, []);
  
  const removeFileById = useCallback((fileId: string) => {
    dispatch({ type: 'REMOVE_UPLOADED_FILE', payload: fileId });
  }, []);
  
  // Optimized query history operations
  const addToHistoryOptimized = useCallback((query: string) => {
    if (query.trim() && !state.queryHistoryMap.has(query)) {
      dispatch({ type: 'ADD_TO_HISTORY', payload: query });
    }
  }, [state.queryHistoryMap]);
  
  // Batch state updates for performance
  const batchUpdate = useCallback((updates: Partial<DashboardState>) => {
    dispatch({ type: 'BATCH_UPDATE', payload: updates });
  }, []);
  
  // Cleanup function for memory management
  const cleanup = useCallback(() => {
    // Cancel ongoing API calls
    apiCallsRef.current.forEach((controller) => {
      controller.abort();
    });
    apiCallsRef.current.clear();
    
    // Clear intervals
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }
  }, []);
  
  // Progress simulation with better memory management
  const startProgressSimulation = useCallback(() => {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
    }
    
    progressIntervalRef.current = setInterval(() => {
      dispatch({ 
        type: 'SET_ANALYSIS_PROGRESS', 
        payload: state.analysisProgress >= 85 ? 85 : Math.min(state.analysisProgress + Math.random() * 15, 85)
      });
      
      if (state.analysisProgress >= 85) {
        if (progressIntervalRef.current) {
          clearInterval(progressIntervalRef.current);
          progressIntervalRef.current = null;
        }
      }
    }, 500);
  }, []);
  
  const stopProgressSimulation = useCallback(() => {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }
  }, []);
  
  return {
    // State
    state,
    derivedState,
    
    // Basic actions
    dispatch,
    batchUpdate,
    
    // File operations
    handleFileUpload,
    addFileWithId,
    removeFileById,
    
    // Input operations
    setQuery: debouncedSetQuery,
    setTextInput: debouncedSetTextInput,
    
    // History operations
    addToHistory: addToHistoryOptimized,
    
    // Progress management
    startProgressSimulation,
    stopProgressSimulation,
    
    // Cleanup
    cleanup,
    
    // Refs for external access
    apiCallsRef,
    progressIntervalRef
  };
}

// Performance monitoring hook
export function usePerformanceMonitor() {
  const renderCountRef = useRef(0);
  const lastRenderTimeRef = useRef(Date.now());
  
  renderCountRef.current += 1;
  const currentTime = Date.now();
  const timeSinceLastRender = currentTime - lastRenderTimeRef.current;
  lastRenderTimeRef.current = currentTime;
  
  const performanceData = useMemo(() => ({
    renderCount: renderCountRef.current,
    timeSinceLastRender,
    averageRenderTime: currentTime / renderCountRef.current
  }), [timeSinceLastRender, currentTime]);
  
  // Log performance issues in development
  if (process.env.NODE_ENV === 'development' && timeSinceLastRender > 16) {
    console.warn(`Slow render detected: ${timeSinceLastRender}ms (target: 16ms for 60fps)`);
  }
  
  return performanceData;
}