/**
 * Strict type definitions for backend SSE (Server-Sent Events) stream
 * 
 * Must match the StreamEvent Pydantic model in src/backend/api/analyze.py
 */

export interface ExecutionPlan {
  model?: string;
  execution_method?: string;
  reasoning?: string;
  review_level?: string;
  complexity_score?: number;
}

export interface StreamEvent {
  step: 'init' | 'validation' | 'loading' | 'routing' | 'mode' | 'thinking' | 'token' | 'agent_start' | 'agent_complete' | 'formatting' | 'complete' | 'error' | 'plan';
  message?: string;
  progress?: number;
  token?: string;
  error?: string;
  result?: AnalysisResult;
  files?: string[];
  plan?: ExecutionPlan;
}

export type AgentStatus = 'idle' | 'thinking' | 'delegating' | 'working' | 'complete' | 'error';

export interface AgentState {
  id: string;
  name: string;
  status: AgentStatus;
  message?: string;
  target?: string; // For delegation
}

/**
 * Matches backend AnalyzeResponse Pydantic model + extra fields
 * sent via the streaming complete event in analyze.py
 */
export interface AnalysisResult {
  // --- Core fields (in AnalyzeResponse Pydantic model) ---
  result: string | any; // Allow any for backward compat if needed, but string preferred
  visualization?: any;
  code?: string;
  generated_code?: string;
  execution_id?: string;
  execution_time?: number;
  execution_method?: string;
  query: string;
  filename?: string | null;
  filenames?: string[] | null;
  analysis_id?: string | null;
  status: string;
  error?: string | null;
  agent?: string;

  // --- Extended fields (sent by stream complete event) ---
  model?: string;
  model_used?: string; // Alternate name
  token_count?: number;
  agent_used?: boolean;
  plan?: ExecutionPlan;

  // --- Agent result fields (injected by specific plugins) ---
  success?: boolean;
  answer?: string; // Alternate to result
  explanation?: string;
  routing_tier?: string;
  routing_info?: Record<string, any>;
  cache_hit?: boolean;
  confidence?: number;
  type?: string;
  preview?: any[]; // Array of objects usually
  describe?: any;
  value_counts?: any;
  filtered_count?: number;
  metadata?: Record<string, any>;
  suggestion?: string; // For errors
}
