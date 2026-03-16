export interface EnvValues {
  WORKER_URL: string;
  LOCAL_MODELS_DIR: string;
  LOCAL_DATASETS_DIR: string;
  HF_TOKEN: string;
  LANGFUSE_PUBLIC_KEY: string;
  LANGFUSE_SECRET_KEY: string;
  LANGFUSE_BASE_URL: string;
  LLM_API_KEY: string;
  LLAMA_SERVER_PATH: string;
  EMBEDDING_PORT: string;
  GENERATION_PORT: string;
}

export const ENV_KEY_LABELS: Record<keyof EnvValues, string> = {
  WORKER_URL: 'Worker URL',
  LOCAL_MODELS_DIR: 'Local Models Dir',
  LOCAL_DATASETS_DIR: 'Local Datasets Dir',
  HF_TOKEN: 'HuggingFace Token',
  LANGFUSE_PUBLIC_KEY: 'Langfuse Public Key',
  LANGFUSE_SECRET_KEY: 'Langfuse Secret Key',
  LANGFUSE_BASE_URL: 'Langfuse Base URL',
  LLM_API_KEY: 'LLM API Key',
  LLAMA_SERVER_PATH: 'llama-server Path',
  EMBEDDING_PORT: 'Embedding Port',
  GENERATION_PORT: 'Generation Port',
};

export const SENSITIVE_KEYS: Set<keyof EnvValues> = new Set([
  'HF_TOKEN',
  'LANGFUSE_PUBLIC_KEY',
  'LANGFUSE_SECRET_KEY',
  'LLM_API_KEY',
]);

export interface AppStatus {
  config_loaded: boolean;
  run_id: string | null;
  is_running: boolean;
}

export interface ConfigLoadResponse {
  ok: boolean;
  config: Record<string, unknown> | null;
  yaml_text: string;
  error: string | null;
}

// ---------------------------------------------------------------------------
// Structured SSE event types emitted by the orchestrator API
// ---------------------------------------------------------------------------

export interface RunStartEvent {
  type: 'run_start';
  run_id: string;
  config_index: number;
  total_configs: number;
  rep: number;
  total_reps: number;
  session_id: string;
  total_entries: number;
  retrieval_model: string;
  retrieval_quantization: string;
  generation_model: string;
  generation_quantization: string;
  k: number;
}

export interface EntryResultEvent {
  type: 'entry_result';
  entry_index: number;
  total_entries: number;
  run_id: string;
  request: {
    claim_id: string;
    input: string;
  };
  response: {
    output: string;
    recall_at_k: number | null;
    precision_at_k: number | null;
    mrr: number | null;
    is_abstention: boolean;
    latency_ms: number;
    tokens_per_second: number;
    ram_mb: number;
  };
}

export interface EntryErrorEvent {
  type: 'entry_error';
  entry_index: number;
  claim_id: string;
  message: string;
}

export interface RunCompleteEvent {
  type: 'run_complete';
  run_id: string;
  avg_recall: number;
  avg_precision: number;
  avg_mrr: number;
  abstentions: number;
  total: number;
}

export interface DryRunResultEvent {
  type: 'dry_run_result';
  total_entries: number;
  total_configs: number;
  run_ids: string[];
}

export interface StoppedEvent {
  type: 'stopped';
  run_id: string;
  completed_entries: number;
  total_entries: number;
  avg_recall: number;
  avg_precision: number;
  avg_mrr: number;
  abstentions: number;
  total: number;
}

export interface FatalErrorEvent {
  type: 'error';
  message: string;
}

export interface DoneEvent {
  type: 'done';
}

export type RunEvent =
  | RunStartEvent
  | EntryResultEvent
  | EntryErrorEvent
  | RunCompleteEvent
  | StoppedEvent
  | DryRunResultEvent
  | FatalErrorEvent
  | DoneEvent;
