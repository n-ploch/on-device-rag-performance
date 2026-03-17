
export interface WorkerCheckResponse {
  ok: boolean;
  status: string | null;
  backend: string | null;
  models_loaded: boolean | null;
  error: string | null;
}

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
