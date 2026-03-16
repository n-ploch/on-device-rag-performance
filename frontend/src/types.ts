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

export type SSEMessage =
  | { type: 'log'; text: string }
  | { type: 'error'; text: string }
  | { type: 'done' };

export interface ConfigLoadResponse {
  ok: boolean;
  config: Record<string, unknown> | null;
  yaml_text: string;
  error: string | null;
}
