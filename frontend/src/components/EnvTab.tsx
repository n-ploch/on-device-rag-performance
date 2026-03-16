import { useEffect, useState } from 'react';
import { getEnv, saveEnv } from '../api';
import { ENV_KEY_LABELS, SENSITIVE_KEYS } from '../types';
import type { EnvValues } from '../types';

const EMPTY_ENV: EnvValues = {
  WORKER_URL: '',
  LOCAL_MODELS_DIR: '',
  LOCAL_DATASETS_DIR: '',
  HF_TOKEN: '',
  LANGFUSE_PUBLIC_KEY: '',
  LANGFUSE_SECRET_KEY: '',
  LANGFUSE_BASE_URL: '',
  LLM_API_KEY: '',
  LLAMA_SERVER_PATH: '',
  EMBEDDING_PORT: '',
  GENERATION_PORT: '',
};

export function EnvTab() {
  const [values, setValues] = useState<EnvValues>(EMPTY_ENV);
  const [revealed, setRevealed] = useState<Set<string>>(new Set());
  const [status, setStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [errorMsg, setErrorMsg] = useState('');

  useEffect(() => {
    getEnv()
      .then((v) => setValues(v))
      .catch(() => {/* api not yet running */});
  }, []);

  function toggleReveal(key: string) {
    setRevealed((prev) => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  }

  async function handleSave() {
    setStatus('saving');
    setErrorMsg('');
    try {
      await saveEnv(values);
      setStatus('saved');
      setTimeout(() => setStatus('idle'), 2000);
    } catch (e: unknown) {
      setErrorMsg(String(e));
      setStatus('error');
    }
  }

  const keys = Object.keys(ENV_KEY_LABELS) as (keyof EnvValues)[];

  return (
    <div className="tab-content">
      <h2>Environment Variables</h2>
      <p className="hint">
        Values are applied to the running API process for this session only. Restart the
        API server to reload from <code>.env</code>.
      </p>

      <div className="env-grid">
        {keys.map((key) => {
          const isSensitive = SENSITIVE_KEYS.has(key);
          const isRevealed = revealed.has(key);
          return (
            <div key={key} className="env-row">
              <label htmlFor={key} className="env-label">
                {ENV_KEY_LABELS[key]}
                <span className="env-key-name">{key}</span>
              </label>
              <div className="env-input-wrap">
                <input
                  id={key}
                  type={isSensitive && !isRevealed ? 'password' : 'text'}
                  value={values[key]}
                  onChange={(e) => setValues((v) => ({ ...v, [key]: e.target.value }))}
                  placeholder={isSensitive ? '••••••••' : '(not set)'}
                  autoComplete="off"
                />
                {isSensitive && (
                  <button
                    className="reveal-btn"
                    type="button"
                    onClick={() => toggleReveal(key)}
                    title={isRevealed ? 'Hide' : 'Show'}
                  >
                    {isRevealed ? '🙈' : '👁'}
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <div className="env-actions">
        <button onClick={handleSave} disabled={status === 'saving'}>
          {status === 'saving' ? 'Saving…' : 'Save'}
        </button>
        {status === 'saved' && <span className="success-msg">✓ Saved</span>}
        {status === 'error' && <span className="error-msg">{errorMsg}</span>}
      </div>
    </div>
  );
}
