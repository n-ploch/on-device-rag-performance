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

interface Props {
  onBack: () => void;
  onNext: () => void;
}

export function EnvTab({ onBack, onNext }: Props) {
  const [values, setValues] = useState<EnvValues>(EMPTY_ENV);
  const [revealed, setRevealed] = useState<Set<string>>(new Set());
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [errorMsg, setErrorMsg] = useState('');

  useEffect(() => {
    getEnv()
      .then((v) => setValues(v))
      .catch(() => {});
  }, []);

  const allSet = Object.values(values).every((v) => v.trim() !== '');

  function toggleReveal(key: string) {
    setRevealed((prev) => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  }

  async function handleSave() {
    setSaveStatus('saving');
    setErrorMsg('');
    try {
      await saveEnv(values);
      setSaveStatus('saved');
      setTimeout(() => setSaveStatus('idle'), 2000);
    } catch (e: unknown) {
      setErrorMsg(String(e));
      setSaveStatus('error');
    }
  }

  const keys = Object.keys(ENV_KEY_LABELS) as (keyof EnvValues)[];

  return (
    <div className="tab-content">
      <div className="tab-top-nav">
        <button className="btn-back" onClick={onBack}>← Back</button>
      </div>

      <h2>Environment Variables</h2>
      <p className="hint">
        Values are applied to the running API process for this session only. Restart the
        API server to reload from <code>.env</code>.
      </p>

      <div className="env-grid">
        {keys.map((key) => {
          const isSensitive = SENSITIVE_KEYS.has(key);
          const isRevealed = revealed.has(key);
          const isEmpty = !values[key].trim();
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
                  className={isEmpty ? 'input-empty' : ''}
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
        <button onClick={handleSave} disabled={saveStatus === 'saving'}>
          {saveStatus === 'saving' ? 'Saving…' : 'Save'}
        </button>
        {saveStatus === 'saved' && <span className="success-msg">✓ Saved</span>}
        {saveStatus === 'error' && <span className="error-msg">{errorMsg}</span>}
      </div>

      <div className="tab-nav-actions">
        <button
          className={allSet ? 'btn-next btn-next-ready' : 'btn-next'}
          onClick={onNext}
        >
          Next: Run →
        </button>
        {!allSet && (
          <span className="nav-hint">Fill in all environment variables to continue</span>
        )}
      </div>
    </div>
  );
}
