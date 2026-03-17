import { useEffect, useRef, useState } from 'react';
import { checkWorker, getWorkerUrl, loadConfigFromContent, loadConfigFromPath, setWorkerUrl } from '../api';
import type { WorkerCheckResponse } from '../types';

interface Props {
  onConfigLoaded: () => void;
  configLoaded: boolean;
  onNext: () => void;
}

export function SetupTab({ onConfigLoaded, configLoaded, onNext }: Props) {
  const [pathInput, setPathInput] = useState('');
  const [yamlText, setYamlText] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const [workerUrl, setWorkerUrlState] = useState('');
  const committedUrlRef = useRef('');
  const [workerCheck, setWorkerCheck] = useState<WorkerCheckResponse | null>(null);
  const [checking, setChecking] = useState(false);

  useEffect(() => {
    getWorkerUrl().then((url) => {
      setWorkerUrlState(url);
      committedUrlRef.current = url;
    }).catch(() => {});
  }, []);

  async function commitWorkerUrl(url: string) {
    const trimmed = url.trim();
    if (!trimmed || trimmed === committedUrlRef.current) return;
    try {
      const saved = await setWorkerUrl(trimmed);
      committedUrlRef.current = saved;
      setWorkerUrlState(saved);
      setWorkerCheck(null);
    } catch {
      setWorkerUrlState(committedUrlRef.current);
    }
  }

  async function handleCheckWorker() {
    setChecking(true);
    setWorkerCheck(null);
    try {
      const result = await checkWorker();
      setWorkerCheck(result);
    } catch (e: unknown) {
      setWorkerCheck({ ok: false, status: null, backend: null, models_loaded: null, error: String(e) });
    } finally {
      setChecking(false);
    }
  }

  async function handleLoadPath() {
    if (!pathInput.trim()) return;
    setLoading(true);
    setError('');
    try {
      const res = await loadConfigFromPath(pathInput.trim());
      if (res.ok) {
        setYamlText(res.yaml_text);
        onConfigLoaded();
      } else {
        setError(res.error ?? 'Unknown error');
      }
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async (evt) => {
      const content = evt.target?.result as string;
      setLoading(true);
      setError('');
      try {
        const res = await loadConfigFromContent(content);
        if (res.ok) {
          setYamlText(res.yaml_text);
          onConfigLoaded();
        } else {
          setError(res.error ?? 'Unknown error');
        }
      } catch (ex: unknown) {
        setError(String(ex));
      } finally {
        setLoading(false);
        if (fileRef.current) fileRef.current.value = '';
      }
    };
    reader.readAsText(file);
  }

  const workerOk = workerCheck?.ok === true;
  const canProceed = configLoaded && workerOk;

  const navHint = !configLoaded && !workerOk
    ? 'Load a config and check the worker to continue'
    : !configLoaded
      ? 'Load a config file to continue'
      : 'Check worker connectivity to continue';

  return (
    <div className="tab-content">

      {/* ── a. Configuration file ──────────────────────────────────────── */}
      <h2 className="setup-section-heading">
        <span className="setup-section-letter">a.</span> Load config
        {configLoaded && <span className="section-status-ok">✓</span>}
      </h2>

      <div className="input-row">
        <input
          type="text"
          className="path-input"
          placeholder="Path to config YAML (e.g. config/sample_config.yaml)"
          value={pathInput}
          onChange={(e) => setPathInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleLoadPath()}
        />
        <button onClick={handleLoadPath} disabled={loading || !pathInput.trim()}>
          Load
        </button>
        <span className="divider">or</span>
        <button onClick={() => fileRef.current?.click()} disabled={loading}>
          Browse…
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".yaml,.yml"
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />
      </div>

      {error && <p className="error-msg">{error}</p>}

      {yamlText && (
        <div className="yaml-display">
          <div className="yaml-header">
            <span className="loaded-badge">✓ Config loaded</span>
          </div>
          <pre className="yaml-pre"><code>{yamlText}</code></pre>
        </div>
      )}

      <div className="section-divider" />

      {/* ── b. Worker connection ───────────────────────────────────────── */}
      <h2 className="setup-section-heading">
        <span className="setup-section-letter">b.</span> Check worker connection
        {workerOk && <span className="section-status-ok">✓</span>}
      </h2>

      <div className="worker-check-row">
        <span className="worker-url-label">Worker URL</span>
        <input
          className="worker-url-input mono"
          type="text"
          value={workerUrl}
          onChange={(e) => { setWorkerUrlState(e.target.value); setWorkerCheck(null); }}
          onBlur={(e) => commitWorkerUrl(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && commitWorkerUrl(workerUrl)}
          placeholder="http://localhost:8000"
        />
        <button
          className="btn-check"
          onClick={handleCheckWorker}
          disabled={checking || !workerUrl}
        >
          {checking ? 'Checking…' : 'Check'}
        </button>
      </div>

      {workerCheck && (
        workerCheck.ok ? (
          <div className="worker-check-ok">
            <span className="check-icon">✓</span>
            <span>Worker reachable</span>
            {workerCheck.backend && <span className="meta-dim">· {workerCheck.backend}</span>}
            {workerCheck.models_loaded != null && (
              <span className="meta-dim">
                · models {workerCheck.models_loaded ? 'loaded' : 'not loaded'}
              </span>
            )}
          </div>
        ) : (
          <div className="worker-check-err">
            <span className="check-icon">✗</span>
            <span>{workerCheck.error ?? 'Unknown error'}</span>
          </div>
        )
      )}

      {/* ── Navigation ─────────────────────────────────────────────────── */}
      <div className="tab-nav-actions">
        <button
          className={canProceed ? 'btn-next btn-next-ready' : 'btn-next'}
          onClick={canProceed ? onNext : undefined}
        >
          Next: Run →
        </button>
        {!canProceed && <span className="nav-hint">{navHint}</span>}
      </div>
    </div>
  );
}
