import { useEffect, useState } from 'react';
import './App.css';
import { getStatus } from './api';
import { EnvTab } from './components/EnvTab';
import { RunTab } from './components/RunTab';
import { SetupTab } from './components/SetupTab';
import type { AppStatus } from './types';

type Tab = 'setup' | 'env' | 'run';

const DEFAULT_STATUS: AppStatus = {
  config_loaded: false,
  run_id: null,
  is_running: false,
};

export default function App() {
  const [tab, setTab] = useState<Tab>('setup');
  const [status, setStatus] = useState<AppStatus>(DEFAULT_STATUS);

  // Poll /api/status every 2 s so Run tab buttons stay in sync
  useEffect(() => {
    const refresh = () =>
      getStatus()
        .then(setStatus)
        .catch(() => {/* api not yet reachable */});
    refresh();
    const id = setInterval(refresh, 2000);
    return () => clearInterval(id);
  }, []);

  function onConfigLoaded() {
    setStatus((s) => ({ ...s, config_loaded: true }));
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">RAG Evaluation</h1>
        <div className="status-pill">
          {status.config_loaded ? (
            <span className="pill loaded">
              Config loaded{status.run_id ? ` · ${status.run_id}` : ''}
            </span>
          ) : (
            <span className="pill empty">No config</span>
          )}
          {status.is_running && <span className="pill running">Running…</span>}
        </div>
      </header>

      <nav className="tab-nav">
        {(['setup', 'env', 'run'] as Tab[]).map((t, i) => (
          <button
            key={t}
            className={`tab-btn ${tab === t ? 'active' : ''}`}
            onClick={() => setTab(t)}
          >
            {i + 1}.{' '}
            {t === 'setup' ? 'Setup' : t === 'env' ? 'Environment' : 'Run'}
          </button>
        ))}
      </nav>

      <main className="tab-panel">
        {tab === 'setup' && <SetupTab onConfigLoaded={onConfigLoaded} />}
        {tab === 'env' && <EnvTab />}
        {tab === 'run' && <RunTab status={status} />}
      </main>
    </div>
  );
}
