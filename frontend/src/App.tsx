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

const TAB_LABELS: Record<Tab, string> = {
  setup: 'Load Config',
  env: 'Environment',
  run: 'Run',
};

export default function App() {
  const [tab, setTab] = useState<Tab>('setup');
  const [status, setStatus] = useState<AppStatus>(DEFAULT_STATUS);

  useEffect(() => {
    const refresh = () =>
      getStatus()
        .then(setStatus)
        .catch(() => {});
    refresh();
    const id = setInterval(refresh, 2000);
    return () => clearInterval(id);
  }, []);

  function onConfigLoaded() {
    setStatus((s) => ({ ...s, config_loaded: true }));
  }

  const tabs: Tab[] = ['setup', 'env', 'run'];

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
        {tabs.map((t, i) => (
          <button
            key={t}
            className={`tab-btn ${tab === t ? 'active' : ''}`}
            onClick={() => setTab(t)}
          >
            {i + 1}. {TAB_LABELS[t]}
          </button>
        ))}
      </nav>

      <main className="tab-panel">
        {tab === 'setup' && (
          <SetupTab
            onConfigLoaded={onConfigLoaded}
            configLoaded={status.config_loaded}
            onNext={() => setTab('env')}
          />
        )}
        {tab === 'env' && (
          <EnvTab
            onBack={() => setTab('setup')}
            onNext={() => setTab('run')}
          />
        )}
        {tab === 'run' && (
          <RunTab
            status={status}
            onBack={() => setTab('env')}
          />
        )}
      </main>
    </div>
  );
}
