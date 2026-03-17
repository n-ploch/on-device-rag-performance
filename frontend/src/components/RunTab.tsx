import { useEffect, useRef, useState } from 'react';
import { startRun, stopRun } from '../api';
import type {
  AppStatus,
  DryRunResultEvent,
  EntryErrorEvent,
  EntryResultEvent,
  RunCompleteEvent,
  RunEvent,
  RunStartEvent,
  StoppedEvent,
} from '../types';

interface Props {
  status: AppStatus;
  onBack: () => void;
}

interface RunState {
  runStart: RunStartEvent | null;
  lastResult: EntryResultEvent | null;
  summary: RunCompleteEvent | null;
  stopped: StoppedEvent | null;
  dryRun: DryRunResultEvent | null;
  entryErrors: EntryErrorEvent[];
  fatalError: string | null;
}

const EMPTY_STATE: RunState = {
  runStart: null,
  lastResult: null,
  summary: null,
  stopped: null,
  dryRun: null,
  entryErrors: [],
  fatalError: null,
};

function fmt(n: number, decimals = 3) {
  return n.toFixed(decimals);
}

function truncate(s: string, max = 300) {
  return s.length > max ? s.slice(0, max) + '…' : s;
}

// ── Sub-components ────────────────────────────────────────────────────────────

function ProgressBar({ value, max }: { value: number; max: number }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="progress-bar-wrap">
      <div className="progress-bar-track">
        <div className="progress-bar-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="progress-label">
        {value} / {max} entries ({pct}%)
      </span>
    </div>
  );
}

function RunHeader({ info }: { info: RunStartEvent }) {
  return (
    <div className="progress-card">
      <div className="progress-card-title">
        <span className="run-id-badge">{info.run_id}</span>
        {info.total_configs > 1 && (
          <span className="dim">
            Config {info.config_index}/{info.total_configs}
          </span>
        )}
        {info.total_reps > 1 && (
          <span className="dim">
            Rep {info.rep}/{info.total_reps}
          </span>
        )}
      </div>
      <div className="meta-row">
        <span className="meta-item">
          <span className="meta-label">Retrieval</span>
          <span className="meta-value">
            {info.retrieval_model.split('/').pop()}
            <span className="quant-badge">{info.retrieval_quantization}</span>
            · top-{info.k}
          </span>
        </span>
        <span className="meta-item">
          <span className="meta-label">Generation</span>
          <span className="meta-value">
            {info.generation_model.split('/').pop()}
            <span className="quant-badge">{info.generation_quantization}</span>
          </span>
        </span>
        <span className="meta-item">
          <span className="meta-label">Session ID</span>
          <span className="meta-value mono">{info.session_id}</span>
        </span>
      </div>
    </div>
  );
}

function LastResultCard({ result }: { result: EntryResultEvent }) {
  const r = result.response;
  return (
    <div className="result-card">
      <div className="result-card-header">
        <span className="result-entry-badge">Entry #{result.entry_index}</span>
        <span className="claim-id mono">{result.request.claim_id}</span>
        {r.is_abstention && <span className="abstention-badge">ABSTAINED</span>}
      </div>

      <div className="result-section">
        <div className="result-label">Query</div>
        <div className="result-text query-text">{truncate(result.request.input, 240)}</div>
      </div>

      <div className="result-section">
        <div className="result-label">Response</div>
        <div className="result-text response-text">{truncate(r.output, 420)}</div>
      </div>

      <div className="metrics-row">
        {r.recall_at_k !== null && (
          <>
            <div className="metric-chip">
              <span className="metric-name">Recall@k</span>
              <span className="metric-value">{fmt(r.recall_at_k)}</span>
            </div>
            <div className="metric-chip">
              <span className="metric-name">Precision@k</span>
              <span className="metric-value">{fmt(r.precision_at_k!)}</span>
            </div>
            <div className="metric-chip">
              <span className="metric-name">MRR</span>
              <span className="metric-value">{fmt(r.mrr!)}</span>
            </div>
          </>
        )}
        <div className="metric-chip perf">
          <span className="metric-name">Latency</span>
          <span className="metric-value">{r.latency_ms.toFixed(0)} ms</span>
        </div>
        <div className="metric-chip perf">
          <span className="metric-name">Speed</span>
          <span className="metric-value">{r.tokens_per_second.toFixed(1)} tok/s</span>
        </div>
        <div className="metric-chip perf">
          <span className="metric-name">RAM</span>
          <span className="metric-value">{r.ram_mb.toFixed(0)} MB</span>
        </div>
      </div>
    </div>
  );
}

function SummaryCard({ summary }: { summary: RunCompleteEvent }) {
  return (
    <div className="summary-card">
      <div className="summary-title">Run complete — {summary.run_id}</div>
      <div className="metrics-row">
        <div className="metric-chip summary-chip">
          <span className="metric-name">Avg Recall@k</span>
          <span className="metric-value">{fmt(summary.avg_recall)}</span>
        </div>
        <div className="metric-chip summary-chip">
          <span className="metric-name">Avg Precision@k</span>
          <span className="metric-value">{fmt(summary.avg_precision)}</span>
        </div>
        <div className="metric-chip summary-chip">
          <span className="metric-name">Avg MRR</span>
          <span className="metric-value">{fmt(summary.avg_mrr)}</span>
        </div>
      </div>
    </div>
  );
}

function DryRunCard({ result }: { result: DryRunResultEvent }) {
  return (
    <div className="dry-run-card">
      <div className="dry-run-title">Dry run validated</div>
      <p>
        Ready to evaluate <strong>{result.total_configs}</strong> config{result.total_configs !== 1 ? 's' : ''}.
      </p>
      <div className="dry-run-ids">
        {result.run_ids.map((id) => (
          <span key={id} className="run-id-badge">{id}</span>
        ))}
      </div>
    </div>
  );
}

function StoppedCard({ ev }: { ev: StoppedEvent }) {
  return (
    <div className="stopped-card">
      <div className="stopped-title">
        Stopped — {ev.run_id}
      </div>
      <p className="stopped-subtitle">
        Completed {ev.completed_entries} of {ev.total_entries} entries before stopping.
      </p>
      {ev.total > 0 && (
        <div className="metrics-row">
          <div className="metric-chip summary-chip">
            <span className="metric-name">Avg Recall@k</span>
            <span className="metric-value">{fmt(ev.avg_recall)}</span>
          </div>
          <div className="metric-chip summary-chip">
            <span className="metric-name">Avg Precision@k</span>
            <span className="metric-value">{fmt(ev.avg_precision)}</span>
          </div>
          <div className="metric-chip summary-chip">
            <span className="metric-name">Avg MRR</span>
            <span className="metric-value">{fmt(ev.avg_mrr)}</span>
          </div>
        </div>
      )}
    </div>
  );
}

function ErrorBox({ title, message }: { title: string; message: string }) {
  return (
    <div className="error-box">
      <span className="error-box-icon">⚠</span>
      <div>
        <div className="error-box-title">{title}</div>
        <div className="error-box-msg">{message}</div>
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export function RunTab({ status, onBack }: Props) {
  const [runState, setRunState] = useState<RunState>(EMPTY_STATE);
  const [running, setRunning] = useState(false);
  const [stopping, setStopping] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    setRunning(status.is_running);
  }, [status.is_running]);

  function handleEvent(event: RunEvent) {
    switch (event.type) {
      case 'run_start':
        setRunState((s) => ({
          ...s,
          runStart: event,
          lastResult: null,
          summary: null,
          stopped: null,
          dryRun: null,
          fatalError: null,
        }));
        break;
      case 'entry_result':
        setRunState((s) => ({ ...s, lastResult: event }));
        break;
      case 'entry_error':
        setRunState((s) => ({ ...s, entryErrors: [...s.entryErrors, event] }));
        break;
      case 'run_complete':
        setRunState((s) => ({ ...s, summary: event }));
        break;
      case 'stopped':
        setRunState((s) => ({ ...s, stopped: event }));
        break;
      case 'dry_run_result':
        setRunState((s) => ({ ...s, dryRun: event }));
        break;
      case 'error':
        setRunState((s) => ({ ...s, fatalError: event.message }));
        break;
      case 'done':
        break;
    }
  }

  function handleRun(dryRun: boolean) {
    setRunState({ ...EMPTY_STATE });
    setRunning(true);
    setStopping(false);
    abortRef.current = startRun(
      dryRun,
      handleEvent,
      () => { setRunning(false); setStopping(false); },
      (msg) => setRunState((s) => ({ ...s, fatalError: msg })),
    );
  }

  async function handleStop() {
    setStopping(true);
    try {
      await stopRun();
    } catch {
      // If the call fails the stream will still close eventually
      setStopping(false);
    }
  }

  const canRun = status.config_loaded && !running;
  const disabledReason = !status.config_loaded
    ? 'Load a config in the Setup tab first.'
    : '';

  const { runStart, lastResult, summary, dryRun, entryErrors, fatalError } = runState;
  const hasAnyOutput = runStart || dryRun || fatalError;

  return (
    <div className="tab-content">
      <div className="tab-top-nav">
        <button className="btn-back" onClick={onBack}>← Back</button>
      </div>

      <h2>Run Evaluation</h2>

      {/* Controls */}
      <div className="run-controls">
        <button
          className="btn-primary"
          onClick={() => handleRun(false)}
          disabled={!canRun}
          title={disabledReason}
        >
          ▶ Run
        </button>
        <button
          className="btn-secondary"
          onClick={() => handleRun(true)}
          disabled={!canRun}
          title={disabledReason}
        >
          Dry Run
        </button>
        {running && (
          <button className="btn-stop" onClick={handleStop} disabled={stopping}>
            {stopping ? '⏳ Stopping…' : '■ Stop'}
          </button>
        )}
      </div>

      {disabledReason && (
        <p className="disabled-reason">{disabledReason}</p>
      )}

      {/* Progress section */}
      {runStart && (
        <div className="progress-section">
          <RunHeader info={runStart} />
          <ProgressBar
            value={lastResult?.entry_index ?? 0}
            max={runStart.total_entries}
          />
        </div>
      )}

      {/* Last result */}
      {lastResult && <LastResultCard result={lastResult} />}

      {/* Summary or stopped partial summary */}
      {summary && <SummaryCard summary={summary} />}
      {runState.stopped && <StoppedCard ev={runState.stopped} />}

      {/* Dry run result */}
      {dryRun && <DryRunCard result={dryRun} />}

      {/* Entry errors */}
      {entryErrors.map((e) => (
        <ErrorBox
          key={`${e.entry_index}-${e.claim_id}`}
          title={`Entry #${e.entry_index} failed — ${e.claim_id}`}
          message={e.message}
        />
      ))}

      {/* Fatal error */}
      {fatalError && <ErrorBox title="Error" message={fatalError} />}

      {/* Idle placeholder */}
      {!hasAnyOutput && !running && (
        <div className="run-idle">
          <p>Press <strong>Run</strong> to start an evaluation or <strong>Dry Run</strong> to validate the config.</p>
        </div>
      )}
    </div>
  );
}
