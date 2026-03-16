import { useEffect, useRef, useState } from 'react';
import { startRun } from '../api';
import type { AppStatus, SSEMessage } from '../types';

interface Props {
  status: AppStatus;
}

export function RunTab({ status }: Props) {
  const [lines, setLines] = useState<string[]>([]);
  const [verbose, setVerbose] = useState(true);
  const [showOutput, setShowOutput] = useState(true);
  const [running, setRunning] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const terminalRef = useRef<HTMLDivElement>(null);

  // Auto-scroll terminal to bottom on new output
  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [lines]);

  // Sync running state with global status (e.g. after page reload)
  useEffect(() => {
    setRunning(status.is_running);
  }, [status.is_running]);

  function handleMessage(msg: SSEMessage) {
    if (msg.type === 'log') {
      setLines((l) => [...l, msg.text]);
    } else if (msg.type === 'error') {
      setLines((l) => [...l, `ERROR: ${msg.text}`]);
    }
  }

  function handleDone() {
    setRunning(false);
  }

  function handleError(err: string) {
    setLines((l) => [...l, `ERROR: ${err}`]);
  }

  function handleRun(dryRun: boolean) {
    setLines([]);
    setRunning(true);
    abortRef.current = startRun(dryRun, verbose, handleMessage, handleDone, handleError);
  }

  function handleStop() {
    abortRef.current?.abort();
    setRunning(false);
    setLines((l) => [...l, '— run cancelled —']);
  }

  const canRun = status.config_loaded && !running;
  const disabledReason = !status.config_loaded
    ? 'Load a config in the Setup tab first.'
    : running
    ? 'Evaluation in progress…'
    : '';

  return (
    <div className="tab-content run-tab">
      <h2>Run Evaluation</h2>

      <div className="run-toggles">
        <label className="toggle-label">
          <input
            type="checkbox"
            checked={showOutput}
            onChange={(e) => setShowOutput(e.target.checked)}
          />
          Show Output
        </label>
        <label className="toggle-label">
          <input
            type="checkbox"
            checked={verbose}
            onChange={(e) => setVerbose(e.target.checked)}
          />
          Verbose
        </label>
      </div>

      <div className="run-buttons">
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
          <button className="btn-stop" onClick={handleStop}>
            ■ Stop
          </button>
        )}
      </div>

      {disabledReason && !running && (
        <p className="disabled-reason">{disabledReason}</p>
      )}

      {showOutput && (
        <div ref={terminalRef} className="terminal">
          {lines.length === 0 ? (
            <span className="terminal-placeholder">Output will appear here…</span>
          ) : (
            lines.map((line, i) => (
              <div
                key={i}
                className={line.startsWith('ERROR') ? 'terminal-line error' : 'terminal-line'}
              >
                {line}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
