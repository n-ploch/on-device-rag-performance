import type { AppStatus, ConfigLoadResponse, RunEvent } from './types';

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(data.detail ?? res.statusText);
  }
  return res.json();
}

export async function loadConfigFromPath(path: string): Promise<ConfigLoadResponse> {
  return post('/api/config/load', { path });
}

export async function loadConfigFromContent(content: string): Promise<ConfigLoadResponse> {
  return post('/api/config/load', { content });
}


export async function getStatus(): Promise<AppStatus> {
  const res = await fetch('/api/status');
  return res.json();
}

export async function stopRun(): Promise<void> {
  await post('/api/stop', {});
}

/**
 * Start a run (or dry run) and stream structured progress events back.
 *
 * Uses fetch + ReadableStream because EventSource only supports GET requests.
 * Returns an AbortController the caller can use to cancel the stream.
 */
export function startRun(
  dryRun: boolean,
  onEvent: (event: RunEvent) => void,
  onDone: () => void,
  onFatalError: (msg: string) => void,
): AbortController {
  const controller = new AbortController();
  const endpoint = dryRun ? '/api/dry-run' : '/api/run';

  (async () => {
    let res: Response;
    try {
      res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
        signal: controller.signal,
      });
    } catch (err: unknown) {
      if ((err as Error).name !== 'AbortError') {
        onFatalError(String(err));
        onDone();
      }
      return;
    }

    if (!res.ok) {
      const data = await res.json().catch(() => ({ detail: res.statusText }));
      onFatalError(data.detail ?? res.statusText);
      onDone();
      return;
    }

    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const event: RunEvent = JSON.parse(line.slice(6));
            onEvent(event);
            if (event.type === 'done') {
              onDone();
              return;
            }
          } catch {
            // malformed JSON line — ignore
          }
        }
      }
    } catch (err: unknown) {
      if ((err as Error).name !== 'AbortError') {
        onFatalError(String(err));
      }
    } finally {
      onDone();
    }
  })();

  return controller;
}
