import { useRef, useState } from 'react';
import { loadConfigFromContent, loadConfigFromPath } from '../api';

interface Props {
  onConfigLoaded: () => void;
}

export function SetupTab({ onConfigLoaded }: Props) {
  const [pathInput, setPathInput] = useState('');
  const [yamlText, setYamlText] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

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
        // reset input so the same file can be re-loaded
        if (fileRef.current) fileRef.current.value = '';
      }
    };
    reader.readAsText(file);
  }

  return (
    <div className="tab-content">
      <h2>Configuration</h2>

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
    </div>
  );
}
