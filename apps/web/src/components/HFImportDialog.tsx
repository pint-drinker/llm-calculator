import { useState } from 'react';
import type { ModelConfig } from '@llm-calc/core';
import { importHF } from '../api.js';

interface Props {
  onClose: () => void;
  onImported: (m: ModelConfig) => void;
}

export function HFImportDialog({ onClose, onImported }: Props) {
  const [modelId, setModelId] = useState('');
  const [token, setToken] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<{
    model: ModelConfig;
    warnings: string[];
    adapter: string;
  } | null>(null);

  const onFetch = async () => {
    setLoading(true);
    setError(null);
    setPreview(null);
    try {
      const result = await importHF(modelId, { token: token || undefined });
      setPreview(result);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink-950/70 p-6">
      <div className="panel w-full max-w-lg">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="font-medium">Import from HuggingFace Hub</h3>
          <button className="btn" onClick={onClose}>
            ✕
          </button>
        </div>
        <label className="label">Model ID</label>
        <input
          autoFocus
          className="mt-1 w-full rounded border border-ink-700 bg-ink-800 px-2 py-1.5 text-sm"
          placeholder="e.g. meta-llama/Llama-3.1-8B"
          value={modelId}
          onChange={(e) => setModelId(e.target.value)}
        />
        <label className="label mt-3 block">Token (only for gated models)</label>
        <input
          type="password"
          className="mt-1 w-full rounded border border-ink-700 bg-ink-800 px-2 py-1.5 text-sm"
          placeholder="hf_…"
          value={token}
          onChange={(e) => setToken(e.target.value)}
        />
        <div className="mt-4 flex justify-end gap-2">
          <button className="btn" onClick={onClose}>
            Cancel
          </button>
          <button
            className="btn-primary disabled:opacity-50"
            disabled={!modelId || loading}
            onClick={onFetch}
          >
            {loading ? 'Fetching…' : 'Fetch'}
          </button>
        </div>
        {error && (
          <p className="mt-3 rounded border border-danger-500/40 bg-danger-500/10 p-2 text-xs text-danger-400">
            {error}
          </p>
        )}
        {preview && (
          <div className="mt-4 rounded border border-ink-700 bg-ink-800 p-3 text-xs">
            <div className="mb-1 font-medium">Preview — adapter: {preview.adapter}</div>
            <dl className="grid grid-cols-2 gap-y-1">
              <dt className="text-ink-400">Name</dt>
              <dd className="font-mono">{preview.model.name}</dd>
              <dt className="text-ink-400">Params</dt>
              <dd className="font-mono">{(preview.model.params / 1e9).toFixed(2)}B</dd>
              {preview.model.active_params && (
                <>
                  <dt className="text-ink-400">Active</dt>
                  <dd className="font-mono">{(preview.model.active_params / 1e9).toFixed(2)}B</dd>
                </>
              )}
              <dt className="text-ink-400">Layers</dt>
              <dd className="font-mono">
                {preview.model.layers.filter((l) => l.kind === 'full').length} full ·{' '}
                {preview.model.layers.filter((l) => l.kind === 'linear').length} linear
              </dd>
              <dt className="text-ink-400">Hidden</dt>
              <dd className="font-mono">{preview.model.hidden_dim}</dd>
            </dl>
            {preview.warnings.map((w) => (
              <p key={w} className="mt-2 text-xs text-yellow-400">
                ⚠ {w}
              </p>
            ))}
            <div className="mt-3 flex justify-end">
              <button className="btn-primary" onClick={() => onImported(preview.model)}>
                Use this model
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
