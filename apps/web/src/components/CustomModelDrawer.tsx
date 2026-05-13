import { useState } from 'react';
import type { AttentionLayer, ModelConfig } from '@llm-calc/core';

interface Props {
  onClose: () => void;
  onSave: (m: ModelConfig) => void;
}

interface LayerRow {
  id: number;
  kind: 'full' | 'linear' | 'none';
  n_kv_heads: number;
  head_dim: number;
  state_size_bytes: number;
}

let nextId = 1;
const blankRow = (kind: LayerRow['kind'] = 'full'): LayerRow => ({
  id: nextId++,
  kind,
  n_kv_heads: 8,
  head_dim: 128,
  state_size_bytes: 1_310_720,
});

export function CustomModelDrawer({ onClose, onSave }: Props) {
  const [name, setName] = useState('custom-model');
  const [params, setParams] = useState(7);
  const [activeParams, setActiveParams] = useState<number | ''>('');
  const [hidden, setHidden] = useState(4096);
  const [vocab, setVocab] = useState(32000);
  const [rows, setRows] = useState<LayerRow[]>(
    Array.from({ length: 32 }, () => blankRow('full')),
  );

  const updateRow = (id: number, patch: Partial<LayerRow>) =>
    setRows((r) => r.map((row) => (row.id === id ? { ...row, ...patch } : row)));

  const save = () => {
    const layers: AttentionLayer[] = rows.map((r) => {
      if (r.kind === 'full') return { kind: 'full', n_kv_heads: r.n_kv_heads, head_dim: r.head_dim };
      if (r.kind === 'linear') return { kind: 'linear', state_size_bytes: r.state_size_bytes };
      return { kind: 'none' };
    });
    const model: ModelConfig = {
      name,
      params: params * 1e9,
      ...(typeof activeParams === 'number' && activeParams > 0
        ? { active_params: activeParams * 1e9 }
        : {}),
      hidden_dim: hidden,
      vocab_size: vocab,
      layers,
      architecture: 'custom',
    };
    onSave(model);
  };

  return (
    <div className="fixed inset-y-0 right-0 z-50 flex w-full max-w-2xl flex-col gap-4 overflow-auto bg-ink-900 p-5 shadow-xl">
      <div className="flex items-center justify-between">
        <h3 className="font-medium">Custom model</h3>
        <button className="btn" onClick={onClose}>
          ✕
        </button>
      </div>
      <div className="grid grid-cols-2 gap-3 text-sm">
        <label className="flex flex-col gap-1">
          <span className="label">Name</span>
          <input
            className="rounded border border-ink-700 bg-ink-800 px-2 py-1"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="label">Params (B)</span>
          <input
            type="number"
            min={0.1}
            step={0.1}
            className="rounded border border-ink-700 bg-ink-800 px-2 py-1"
            value={params}
            onChange={(e) => setParams(Number(e.target.value))}
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="label">Active params (B, MoE)</span>
          <input
            type="number"
            min={0}
            step={0.1}
            className="rounded border border-ink-700 bg-ink-800 px-2 py-1"
            value={activeParams}
            onChange={(e) =>
              setActiveParams(e.target.value === '' ? '' : Number(e.target.value))
            }
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="label">Hidden dim</span>
          <input
            type="number"
            className="rounded border border-ink-700 bg-ink-800 px-2 py-1"
            value={hidden}
            onChange={(e) => setHidden(Number(e.target.value))}
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="label">Vocab size</span>
          <input
            type="number"
            className="rounded border border-ink-700 bg-ink-800 px-2 py-1"
            value={vocab}
            onChange={(e) => setVocab(Number(e.target.value))}
          />
        </label>
      </div>

      <div>
        <div className="mb-2 flex items-center justify-between">
          <h4 className="label">Layers ({rows.length})</h4>
          <div className="flex gap-2">
            <button className="btn" onClick={() => setRows((r) => [...r, blankRow('full')])}>
              + Full
            </button>
            <button className="btn" onClick={() => setRows((r) => [...r, blankRow('linear')])}>
              + Linear
            </button>
            <button className="btn" onClick={() => setRows([])}>
              Clear
            </button>
          </div>
        </div>
        <div className="max-h-96 overflow-auto rounded border border-ink-700">
          <table className="w-full text-xs">
            <thead className="bg-ink-800 text-ink-400">
              <tr>
                <th className="px-2 py-1 text-left">#</th>
                <th className="px-2 py-1 text-left">Kind</th>
                <th className="px-2 py-1 text-left">KV heads</th>
                <th className="px-2 py-1 text-left">Head dim</th>
                <th className="px-2 py-1 text-left">State bytes</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={row.id} className="border-t border-ink-700">
                  <td className="px-2 py-1 text-ink-500">{i}</td>
                  <td className="px-2 py-1">
                    <select
                      className="rounded bg-ink-800 px-1 py-0.5"
                      value={row.kind}
                      onChange={(e) => updateRow(row.id, { kind: e.target.value as LayerRow['kind'] })}
                    >
                      <option value="full">full</option>
                      <option value="linear">linear</option>
                      <option value="none">none</option>
                    </select>
                  </td>
                  <td className="px-2 py-1">
                    <input
                      type="number"
                      disabled={row.kind !== 'full'}
                      className="w-16 rounded bg-ink-800 px-1 py-0.5 disabled:opacity-40"
                      value={row.n_kv_heads}
                      onChange={(e) => updateRow(row.id, { n_kv_heads: Number(e.target.value) })}
                    />
                  </td>
                  <td className="px-2 py-1">
                    <input
                      type="number"
                      disabled={row.kind !== 'full'}
                      className="w-16 rounded bg-ink-800 px-1 py-0.5 disabled:opacity-40"
                      value={row.head_dim}
                      onChange={(e) => updateRow(row.id, { head_dim: Number(e.target.value) })}
                    />
                  </td>
                  <td className="px-2 py-1">
                    <input
                      type="number"
                      disabled={row.kind !== 'linear'}
                      className="w-24 rounded bg-ink-800 px-1 py-0.5 disabled:opacity-40"
                      value={row.state_size_bytes}
                      onChange={(e) =>
                        updateRow(row.id, { state_size_bytes: Number(e.target.value) })
                      }
                    />
                  </td>
                  <td className="px-2 py-1">
                    <button
                      className="text-ink-400 hover:text-danger-400"
                      onClick={() => setRows((r) => r.filter((x) => x.id !== row.id))}
                    >
                      ✕
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="flex justify-end gap-2">
        <button className="btn" onClick={onClose}>
          Cancel
        </button>
        <button className="btn-primary" onClick={save}>
          Save model
        </button>
      </div>
    </div>
  );
}
