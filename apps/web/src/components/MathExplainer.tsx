import { useMemo, useState } from 'react';
import { explain, type GPU, type InferenceConfig } from '@llm-calc/core';
import { useStore } from '../store.js';
import { fmtNumber } from '../format.js';

interface Props {
  config: InferenceConfig;
  gpu: GPU;
}

const HIGHLIGHT_MAP: Record<string, string> = {
  weights_bytes: 'weights',
  kv_cache_bytes: 'kv',
  kv_bytes_per_token: 'kv',
  linear_state_bytes: 'linear',
  activations_bytes: 'activations',
  framework_overhead_bytes: 'overhead',
};

export function MathExplainer({ config, gpu }: Props) {
  const trace = useMemo(() => explain(config, gpu), [config, gpu]);
  const highlighted = useStore((s) => s.highlightedValue);
  const setHighlighted = useStore((s) => s.setHighlighted);
  const [open, setOpen] = useState(true);

  return (
    <section className="panel">
      <button
        className="flex w-full items-center justify-between"
        onClick={() => setOpen((o) => !o)}
      >
        <h3 className="text-sm font-medium">Math explainer ({trace.steps.length} steps)</h3>
        <span className="text-ink-400">{open ? '▾' : '▸'}</span>
      </button>
      {open && (
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-ink-400">
              <tr className="border-b border-ink-700">
                <th className="py-1 pr-3 text-left">Step</th>
                <th className="py-1 pr-3 text-left">Formula</th>
                <th className="py-1 pr-3 text-left">Substituted</th>
                <th className="py-1 pr-3 text-right">Result</th>
                <th className="py-1 pr-3 text-left">Units</th>
              </tr>
            </thead>
            <tbody>
              {trace.steps.map((s) => {
                const grp = HIGHLIGHT_MAP[s.name];
                const isOn = grp && highlighted === grp;
                return (
                  <tr
                    key={s.name}
                    className={'border-b border-ink-800 ' + (isOn ? 'bg-ink-800' : '')}
                  >
                    <td className="py-1 pr-3 font-mono text-ink-200">{s.name}</td>
                    <td className="py-1 pr-3 formula-cell text-ink-300">{s.formula}</td>
                    <td className="py-1 pr-3 formula-cell text-ink-300">{s.substituted}</td>
                    <td className="py-1 pr-3 text-right">
                      <button
                        className="font-mono hover:underline"
                        onMouseEnter={() => grp && setHighlighted(grp)}
                        onMouseLeave={() => setHighlighted(null)}
                        onClick={() => setHighlighted(isOn ? null : grp ?? null)}
                      >
                        {fmtNumber(s.result)}
                      </button>
                    </td>
                    <td className="py-1 pr-3 text-ink-400">{s.units}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
