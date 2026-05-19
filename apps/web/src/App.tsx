import { useEffect, useMemo, useState } from 'react';
import { calculate } from '@llm-calc/core';
import { useStore, selectConfig } from './store.js';
import { writeUrl } from './url.js';
import { Sidebar } from './components/Sidebar.js';
import { ControlPanel } from './components/ControlPanel.js';
import { ResultsHeader } from './components/ResultsHeader.js';
import { MemoryVsContext } from './components/MemoryVsContext.js';
import { CrossoverChart } from './components/CrossoverChart.js';
import { MathExplainer } from './components/MathExplainer.js';

export function App() {
  const state = useStore();
  const config = useMemo(() => selectConfig(state), [state]);
  const result = useMemo(() => calculate(config, state.gpu), [config, state.gpu]);
  const [shareToast, setShareToast] = useState<string | null>(null);

  useEffect(() => {
    writeUrl({
      model: state.model,
      gpu: state.gpu,
      weight_quant: state.weight_quant,
      kv_quant: state.kv_quant,
      context_length: state.context_length,
      batch_size: state.batch_size,
      tensor_parallel: state.tensor_parallel,
    });
  }, [state]);

  const onShare = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setShareToast('Link copied');
    } catch {
      setShareToast('Copy failed — select the URL bar manually.');
    }
    setTimeout(() => setShareToast(null), 1800);
  };

  return (
    <div className="grid h-full grid-cols-[280px_1fr] gap-4 p-4">
      <Sidebar />
      <main className="flex flex-col gap-4 overflow-auto pr-2">
        <header className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold">LLM VRAM & Throughput Calculator</h1>
            <p className="text-xs text-ink-400">
              Hybrid-aware. Independent weight/KV quantization. Tensor parallelism. MoE.
            </p>
          </div>
          <div className="flex items-center gap-2">
            {shareToast && (
              <span className="rounded bg-accent-600 px-2 py-1 text-xs text-ink-950">
                {shareToast}
              </span>
            )}
            <button className="btn" onClick={onShare}>
              Share link
            </button>
          </div>
        </header>
        <ControlPanel />
        <ResultsHeader result={result} />
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <MemoryVsContext config={config} gpu={state.gpu} />
          <CrossoverChart config={config} gpu={state.gpu} />
        </div>
        <MathExplainer config={config} gpu={state.gpu} />
      </main>
    </div>
  );
}
