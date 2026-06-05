import { useEffect, useMemo, useState } from 'react';
import { calculate } from '@llm-calc/core';
import { useStore, selectConfig, selectEffectiveGpu } from './store.js';
import { writeUrl } from './url.js';
import { fmtContext } from './format.js';
import { Sidebar } from './components/Sidebar.js';
import { ControlPanel } from './components/ControlPanel.js';
import { ResultsHeader } from './components/ResultsHeader.js';
import { MemoryVsContext } from './components/MemoryVsContext.js';
import { CrossoverChart } from './components/CrossoverChart.js';
import { TtftVsContext } from './components/TtftVsContext.js';
import { MathExplainer } from './components/MathExplainer.js';

export function App() {
  const state = useStore();
  const config = useMemo(() => selectConfig(state), [state]);
  const gpu = useMemo(() => selectEffectiveGpu(state), [state.gpu, state.usable_memory_fraction]);
  const result = useMemo(() => calculate(config, gpu), [config, gpu]);
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
      inference_engine: state.inference_engine,
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
          <div className="flex items-center gap-3">
            {shareToast && (
              <span className="rounded bg-accent-600 px-2 py-1 text-xs text-ink-950">
                {shareToast}
              </span>
            )}
            <EngineToggle />
            <button className="btn" onClick={onShare}>
              Share link
            </button>
          </div>
        </header>
        <SciToggle />
        <ControlPanel />
        <ResultsHeader result={result} />
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <MemoryVsContext config={config} gpu={gpu} />
          <CrossoverChart config={config} gpu={gpu} />
          <TtftVsContext config={config} gpu={gpu} />
        </div>
        <MathExplainer config={config} gpu={gpu} />
      </main>
    </div>
  );
}

function EngineToggle() {
  const engine = useStore((s) => s.inference_engine);
  const setEngine = useStore((s) => s.setInferenceEngine);

  return (
    <div
      className="flex items-center gap-2"
      title="Throughput & TTFT estimates are calibrated per engine. llama.cpp runs well below the hardware roofline (GGUF dequant overhead, weak MoE expert kernels); sglang-style serving gets close to it."
    >
      <span className="label">Engine</span>
      <div className="segmented">
        <button data-active={engine === 'sglang'} onClick={() => setEngine('sglang')}>
          sglang
        </button>
        <button data-active={engine === 'llama_cpp'} onClick={() => setEngine('llama_cpp')}>
          llama.cpp
        </button>
      </div>
    </div>
  );
}

function SciToggle() {
  const enabled = useStore((s) => s.sci_enabled);
  const limit = useStore((s) => s.sci_context_limit);
  const setEnabled = useStore((s) => s.setSciEnabled);
  const setLimit = useStore((s) => s.setSciContextLimit);

  return (
    <div
      className={
        'flex items-center gap-3 rounded-lg border px-4 py-2 transition-colors ' +
        (enabled
          ? 'border-accent-500/50 bg-accent-600/10'
          : 'border-ink-700 bg-ink-800/50')
      }
    >
      <button
        role="switch"
        aria-checked={enabled}
        onClick={() => setEnabled(!enabled)}
        className={
          'relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full transition-colors ' +
          (enabled ? 'bg-accent-500' : 'bg-ink-600')
        }
      >
        <span
          className={
            'pointer-events-none inline-block h-4 w-4 translate-y-0.5 rounded-full bg-white shadow transition-transform ' +
            (enabled ? 'translate-x-[18px]' : 'translate-x-0.5')
          }
        />
      </button>
      <span className="text-sm font-medium">Subconscious Context Intelligence</span>
      {enabled && (
        <div className="ml-auto flex items-center gap-2 text-xs text-ink-300">
          <span>Effective limit:</span>
          <input
            type="number"
            min={1024}
            step={1024}
            value={limit}
            onChange={(e) => setLimit(Math.max(1024, Number(e.target.value) || 40960))}
            className="w-20 rounded border border-ink-600 bg-ink-800 px-2 py-0.5 text-sm font-mono"
          />
          <span className="text-ink-500">({fmtContext(limit)})</span>
        </div>
      )}
    </div>
  );
}
