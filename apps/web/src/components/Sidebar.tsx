import { builtInGpus, builtInModels } from '@llm-calc/core';
import type { GPU, ModelConfig } from '@llm-calc/core';
import { useStore } from '../store.js';
import { Presets } from './Presets.js';

export function Sidebar() {
  const {
    model,
    gpu,
    initial_prompt_tokens,
    ttft_threshold_s,
    throughput_threshold_tps,
    setModel,
    setGpu,
    setInitialPrompt,
    setTtftThreshold,
    setThroughputThreshold,
  } = useStore();

  return (
    <aside className="panel flex flex-col gap-4 overflow-auto">
      <section>
        <h2 className="label mb-1">Model</h2>
        <select
          className="w-full rounded border border-ink-700 bg-ink-800 px-2 py-1.5 text-sm"
          value={model.name}
          onChange={(e) => {
            const m = builtInModels.find((x) => x.name === e.target.value);
            if (m) setModel(m);
          }}
        >
          {builtInModels.map((m) => (
            <option key={m.name} value={m.name}>
              {m.name}
            </option>
          ))}
        </select>
        <ModelSummary model={model} />
      </section>

      <section>
        <h2 className="label mb-1">GPU</h2>
        <select
          className="w-full rounded border border-ink-700 bg-ink-800 px-2 py-1.5 text-sm"
          value={gpu.name}
          onChange={(e) => {
            const g = builtInGpus.find((x) => x.name === e.target.value);
            if (g) setGpu(g);
          }}
        >
          {builtInGpus.map((g) => (
            <option key={g.name} value={g.name}>
              {g.name} — {g.vram_gb.toFixed(0)} GB
            </option>
          ))}
        </select>
        <GpuSummary gpu={gpu} />
      </section>

      <section>
        <h2 className="label mb-1">Experience thresholds</h2>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-[10px] text-ink-400">Max TTFT (s)</label>
            <input
              type="number"
              min={0.1}
              step={0.5}
              value={ttft_threshold_s}
              onChange={(e) => setTtftThreshold(Math.max(0.1, Number(e.target.value) || 5))}
              className="w-full rounded border border-ink-700 bg-ink-800 px-2 py-1 text-sm"
            />
          </div>
          <div>
            <label className="text-[10px] text-ink-400">Min tok/s</label>
            <input
              type="number"
              min={1}
              step={1}
              value={throughput_threshold_tps}
              onChange={(e) => setThroughputThreshold(Math.max(1, Number(e.target.value) || 15))}
              className="w-full rounded border border-ink-700 bg-ink-800 px-2 py-1 text-sm"
            />
          </div>
        </div>
        <div className="mt-2">
          <label className="text-[10px] text-ink-400">Initial prompt (tokens)</label>
          <input
            type="number"
            min={100}
            step={500}
            value={initial_prompt_tokens}
            onChange={(e) => setInitialPrompt(Math.max(100, Number(e.target.value) || 2000))}
            className="w-full rounded border border-ink-700 bg-ink-800 px-2 py-1 text-sm"
          />
        </div>
        <p className="mt-1.5 text-[10px] text-ink-500">
          TTFT is computed at initial prompt size. Thresholds flag values in results.
        </p>
      </section>

      <section>
        <h2 className="label mb-1">Presets</h2>
        <Presets />
      </section>
    </aside>
  );
}

function ModelSummary({ model }: { model: ModelConfig }) {
  const full = model.layers.filter((l) => l.kind === 'full').length;
  const linear = model.layers.filter((l) => l.kind === 'linear').length;
  return (
    <dl className="mt-3 grid grid-cols-2 gap-y-1 text-xs text-ink-300">
      <dt>Params</dt>
      <dd className="font-mono">{(model.params / 1e9).toFixed(1)}B</dd>
      {model.active_params && (
        <>
          <dt>Active</dt>
          <dd className="font-mono">{(model.active_params / 1e9).toFixed(1)}B</dd>
        </>
      )}
      <dt>Layers</dt>
      <dd className="font-mono">
        {full} full · {linear} linear
      </dd>
      <dt>Hidden</dt>
      <dd className="font-mono">{model.hidden_dim}</dd>
    </dl>
  );
}

function GpuSummary({ gpu }: { gpu: GPU }) {
  return (
    <dl className="mt-2 grid grid-cols-2 gap-y-1 text-xs text-ink-300">
      <dt>VRAM</dt>
      <dd className="font-mono">{gpu.vram_gb.toFixed(0)} GB</dd>
      <dt>Bandwidth</dt>
      <dd className="font-mono">{gpu.memory_bandwidth_gbs.toFixed(0)} GB/s</dd>
      <dt>FP16</dt>
      <dd className="font-mono">{gpu.fp16_tflops.toFixed(0)} TF</dd>
      {gpu.fp8_tflops && (
        <>
          <dt>FP8</dt>
          <dd className="font-mono">{gpu.fp8_tflops.toFixed(0)} TF</dd>
        </>
      )}
    </dl>
  );
}
