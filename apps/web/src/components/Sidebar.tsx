import { useState } from 'react';
import { builtInGpus, builtInModels } from '@llm-calc/core';
import type { GPU, ModelConfig } from '@llm-calc/core';
import { useStore } from '../store.js';
import { HFImportDialog } from './HFImportDialog.js';
import { CustomModelDrawer } from './CustomModelDrawer.js';
import { Presets } from './Presets.js';

export function Sidebar() {
  const { model, gpu, customModels, setModel, setGpu, addCustomModel } = useStore();
  const [hfOpen, setHfOpen] = useState(false);
  const [customOpen, setCustomOpen] = useState(false);
  const allModels: ModelConfig[] = [...customModels, ...builtInModels];

  return (
    <aside className="panel flex flex-col gap-4 overflow-auto">
      <section>
        <h2 className="label mb-1">Model</h2>
        <select
          className="w-full rounded border border-ink-700 bg-ink-800 px-2 py-1.5 text-sm"
          value={model.name}
          onChange={(e) => {
            const m = allModels.find((x) => x.name === e.target.value);
            if (m) setModel(m);
          }}
        >
          {customModels.length > 0 && (
            <optgroup label="Custom / Imported">
              {customModels.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name}
                </option>
              ))}
            </optgroup>
          )}
          <optgroup label="Catalog">
            {builtInModels.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name}
              </option>
            ))}
          </optgroup>
        </select>
        <div className="mt-2 flex gap-2">
          <button className="btn flex-1" onClick={() => setHfOpen(true)}>
            + Import from HuggingFace
          </button>
          <button className="btn flex-1" onClick={() => setCustomOpen(true)}>
            + Custom
          </button>
        </div>
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
        <h2 className="label mb-1">Presets</h2>
        <Presets />
      </section>

      {hfOpen && (
        <HFImportDialog
          onClose={() => setHfOpen(false)}
          onImported={(m) => {
            addCustomModel(m);
            setHfOpen(false);
          }}
        />
      )}
      {customOpen && (
        <CustomModelDrawer
          onClose={() => setCustomOpen(false)}
          onSave={(m) => {
            addCustomModel(m);
            setCustomOpen(false);
          }}
        />
      )}
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
