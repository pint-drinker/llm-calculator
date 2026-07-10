import { useState } from 'react';
import { ENGINE_WEIGHT_QUANTS, KV_QUANTS, estimateMmprojBytes } from '@llm-calc/core';
import type { KVQuant, TensorParallel, WeightQuant } from '@llm-calc/core';
import { useStore } from '../store.js';
import { fmtContext, fmtGB } from '../format.js';

const TPS: TensorParallel[] = [1, 2, 4, 8];

export function ControlPanel() {
  const {
    weight_quant,
    kv_quant,
    context_length,
    batch_size,
    tensor_parallel,
    include_mmproj,
    inference_engine,
    model,
    setWeightQuant,
    setKvQuant,
    setContext,
    setBatch,
    setTp,
    setIncludeMmproj,
  } = useStore();
  const mmprojGb = estimateMmprojBytes(model, true) / 1e9;

  return (
    <section className="panel grid grid-cols-1 gap-4 lg:grid-cols-6">
      <LogContextSlider value={context_length} onChange={setContext} />
      <div>
        <Label
          icon="👥"
          text="Batch size"
          tooltip="Number of concurrent sequences being processed. Higher batch sizes increase KV cache memory but improve GPU utilization."
        />
        <input
          type="number"
          min={1}
          max={256}
          value={batch_size}
          onChange={(e) => setBatch(Math.max(1, Number(e.target.value) || 1))}
          className="w-full rounded border border-ink-700 bg-ink-800 px-2 py-1.5 text-sm"
        />
      </div>
      <Segmented
        label="Weight quant"
        icon="⚖️"
        tooltip="Precision used to store model weights. Lower precision reduces VRAM usage but may affect output quality. Options follow the selected engine: GGUF quants (q8_0, q4_k_m, etc.) for llama.cpp, HF-style formats for sglang."
        value={weight_quant}
        options={ENGINE_WEIGHT_QUANTS[inference_engine] as WeightQuant[]}
        onChange={setWeightQuant}
      />
      <Segmented
        label="KV cache quant"
        icon="🗄️"
        tooltip="Precision for the key-value cache that stores attention state. Lower precision dramatically reduces memory at long contexts with minimal quality loss."
        value={kv_quant}
        options={KV_QUANTS as KVQuant[]}
        onChange={setKvQuant}
      />
      <div>
        <Label
          icon="🧩"
          text="mmproj"
          tooltip="Include an estimated multimodal projector file: 15% of model params stored at BF16 precision, independent of weight quantization."
        />
        <button
          role="switch"
          aria-checked={include_mmproj}
          onClick={() => setIncludeMmproj(!include_mmproj)}
          className={
            'w-full rounded border px-2 py-1.5 text-left text-sm transition-colors ' +
            (include_mmproj
              ? 'border-accent-500/60 bg-accent-600/20 text-accent-300'
              : 'border-ink-700 bg-ink-800 text-ink-300')
          }
        >
          <span className="font-medium">{include_mmproj ? 'Included' : 'Excluded'}</span>
          <span className="ml-2 font-mono text-xs text-ink-400">{fmtGB(mmprojGb)}</span>
        </button>
      </div>
      <Segmented
        label="Tensor parallel"
        icon="⚡"
        tooltip="Number of GPUs to split the model across. Splits weights and KV cache evenly, multiplying available VRAM and memory bandwidth."
        value={tensor_parallel}
        options={TPS}
        onChange={(v) => setTp(v as TensorParallel)}
        format={(v) => `×${v}`}
      />
    </section>
  );
}

function Label({ icon, text, tooltip }: { icon: string; text: string; tooltip: string }) {
  const [show, setShow] = useState(false);
  return (
    <div className="label mb-1 flex items-center gap-1.5">
      <span>{icon}</span>
      <span>{text}</span>
      <span
        className="relative cursor-help text-ink-500 hover:text-ink-300"
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
      >
        <svg className="inline h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <circle cx="12" cy="12" r="10" />
          <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
          <line x1="12" y1="17" x2="12.01" y2="17" />
        </svg>
        {show && (
          <span className="absolute bottom-full left-1/2 z-50 mb-2 w-56 -translate-x-1/2 rounded bg-ink-700 px-3 py-2 text-xs font-normal leading-relaxed text-ink-200 shadow-lg">
            {tooltip}
          </span>
        )}
      </span>
    </div>
  );
}

function LogContextSlider({ value, onChange }: { value: number; onChange: (n: number) => void }) {
  const sciEnabled = useStore((s) => s.sci_enabled);
  const sciLimit = useStore((s) => s.sci_context_limit);
  const min = Math.log2(1024);
  const max = Math.log2(1_048_576);
  const cur = Math.log2(Math.max(1024, value));

  const showMarker = sciEnabled && sciLimit < value;
  const markerPct = showMarker
    ? ((Math.log2(sciLimit) - min) / (max - min)) * 100
    : 0;

  const effectiveContext = sciEnabled ? Math.min(value, sciLimit) : value;

  return (
    <div className="col-span-1 lg:col-span-2">
      <div className="mb-1 flex items-center justify-between">
        <Label
          icon="📏"
          text="Context length"
          tooltip="Total number of tokens (input + output) the model can process at once. Longer contexts require more KV cache memory."
        />
        <span className="font-mono text-sm">
          {fmtContext(value)}
          {showMarker && (
            <span className="ml-1 text-accent-400">(effective: {fmtContext(effectiveContext)})</span>
          )}
        </span>
      </div>
      <div className="relative">
        <input
          type="range"
          min={min}
          max={max}
          step={0.05}
          value={cur}
          onChange={(e) => {
            const n = Math.round(Math.pow(2, Number(e.target.value)));
            onChange(n);
          }}
          className="w-full accent-accent-500"
        />
        {showMarker && (
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-accent-400 pointer-events-none"
            style={{ left: `${markerPct}%` }}
            title={`SCI effective limit: ${fmtContext(sciLimit)}`}
          />
        )}
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-ink-500">
        <span>1K</span>
        <span>8K</span>
        <span>32K</span>
        <span>128K</span>
        <span>512K</span>
        <span>1M</span>
      </div>
    </div>
  );
}

function Segmented<T extends string | number>({
  label,
  icon,
  tooltip,
  value,
  options,
  onChange,
  format,
}: {
  label: string;
  icon: string;
  tooltip: string;
  value: T;
  options: T[];
  onChange: (v: T) => void;
  format?: (v: T) => string;
}) {
  return (
    <div>
      <Label icon={icon} text={label} tooltip={tooltip} />
      <div className="segmented flex-wrap">
        {options.map((opt) => (
          <button
            key={String(opt)}
            data-active={value === opt}
            onClick={() => onChange(opt)}
          >
            {format ? format(opt) : String(opt)}
          </button>
        ))}
      </div>
    </div>
  );
}
