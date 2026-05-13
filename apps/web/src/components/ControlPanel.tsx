import { WEIGHT_QUANTS, KV_QUANTS } from '@llm-calc/core';
import type { KVQuant, TensorParallel, WeightQuant } from '@llm-calc/core';
import { useStore } from '../store.js';
import { fmtContext } from '../format.js';

const TPS: TensorParallel[] = [1, 2, 4, 8];

export function ControlPanel() {
  const {
    weight_quant,
    kv_quant,
    context_length,
    batch_size,
    tensor_parallel,
    setWeightQuant,
    setKvQuant,
    setContext,
    setBatch,
    setTp,
  } = useStore();

  return (
    <section className="panel grid grid-cols-1 gap-4 lg:grid-cols-5">
      <LogContextSlider value={context_length} onChange={setContext} />
      <div>
        <div className="label mb-1">Batch size</div>
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
        value={weight_quant}
        options={WEIGHT_QUANTS as WeightQuant[]}
        onChange={setWeightQuant}
      />
      <Segmented
        label="KV cache quant"
        value={kv_quant}
        options={KV_QUANTS as KVQuant[]}
        onChange={setKvQuant}
      />
      <Segmented
        label="Tensor parallel"
        value={tensor_parallel}
        options={TPS}
        onChange={(v) => setTp(v as TensorParallel)}
        format={(v) => `×${v}`}
      />
    </section>
  );
}

function LogContextSlider({ value, onChange }: { value: number; onChange: (n: number) => void }) {
  const min = Math.log2(1024);
  const max = Math.log2(1_048_576);
  const cur = Math.log2(Math.max(1024, value));
  return (
    <div className="col-span-1 lg:col-span-2">
      <div className="mb-1 flex items-center justify-between">
        <span className="label">Context length</span>
        <span className="font-mono text-sm">{fmtContext(value)}</span>
      </div>
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
  value,
  options,
  onChange,
  format,
}: {
  label: string;
  value: T;
  options: T[];
  onChange: (v: T) => void;
  format?: (v: T) => string;
}) {
  return (
    <div>
      <div className="label mb-1">{label}</div>
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
