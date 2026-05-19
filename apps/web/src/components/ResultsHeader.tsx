import { useMemo } from 'react';
import { type CalculationResult, computeMemory, computePrefillTime } from '@llm-calc/core';
import { useStore, selectConfig } from '../store.js';
import { fmtGB, fmtSeconds, fmtTok } from '../format.js';

interface Props {
  result: CalculationResult;
}

export function ResultsHeader({ result }: Props) {
  const gpu = useStore((s) => s.gpu);
  const config = useStore(selectConfig);
  const initialPromptTokens = useStore((s) => s.initial_prompt_tokens);
  const highlighted = useStore((s) => s.highlightedValue);
  const setHighlight = useStore((s) => s.setHighlighted);
  const ttftThreshold = useStore((s) => s.ttft_threshold_s);
  const tpsThreshold = useStore((s) => s.throughput_threshold_tps);
  const { fits, fits_usable, usable_vram_gb } = result;
  const headroomGb = gpu.vram_gb - result.memory.per_gpu_gb;
  const usableHeadroomGb = usable_vram_gb - result.memory.per_gpu_gb;
  const usableFraction = gpu.usable_memory_fraction ?? 1.0;

  const status: 'fits' | 'tight' | 'overflow' = !fits
    ? 'overflow'
    : !fits_usable
      ? 'tight'
      : 'fits';
  const statusLabel = {
    fits: '✓ Fits',
    tight: '⚠ Tight',
    overflow: '✕ Overflow',
  }[status];
  const statusColor = {
    fits: 'text-accent-400',
    tight: 'text-yellow-400',
    overflow: 'text-danger-400',
  }[status];

  const ttft = useMemo(() => {
    const mem = computeMemory(config);
    return computePrefillTime(config, mem, gpu, initialPromptTokens);
  }, [config, gpu, initialPromptTokens]);
  const tps = result.throughput.estimated_tps;
  const ttftExceeds = Number.isFinite(ttft) && ttft > ttftThreshold;
  const tpsBelow = Number.isFinite(tps) && tps > 0 && tps < tpsThreshold;

  return (
    <section className="panel">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-stretch">
        <div className="lg:w-56">
          <div className="label">Fit on {gpu.name}</div>
          <div className={'mt-1 flex items-baseline gap-2 ' + statusColor}>
            <span className="text-3xl font-semibold">{statusLabel}</span>
          </div>
          <div className="mt-1 text-xs text-ink-400">
            {status === 'fits' && `${fmtGB(usableHeadroomGb)} usable headroom`}
            {status === 'tight' &&
              `${fmtGB(-usableHeadroomGb)} over usable, ${fmtGB(headroomGb)} of physical left`}
            {status === 'overflow' && `${fmtGB(-headroomGb)} over physical capacity`}
          </div>
          <div className="mt-2 text-xs text-ink-400">
            Utilization{' '}
            <span className="font-mono text-ink-100">
              {result.utilization_pct.toFixed(1)}%
            </span>
            <span className="ml-2 text-ink-500">
              of {fmtGB(gpu.vram_gb)} ({(usableFraction * 100).toFixed(0)}% usable)
            </span>
          </div>
        </div>

        <div className="flex-1">
          <MemoryBar
            result={result}
            vram_gb={gpu.vram_gb}
            usable_vram_gb={usable_vram_gb}
            highlight={highlighted}
            setHighlight={setHighlight}
          />
        </div>

        <div className="lg:w-48">
          <div className="label">Throughput (decode)</div>
          <div className={'mt-1 text-3xl font-semibold ' + (tpsBelow ? 'text-danger-400' : '')}>
            {fmtTok(tps)}
          </div>
          <div className="mt-1 flex items-center gap-2 text-xs">
            <span
              className={
                'pill ' +
                (result.throughput.bottleneck === 'memory'
                  ? 'bg-accent-600/40 text-accent-400'
                  : 'bg-yellow-600/40 text-yellow-300')
              }
            >
              {result.throughput.bottleneck}-bound
            </span>
          </div>
          {tpsBelow && (
            <div className="mt-2 flex items-center gap-1 text-xs text-danger-400">
              <span>⚠</span>
              <span>Below {tpsThreshold} tok/s threshold</span>
            </div>
          )}
        </div>

        <div className="lg:w-48">
          <div className="label">Time to first token</div>
          <div className={'mt-1 text-3xl font-semibold ' + (ttftExceeds ? 'text-danger-400' : '')}>
            {fmtSeconds(ttft)}
          </div>
          <div className="mt-1 text-xs text-ink-400">
            Prefill at {(initialPromptTokens / 1000).toFixed(1)}K tokens
          </div>
          {ttftExceeds && (
            <div className="mt-2 flex items-center gap-1 text-xs text-danger-400">
              <span>⚠</span>
              <span>Exceeds {ttftThreshold}s threshold</span>
            </div>
          )}
        </div>
      </div>
      {result.warnings.length > 0 && (
        <ul className="mt-3 space-y-1 text-xs text-yellow-400">
          {result.warnings.map((w) => (
            <li key={w}>⚠ {w}</li>
          ))}
        </ul>
      )}
    </section>
  );
}

interface BarProps {
  result: CalculationResult;
  vram_gb: number;
  usable_vram_gb: number;
  highlight: string | null;
  setHighlight: (k: string | null) => void;
}

function MemoryBar({ result, vram_gb, usable_vram_gb, highlight, setHighlight }: BarProps) {
  const m = result.memory;
  const total = m.per_gpu_gb;
  const max = Math.max(total, vram_gb) * 1.05;
  const overflow = total > vram_gb;
  const overUsable = total > usable_vram_gb && !overflow;
  const showUsableLine = usable_vram_gb < vram_gb;
  const segments: { label: string; value: number; color: string; key: string }[] = [
    { label: 'Weights', value: m.weights_gb, color: 'bg-accent-500', key: 'weights' },
    { label: 'KV cache', value: m.kv_cache_gb, color: 'bg-sky-500', key: 'kv' },
    { label: 'Linear', value: m.linear_state_gb, color: 'bg-purple-500', key: 'linear' },
    { label: 'Activations', value: m.activations_gb, color: 'bg-pink-500', key: 'activations' },
    { label: 'Overhead', value: m.framework_overhead_gb, color: 'bg-ink-500', key: 'overhead' },
  ];
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-xs text-ink-400">
        <span className="label">Memory breakdown (per GPU)</span>
        <span className={'font-mono ' + (overflow ? 'text-danger-400' : 'text-ink-200')}>
          {fmtGB(total)} / {fmtGB(vram_gb)}
        </span>
      </div>
      <div className="relative h-9 w-full overflow-hidden rounded bg-ink-800">
        <div
          className={'absolute inset-y-0 left-0 flex ' + (overflow ? 'opacity-90' : '')}
          style={{ width: `${Math.min((total / max) * 100, 100)}%` }}
        >
          {segments.map((s) => (
            <button
              key={s.key}
              title={`${s.label}: ${fmtGB(s.value)}`}
              onMouseEnter={() => setHighlight(s.key)}
              onMouseLeave={() => setHighlight(null)}
              onClick={() => setHighlight(highlight === s.key ? null : s.key)}
              className={
                s.color +
                ' h-full transition-all ' +
                (highlight && highlight !== s.key ? 'opacity-30' : 'opacity-100')
              }
              style={{
                width: `${(s.value / total) * 100}%`,
              }}
            />
          ))}
        </div>
        {showUsableLine && (
          <div
            className="absolute inset-y-0 border-l border-dashed border-yellow-500/70"
            style={{ left: `${(usable_vram_gb / max) * 100}%` }}
            title={`Usable cap: ${fmtGB(usable_vram_gb)}`}
          />
        )}
        <div
          className="absolute inset-y-0 border-l-2 border-dashed border-ink-400"
          style={{ left: `${(vram_gb / max) * 100}%` }}
          title={`Physical capacity: ${fmtGB(vram_gb)}`}
        />
        {overflow && (
          <div className="pointer-events-none absolute inset-0 ring-2 ring-danger-500/60" />
        )}
        {overUsable && (
          <div className="pointer-events-none absolute inset-0 ring-2 ring-yellow-500/50" />
        )}
      </div>
      <div className="mt-2 flex flex-wrap gap-3 text-xs text-ink-300">
        {segments.map((s) => (
          <button
            key={s.key}
            onClick={() => setHighlight(highlight === s.key ? null : s.key)}
            className={
              'flex items-center gap-1.5 rounded px-1 py-0.5 ' +
              (highlight === s.key ? 'bg-ink-700 text-ink-50' : '')
            }
          >
            <span className={'inline-block h-2 w-2 rounded-sm ' + s.color} />
            {s.label}
            <span className="font-mono text-ink-400">{fmtGB(s.value)}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
