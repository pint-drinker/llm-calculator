import type { CalculationResult } from '@llm-calc/core';
import { useStore } from '../store.js';
import { fmtGB, fmtSeconds, fmtTok } from '../format.js';

interface Props {
  result: CalculationResult;
}

export function ResultsHeader({ result }: Props) {
  const gpu = useStore((s) => s.gpu);
  const highlighted = useStore((s) => s.highlightedValue);
  const setHighlight = useStore((s) => s.setHighlighted);
  const fits = result.fits;
  const headroomGb = gpu.vram_gb - result.memory.per_gpu_gb;

  return (
    <section className="panel">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-stretch">
        <div className="lg:w-64">
          <div className="label">Fit on {gpu.name}</div>
          <div
            className={
              'mt-1 flex items-baseline gap-2 ' +
              (fits ? 'text-accent-400' : 'text-danger-400')
            }
          >
            <span className="text-3xl font-semibold">{fits ? '✓ Fits' : '✕ Overflow'}</span>
          </div>
          <div className="mt-1 text-xs text-ink-400">
            {fits
              ? `${fmtGB(headroomGb)} headroom`
              : `${fmtGB(-headroomGb)} over capacity`}
          </div>
          <div className="mt-2 text-xs text-ink-400">
            Utilization{' '}
            <span className="font-mono text-ink-100">
              {result.utilization_pct.toFixed(1)}%
            </span>
          </div>
        </div>

        <div className="flex-1">
          <MemoryBar result={result} vram_gb={gpu.vram_gb} highlight={highlighted} setHighlight={setHighlight} />
        </div>

        <div className="lg:w-56">
          <div className="label">Throughput (decode)</div>
          <div className="mt-1 text-3xl font-semibold">
            {fmtTok(result.throughput.estimated_tps)}
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
          <div className="mt-2 text-xs text-ink-400">
            TTFT{' '}
            <span className="font-mono text-ink-100">
              {fmtSeconds(result.throughput.ttft_seconds)}
            </span>
          </div>
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
  highlight: string | null;
  setHighlight: (k: string | null) => void;
}

function MemoryBar({ result, vram_gb, highlight, setHighlight }: BarProps) {
  const m = result.memory;
  const total = m.per_gpu_gb;
  const max = Math.max(total, vram_gb) * 1.05;
  const overflow = total > vram_gb;
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
        <div
          className="absolute inset-y-0 border-l-2 border-dashed border-ink-400"
          style={{ left: `${(vram_gb / max) * 100}%` }}
          title={`Capacity: ${fmtGB(vram_gb)}`}
        />
        {overflow && (
          <div className="pointer-events-none absolute inset-0 ring-2 ring-danger-500/60" />
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
