import { useMemo } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { calculate, type GPU, type InferenceConfig } from '@llm-calc/core';
import { fmtContext, fmtGB } from '../format.js';

interface Props {
  config: InferenceConfig;
  gpu: GPU;
}

function logSpacedContexts(min = 1024, max = 1_048_576, count = 30): number[] {
  const lmin = Math.log2(min);
  const lmax = Math.log2(max);
  return Array.from({ length: count }, (_, i) => {
    const t = lmin + ((lmax - lmin) * i) / (count - 1);
    return Math.round(Math.pow(2, t));
  });
}

export function CrossoverChart({ config, gpu }: Props) {
  const { data, crossover } = useMemo(() => {
    const points = logSpacedContexts().map((c) => {
      const r = calculate({ ...config, context_length: c }, gpu);
      return {
        context: c,
        modelFiles: r.memory.weights_gb + r.memory.mmproj_gb,
        kv: r.memory.kv_cache_gb,
      };
    });
    let cx: number | null = null;
    for (const p of points) {
      if (p.kv >= p.modelFiles) {
        cx = p.context;
        break;
      }
    }
    return { data: points, crossover: cx };
  }, [config, gpu]);

  return (
    <div className="panel">
      <h3 className="label mb-2">Model files vs KV cache (inversion)</h3>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data}>
          <CartesianGrid stroke="#243246" strokeDasharray="2 4" />
          <XAxis
            dataKey="context"
            scale="log"
            domain={['auto', 'auto']}
            type="number"
            tickFormatter={fmtContext}
            stroke="#9aaac0"
            fontSize={11}
          />
          <YAxis stroke="#9aaac0" fontSize={11} />
          <Tooltip
            contentStyle={{ backgroundColor: '#172235', border: '1px solid #243246' }}
            labelFormatter={(v) => `context: ${fmtContext(Number(v))}`}
            formatter={(v: number) => fmtGB(v)}
          />
          <Line
            type="monotone"
            dataKey="modelFiles"
            name="model files"
            stroke="#3fc8a8"
            dot={false}
            strokeWidth={2}
          />
          <Line type="monotone" dataKey="kv" stroke="#38bdf8" dot={false} strokeWidth={2} />
          {crossover && (
            <ReferenceLine
              x={crossover}
              stroke="#f97366"
              strokeDasharray="3 3"
              label={{ value: `crossover ≈ ${fmtContext(crossover)}`, fill: '#f97366', fontSize: 10, position: 'insideTopLeft' }}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
      <p className="mt-1 text-[10px] text-ink-400">
        Where KV cache overtakes weights plus mmproj — the point at which context length, not model-file size, dominates VRAM.
      </p>
    </div>
  );
}
