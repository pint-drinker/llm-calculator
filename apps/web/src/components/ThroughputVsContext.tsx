import { useMemo } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { calculate, type GPU, type InferenceConfig } from '@llm-calc/core';
import { fmtContext, fmtTok } from '../format.js';

interface Props {
  config: InferenceConfig;
  gpu: GPU;
}

function logSpacedContexts(min = 1024, max = 1_048_576, count = 24): number[] {
  const lmin = Math.log2(min);
  const lmax = Math.log2(max);
  return Array.from({ length: count }, (_, i) => {
    const t = lmin + ((lmax - lmin) * i) / (count - 1);
    return Math.round(Math.pow(2, t));
  });
}

export function ThroughputVsContext({ config, gpu }: Props) {
  const data = useMemo(() => {
    return logSpacedContexts().map((c) => {
      const r = calculate({ ...config, context_length: c }, gpu);
      return {
        context: c,
        estimated: r.throughput.estimated_tps,
        memory: r.throughput.memory_bound_tps,
        compute: r.throughput.compute_bound_tps,
      };
    });
  }, [config, gpu]);

  return (
    <div className="panel">
      <h3 className="label mb-2">Throughput vs context</h3>
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
            formatter={(v: number) => fmtTok(v)}
          />
          <Line type="monotone" dataKey="memory" stroke="#38bdf8" dot={false} strokeDasharray="4 2" />
          <Line type="monotone" dataKey="compute" stroke="#facc15" dot={false} strokeDasharray="4 2" />
          <Line type="monotone" dataKey="estimated" stroke="#3fc8a8" dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
      <div className="mt-1 flex justify-end gap-3 text-[10px] text-ink-400">
        <span className="text-accent-400">estimated</span>
        <span className="text-sky-400">memory-bound</span>
        <span className="text-yellow-300">compute-bound</span>
      </div>
    </div>
  );
}
