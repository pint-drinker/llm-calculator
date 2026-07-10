import { useMemo } from 'react';
import {
  Area,
  AreaChart,
  CartesianGrid,
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
  const out: number[] = [];
  const lmin = Math.log2(min);
  const lmax = Math.log2(max);
  for (let i = 0; i < count; i++) {
    const t = lmin + ((lmax - lmin) * i) / (count - 1);
    out.push(Math.round(Math.pow(2, t)));
  }
  return out;
}

export function MemoryVsContext({ config, gpu }: Props) {
  const data = useMemo(() => {
    return logSpacedContexts().map((c) => {
      const r = calculate({ ...config, context_length: c }, gpu);
      return {
        context: c,
        weights: r.memory.weights_gb,
        mmproj: r.memory.mmproj_gb,
        kv: r.memory.kv_cache_gb,
        linear: r.memory.linear_state_gb,
        activations: r.memory.activations_gb,
        overhead: r.memory.framework_overhead_gb,
        total: r.memory.per_gpu_gb,
      };
    });
  }, [config, gpu]);

  return (
    <div className="panel">
      <h3 className="label mb-2">Memory vs context (per GPU)</h3>
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={data}>
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
          <YAxis tickFormatter={(v) => `${v.toFixed(0)}`} stroke="#9aaac0" fontSize={11} />
          <Tooltip
            contentStyle={{ backgroundColor: '#172235', border: '1px solid #243246' }}
            labelFormatter={(v) => `context: ${fmtContext(Number(v))}`}
            formatter={(v: number) => fmtGB(v)}
          />
          <Area dataKey="weights" stackId="1" stroke="#3fc8a8" fill="#3fc8a8" />
          <Area dataKey="mmproj" stackId="1" stroke="#f97316" fill="#f97316" />
          <Area dataKey="kv" stackId="1" stroke="#38bdf8" fill="#38bdf8" />
          <Area dataKey="linear" stackId="1" stroke="#a855f7" fill="#a855f7" />
          <Area dataKey="activations" stackId="1" stroke="#ec4899" fill="#ec4899" />
          <Area dataKey="overhead" stackId="1" stroke="#4b5d77" fill="#4b5d77" />
          <ReferenceLine y={gpu.vram_gb} stroke="#f97366" strokeDasharray="4 4" label={{ value: `${gpu.name} capacity`, fill: '#f97366', fontSize: 10, position: 'insideTopRight' }} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
