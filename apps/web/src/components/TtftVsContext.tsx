import { useMemo } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  computeMemory,
  computePrefillTime,
  type GPU,
  type InferenceConfig,
} from '@llm-calc/core';
import { useStore } from '../store.js';
import { fmtContext, fmtSeconds } from '../format.js';

interface Props {
  config: InferenceConfig;
  gpu: GPU;
}

const TOKENS_PER_TURN = 500;

function logSpacedContexts(min = 512, max = 1_048_576, count = 30): number[] {
  const out: number[] = [];
  const lmin = Math.log2(min);
  const lmax = Math.log2(max);
  for (let i = 0; i < count; i++) {
    const t = lmin + ((lmax - lmin) * i) / (count - 1);
    out.push(Math.round(Math.pow(2, t)));
  }
  return out;
}

export function TtftVsContext({ config, gpu }: Props) {
  const ttftThreshold = useStore((s) => s.ttft_threshold_s);
  const maxContext = config.context_length;

  const data = useMemo(() => {
    const memory = computeMemory(config);
    return logSpacedContexts(512, maxContext).map((c) => {
      const coldStart = computePrefillTime(config, memory, gpu, c);
      const cachedContext = Math.max(0, c - TOKENS_PER_TURN);
      const cached = computePrefillTime(config, memory, gpu, TOKENS_PER_TURN, cachedContext);
      return { context: c, coldStart, cached };
    });
  }, [config, gpu, maxContext]);

  return (
    <div className="panel">
      <h3 className="label mb-2">TTFT vs context length</h3>
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
          <YAxis
            scale="log"
            domain={['auto', 'auto']}
            type="number"
            allowDataOverflow
            tickFormatter={(v) => fmtSeconds(v)}
            stroke="#9aaac0"
            fontSize={11}
          />
          <Tooltip
            contentStyle={{ backgroundColor: '#172235', border: '1px solid #243246' }}
            labelFormatter={(v) => `context: ${fmtContext(Number(v))}`}
            formatter={(v: number, name: string) => [fmtSeconds(v), name]}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line
            name="Cold start"
            dataKey="coldStart"
            stroke="#f97366"
            dot={false}
            strokeWidth={2}
          />
          <Line
            name="Cached (per turn)"
            dataKey="cached"
            stroke="#3fc8a8"
            dot={false}
            strokeWidth={2}
          />
          <ReferenceLine
            y={ttftThreshold}
            stroke="#facc15"
            strokeDasharray="4 4"
            label={{
              value: `${ttftThreshold}s threshold`,
              fill: '#facc15',
              fontSize: 10,
              position: 'insideTopRight',
            }}
          />
        </LineChart>
      </ResponsiveContainer>
      <p className="mt-1.5 text-[10px] text-ink-500">
        Cold start = full prefill. Cached = {TOKENS_PER_TURN} new tokens/turn with warm KV cache.
      </p>
    </div>
  );
}
