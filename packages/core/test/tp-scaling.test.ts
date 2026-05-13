import { describe, expect, it } from 'vitest';
import { calculate, findGpu, findModel } from '../src/index.js';

describe('Tensor parallelism', () => {
  const model = findModel('Llama-3.1-70B')!;
  const h100 = findGpu('H100 80GB')!;

  const at = (tp: 1 | 2 | 4 | 8) =>
    calculate(
      { model, weight_quant: 'bf16', kv_quant: 'bf16', context_length: 32768, batch_size: 1, tensor_parallel: tp },
      h100,
    );

  it('weights divide by TP per GPU; overhead does not', () => {
    const tp1 = at(1);
    const tp2 = at(2);
    const tp4 = at(4);
    expect(tp2.memory.per_gpu_gb).toBeLessThan(tp1.memory.per_gpu_gb);
    expect(tp4.memory.per_gpu_gb).toBeLessThan(tp2.memory.per_gpu_gb);
    // per_gpu - overhead halves approximately
    const adj = (r: number) => r - 1; // 1 GB framework overhead
    expect(adj(tp2.memory.per_gpu_gb)).toBeCloseTo(adj(tp1.memory.per_gpu_gb) / 2, 0);
  });

  it('throughput scales with TP but with 0.85 efficiency above tp=1', () => {
    const tp1 = at(1);
    const tp2 = at(2);
    // memory_bound_tps = bw * tp * eff / bytes → ratio = tp * eff (0.85 vs 1.0)
    const ratio = tp2.throughput.memory_bound_tps / tp1.throughput.memory_bound_tps;
    expect(ratio).toBeCloseTo(2 * 0.85, 2);
  });
});
