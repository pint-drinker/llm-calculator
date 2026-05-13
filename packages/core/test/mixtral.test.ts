import { describe, expect, it } from 'vitest';
import { calculate, findGpu, findModel } from '../src/index.js';

describe('Mixtral 8x7B (MoE)', () => {
  const model = findModel('Mixtral-8x7B')!;
  const h100 = findGpu('H100 80GB')!;

  it('total params drives memory (~93 GB bf16)', () => {
    const r = calculate(
      { model, weight_quant: 'bf16', kv_quant: 'bf16', context_length: 8192, batch_size: 1, tensor_parallel: 1 },
      h100,
    );
    expect(r.memory.weights_gb).toBeGreaterThan(85);
    expect(r.memory.weights_gb).toBeLessThan(100);
  });

  it('active params drives compute throughput', () => {
    const dense = findModel('Llama-3.1-70B')!;
    const denseR = calculate(
      { model: dense, weight_quant: 'bf16', kv_quant: 'bf16', context_length: 8192, batch_size: 1, tensor_parallel: 2 },
      h100,
    );
    const moeR = calculate(
      { model, weight_quant: 'bf16', kv_quant: 'bf16', context_length: 8192, batch_size: 1, tensor_parallel: 2 },
      h100,
    );
    // 12.9B active vs 70B → MoE should be far faster on compute axis
    expect(moeR.throughput.compute_bound_tps).toBeGreaterThan(denseR.throughput.compute_bound_tps);
  });
});
