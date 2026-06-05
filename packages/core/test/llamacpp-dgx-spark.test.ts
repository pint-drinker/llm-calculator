import { describe, expect, it } from 'vitest';
import type { InferenceConfig } from '../src/index.js';
import { calculate, findGpu, findModel } from '../src/index.js';

const within = (actual: number, expected: number, pct = 0.1) => {
  const diff = Math.abs(actual - expected);
  const tol = Math.abs(expected) * pct;
  return diff <= tol;
};

// Measured llama.cpp decode throughput on DGX Spark (June 2026), the
// calibration targets for the llama_cpp engine efficiency factors.
describe('llama.cpp on DGX Spark validation', () => {
  const dense = findModel('Qwen3.6-27B-Hybrid')!;
  const moe = findModel('Qwen3.6-35B-A3B')!;
  const spark = findGpu('DGX Spark')!;
  const context = 8192;

  const cfg = (overrides: Partial<InferenceConfig>): InferenceConfig => ({
    model: dense,
    weight_quant: 'q8_0',
    kv_quant: 'fp8',
    context_length: context,
    batch_size: 1,
    tensor_parallel: 1,
    ...overrides,
  });

  it('27B dense q8_0: ~7.5 t/s on llama.cpp', () => {
    const r = calculate(cfg({ engine: 'llama_cpp' }), spark);
    expect(within(r.throughput.estimated_tps, 7.5, 0.1)).toBe(true);
  });

  it('27B dense q4_k_m: ~12 t/s on llama.cpp', () => {
    const r = calculate(cfg({ weight_quant: 'q4_k_m', engine: 'llama_cpp' }), spark);
    expect(within(r.throughput.estimated_tps, 12, 0.1)).toBe(true);
  });

  it('35B-A3B MoE q4_k_m: ~57 t/s on llama.cpp', () => {
    const r = calculate(cfg({ model: moe, weight_quant: 'q4_k_m', engine: 'llama_cpp' }), spark);
    expect(within(r.throughput.estimated_tps, 57, 0.1)).toBe(true);
  });

  it('sglang keeps the roofline estimates: ~10.1 / ~18 / ~162 t/s', () => {
    const int8 = calculate(cfg({ weight_quant: 'int8', engine: 'sglang' }), spark);
    expect(within(int8.throughput.estimated_tps, 10.1, 0.1)).toBe(true);

    const awq = calculate(cfg({ weight_quant: 'awq_int4', engine: 'sglang' }), spark);
    expect(within(awq.throughput.estimated_tps, 18, 0.1)).toBe(true);

    const moeAwq = calculate(
      cfg({ model: moe, weight_quant: 'awq_int4', engine: 'sglang' }),
      spark,
    );
    expect(within(moeAwq.throughput.estimated_tps, 162, 0.1)).toBe(true);
  });

  it('omitting engine matches sglang exactly', () => {
    const omitted = calculate(cfg({}), spark);
    const sglang = calculate(cfg({ engine: 'sglang' }), spark);
    expect(omitted.throughput.estimated_tps).toBe(sglang.throughput.estimated_tps);
    expect(omitted.throughput.ttft_seconds).toBe(sglang.throughput.ttft_seconds);
  });

  it('llama.cpp TTFT is ~2x sglang TTFT for dense compute-bound prefill', () => {
    const long = { context_length: 65536 };
    const sglang = calculate(cfg({ ...long, engine: 'sglang' }), spark);
    const llamacpp = calculate(cfg({ ...long, engine: 'llama_cpp' }), spark);
    expect(within(llamacpp.throughput.ttft_seconds, sglang.throughput.ttft_seconds * 2, 0.01)).toBe(
      true,
    );
  });
});
