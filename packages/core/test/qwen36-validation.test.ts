import { describe, expect, it } from 'vitest';
import { calculate, builtInModels, builtInGpus, findModel, findGpu } from '../src/index.js';

const within = (actual: number, expected: number, pct = 0.1) => {
  const diff = Math.abs(actual - expected);
  const tol = Math.abs(expected) * pct;
  return diff <= tol;
};

describe('Qwen3.6-27B validation table', () => {
  const model = findModel('Qwen3.6-27B-Hybrid')!;
  const h100 = findGpu('H100 80GB')!;
  const a100_40 = findGpu('A100 40GB')!;
  const rtx4090 = findGpu('RTX 4090')!;
  const context = 256 * 1024;

  it('catalog loads model and GPUs', () => {
    expect(model).toBeDefined();
    expect(builtInModels.length).toBeGreaterThan(5);
    expect(builtInGpus.length).toBeGreaterThan(5);
  });

  it('bf16 weights + bf16 KV: ~54 GB weights, ~16 GB KV @ 256K, ~72 GB total, fits H100 80GB', () => {
    const r = calculate(
      { model, weight_quant: 'bf16', kv_quant: 'bf16', context_length: context, batch_size: 1, tensor_parallel: 1 },
      h100,
    );
    expect(within(r.memory.weights_gb, 54, 0.1)).toBe(true);
    expect(within(r.memory.kv_cache_gb, 16, 0.1)).toBe(true);
    expect(within(r.memory.total_gb, 72, 0.1)).toBe(true);
    expect(r.fits).toBe(true);
  });

  it('4-bit weights + bf16 KV: ~15 GB weights, ~16 GB KV, ~33 GB total, fits A100 40GB', () => {
    const r = calculate(
      { model, weight_quant: 'awq_int4', kv_quant: 'bf16', context_length: context, batch_size: 1, tensor_parallel: 1 },
      a100_40,
    );
    expect(within(r.memory.weights_gb, 15, 0.1)).toBe(true);
    expect(within(r.memory.kv_cache_gb, 16, 0.1)).toBe(true);
    expect(within(r.memory.total_gb, 33, 0.1)).toBe(true);
    expect(r.fits).toBe(true);
  });

  it('4-bit weights + fp8 KV: ~15 GB weights, ~8 GB KV, ~25 GB total, fits RTX 4090 24GB', () => {
    const r = calculate(
      { model, weight_quant: 'awq_int4', kv_quant: 'fp8', context_length: context, batch_size: 1, tensor_parallel: 1 },
      rtx4090,
    );
    expect(within(r.memory.weights_gb, 15, 0.1)).toBe(true);
    expect(within(r.memory.kv_cache_gb, 8, 0.1)).toBe(true);
    expect(within(r.memory.total_gb, 25, 0.1)).toBe(true);
    expect(r.fits).toBe(true);
  });

  it('4-bit weights + int4 KV: ~15 GB weights, ~4 GB KV, ~21 GB total, fits RTX 4090 24GB', () => {
    const r = calculate(
      { model, weight_quant: 'awq_int4', kv_quant: 'int4', context_length: context, batch_size: 1, tensor_parallel: 1 },
      rtx4090,
    );
    expect(within(r.memory.weights_gb, 15, 0.1)).toBe(true);
    expect(within(r.memory.kv_cache_gb, 4, 0.1)).toBe(true);
    expect(within(r.memory.total_gb, 21, 0.1)).toBe(true);
    expect(r.fits).toBe(true);
  });
});
