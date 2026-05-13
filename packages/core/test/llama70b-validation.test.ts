import { describe, expect, it } from 'vitest';
import { calculate, findGpu, findModel } from '../src/index.js';

describe('Llama 3.1 70B validation', () => {
  const model = findModel('Llama-3.1-70B')!;
  const h100 = findGpu('H100 80GB')!;

  it('bf16 weights ≈ 140 GB → does not fit one H100 80GB', () => {
    const r = calculate(
      { model, weight_quant: 'bf16', kv_quant: 'bf16', context_length: 32768, batch_size: 1, tensor_parallel: 1 },
      h100,
    );
    expect(r.memory.weights_gb).toBeGreaterThan(130);
    expect(r.memory.weights_gb).toBeLessThan(145);
    expect(r.fits).toBe(false);
  });

  it('fits two H100s with TP=2', () => {
    const r = calculate(
      { model, weight_quant: 'bf16', kv_quant: 'bf16', context_length: 32768, batch_size: 1, tensor_parallel: 2 },
      h100,
    );
    expect(r.fits).toBe(true);
  });

  it('GQA KV @ 32K is well under 10 GB (8 KV heads × 128 hd × 80 layers)', () => {
    const r = calculate(
      { model, weight_quant: 'bf16', kv_quant: 'bf16', context_length: 32768, batch_size: 1, tensor_parallel: 1 },
      h100,
    );
    expect(r.memory.kv_cache_gb).toBeLessThan(11);
    expect(r.memory.kv_cache_gb).toBeGreaterThan(8);
  });
});
