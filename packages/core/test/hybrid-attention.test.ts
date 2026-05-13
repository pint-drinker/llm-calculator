import { describe, expect, it } from 'vitest';
import { computeMemory, findModel } from '../src/index.js';

describe('Hybrid attention math', () => {
  const model = findModel('Qwen3.6-27B-Hybrid')!;

  it('has 16 full and 48 linear layers', () => {
    const full = model.layers.filter((l) => l.kind === 'full').length;
    const linear = model.layers.filter((l) => l.kind === 'linear').length;
    expect(full).toBe(16);
    expect(linear).toBe(48);
  });

  it('linear state is ~60 MB and context-independent', () => {
    const at1K = computeMemory({
      model,
      weight_quant: 'bf16',
      kv_quant: 'bf16',
      context_length: 1024,
      batch_size: 1,
      tensor_parallel: 1,
    });
    const at256K = computeMemory({
      model,
      weight_quant: 'bf16',
      kv_quant: 'bf16',
      context_length: 256 * 1024,
      batch_size: 1,
      tensor_parallel: 1,
    });
    expect(at1K.breakdown.linear_state_gb).toBeCloseTo(at256K.breakdown.linear_state_gb, 6);
    // 48 layers × 1.25 MB ≈ 60 MB ≈ 0.063 GB (decimal)
    expect(at1K.breakdown.linear_state_gb * 1000).toBeCloseTo(63, 0);
  });

  it('KV @ 256K bf16 ≈ 16 GB on the 16 full layers only', () => {
    const r = computeMemory({
      model,
      weight_quant: 'bf16',
      kv_quant: 'bf16',
      context_length: 256 * 1024,
      batch_size: 1,
      tensor_parallel: 1,
    });
    // 16 full × 4 KV heads × 256 hd × 2 bytes × 256K tokens = 17.18 GB decimal (= 16 GiB)
    expect(r.breakdown.kv_cache_gb).toBeGreaterThan(16);
    expect(r.breakdown.kv_cache_gb).toBeLessThan(18);
  });
});
