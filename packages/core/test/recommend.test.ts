import { describe, expect, it } from 'vitest';
import {
  builtInGpus,
  findGpu,
  findMaxContext,
  findModel,
  recommendHardware,
} from '../src/index.js';

describe('findMaxContext', () => {
  const model = findModel('Llama-3.1-8B')!;
  const rtx4090 = findGpu('RTX 4090')!;

  it('returns a sensible max for a model that fits at small context', () => {
    const r = findMaxContext({
      model,
      weight_quant: 'bf16',
      kv_quant: 'bf16',
      gpu: rtx4090,
      tensor_parallel: 1,
    });
    expect(r.max_context).toBeGreaterThan(1024);
    expect(r.max_context).toBeLessThan(8_388_608);
  });

  it('returns 0 when weights alone overflow', () => {
    const big = findModel('Llama-3.1-405B')!;
    const r = findMaxContext({
      model: big,
      weight_quant: 'bf16',
      kv_quant: 'bf16',
      gpu: rtx4090,
      tensor_parallel: 1,
    });
    expect(r.max_context).toBe(0);
  });
});

describe('recommendHardware', () => {
  it('returns fitting candidates sorted by smallest-fits first', () => {
    const model = findModel('Llama-3.1-8B')!;
    const cands = recommendHardware({
      model,
      weight_quant: 'bf16',
      kv_quant: 'bf16',
      context_length: 8192,
      gpus: builtInGpus,
    });
    const firstFitting = cands.find((c) => c.fits)!;
    expect(firstFitting).toBeDefined();
    expect(firstFitting.fits).toBe(true);
  });
});
