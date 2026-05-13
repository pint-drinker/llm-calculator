import { describe, expect, it } from 'vitest';
import { explain, findGpu, findModel } from '../src/index.js';

describe('explain', () => {
  it('returns one step per formula with finite results', () => {
    const model = findModel('Qwen3.6-27B-Hybrid')!;
    const gpu = findGpu('H100 80GB')!;
    const trace = explain(
      { model, weight_quant: 'awq_int4', kv_quant: 'fp8', context_length: 262144, batch_size: 1, tensor_parallel: 1 },
      gpu,
    );
    expect(trace.steps.length).toBeGreaterThanOrEqual(10);
    for (const s of trace.steps) {
      expect(Number.isFinite(s.result)).toBe(true);
      expect(s.formula.length).toBeGreaterThan(0);
      expect(s.name.length).toBeGreaterThan(0);
    }
    const names = new Set(trace.steps.map((s) => s.name));
    expect(names.has('weights_bytes')).toBe(true);
    expect(names.has('kv_cache_bytes')).toBe(true);
    expect(names.has('memory_bound_tps')).toBe(true);
    expect(names.has('compute_bound_tps')).toBe(true);
    expect(names.has('ttft_seconds')).toBe(true);
  });
});
