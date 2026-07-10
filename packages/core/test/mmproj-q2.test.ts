import { describe, expect, it } from 'vitest';
import {
  ENGINE_WEIGHT_QUANTS,
  bytesPerParam,
  calculate,
  computeMemory,
  estimateMmprojBytes,
  findGpu,
  findMaxContext,
  findModel,
  recommendHardware,
} from '../src/index.js';
import type { GPU, InferenceConfig } from '../src/index.js';

describe('mmproj memory estimate', () => {
  const model = findModel('Llama-3.1-8B')!;
  const h100 = findGpu('H100 80GB')!;

  function cfg(overrides: Partial<InferenceConfig> = {}): InferenceConfig {
    return {
      model,
      weight_quant: 'bf16',
      kv_quant: 'bf16',
      context_length: 8192,
      batch_size: 1,
      tensor_parallel: 1,
      ...overrides,
    };
  }

  it('adds 15% of params at BF16 precision and divides by tensor parallelism', () => {
    const base = calculate(cfg({ tensor_parallel: 2 }), h100);
    const withMmproj = calculate(cfg({ tensor_parallel: 2, include_mmproj: true }), h100);
    const expectedPerGpuGb = estimateMmprojBytes(model, true) / 1e9 / 2;

    expect(withMmproj.memory.mmproj_gb).toBeCloseTo(expectedPerGpuGb * 2, 5);
    expect(withMmproj.memory.per_gpu_gb - base.memory.per_gpu_gb).toBeCloseTo(
      expectedPerGpuGb,
      5,
    );
  });

  it('keeps the mmproj estimate independent of selected weight quantization', () => {
    const bf16 = computeMemory(cfg({ weight_quant: 'bf16', include_mmproj: true }));
    const q4 = computeMemory(cfg({ weight_quant: 'q4_k_m', include_mmproj: true }));

    expect(q4.breakdown.weights_gb).toBeLessThan(bf16.breakdown.weights_gb);
    expect(q4.breakdown.mmproj_gb).toBeCloseTo(bf16.breakdown.mmproj_gb, 8);
    expect(q4.breakdown.mmproj_gb).toBeCloseTo(
      (model.params * 0.15 * bytesPerParam('bf16')) / 1e9,
      5,
    );
  });

  it('can push a borderline configuration over physical capacity', () => {
    const base = calculate(cfg(), h100);
    const mmprojDeltaGb = estimateMmprojBytes(model, true) / 1e9;
    const boundaryGpu: GPU = {
      name: 'Boundary GPU',
      vram_gb: base.memory.per_gpu_gb + mmprojDeltaGb / 2,
      memory_bandwidth_gbs: 1000,
      fp16_tflops: 100,
    };

    expect(calculate(cfg(), boundaryGpu).fits).toBe(true);
    const withMmproj = calculate(cfg({ include_mmproj: true }), boundaryGpu);
    expect(withMmproj.fits).toBe(false);
    expect(withMmproj.warnings.some((w) => /mmproj/i.test(w))).toBe(true);
  });

  it('reduces max context and affects hardware recommendations', () => {
    const without = findMaxContext({
      model,
      weight_quant: 'bf16',
      kv_quant: 'bf16',
      gpu: h100,
      tensor_parallel: 1,
    });
    const withMmproj = findMaxContext({
      model,
      weight_quant: 'bf16',
      kv_quant: 'bf16',
      gpu: h100,
      tensor_parallel: 1,
      include_mmproj: true,
    });

    expect(withMmproj.max_context).toBeLessThan(without.max_context);

    const base = calculate(cfg(), h100);
    const mmprojDeltaGb = estimateMmprojBytes(model, true) / 1e9;
    const boundaryGpu: GPU = {
      name: 'Recommendation Boundary GPU',
      vram_gb: base.memory.per_gpu_gb + mmprojDeltaGb / 2,
      memory_bandwidth_gbs: 1000,
      fp16_tflops: 100,
    };

    const withoutRecommendation = recommendHardware({
      model,
      weight_quant: 'bf16',
      kv_quant: 'bf16',
      context_length: 8192,
      gpus: [boundaryGpu],
    }).find((c) => c.tensor_parallel === 1)!;
    const withRecommendation = recommendHardware({
      model,
      weight_quant: 'bf16',
      kv_quant: 'bf16',
      context_length: 8192,
      include_mmproj: true,
      gpus: [boundaryGpu],
    }).find((c) => c.tensor_parallel === 1)!;

    expect(withoutRecommendation.fits).toBe(true);
    expect(withRecommendation.fits).toBe(false);
  });
});

describe('llama.cpp q2_k quantization', () => {
  it('is available only for llama.cpp and uses fewer bytes than q3_k_m', () => {
    expect(ENGINE_WEIGHT_QUANTS.llama_cpp).toContain('q2_k');
    expect(ENGINE_WEIGHT_QUANTS.sglang).not.toContain('q2_k');
    expect(bytesPerParam('q2_k')).toBeLessThan(bytesPerParam('q3_k_m'));
  });
});
