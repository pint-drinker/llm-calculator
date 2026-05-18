import { describe, expect, it } from 'vitest';
import { calculate, findGpu, findModel } from '../src/index.js';
import type { GPU, InferenceConfig } from '../src/index.js';

function baseConfig(): InferenceConfig {
  return {
    model: findModel('Llama-3.1-8B')!,
    weight_quant: 'bf16',
    kv_quant: 'bf16',
    context_length: 8192,
    batch_size: 1,
    tensor_parallel: 1,
  };
}

describe('usable_memory_fraction', () => {
  it('returns three states: fits, tight, overflow', () => {
    const gpu: GPU = {
      name: 'Test',
      vram_gb: 20,
      memory_bandwidth_gbs: 1000,
      fp16_tflops: 100,
      usable_memory_fraction: 0.5,
    };
    const cfg = baseConfig();
    // Llama-3.1-8B bf16 ≈ 16 GB weights. With small ctx, per_gpu ≈ 17 GB.
    // usable = 20 × 0.5 = 10 GB. Over usable, under physical → tight.
    const r = calculate(cfg, gpu);
    expect(r.fits).toBe(true);
    expect(r.fits_usable).toBe(false);
    expect(r.usable_vram_gb).toBeCloseTo(10, 5);
    expect(r.warnings.some((w) => /usable VRAM/i.test(w))).toBe(true);
  });

  it('defaults to 1.0 when unset (no warning, fits_usable == fits)', () => {
    const gpu: GPU = {
      name: 'NoCap',
      vram_gb: 80,
      memory_bandwidth_gbs: 2000,
      fp16_tflops: 300,
    };
    const r = calculate(baseConfig(), gpu);
    expect(r.fits_usable).toBe(r.fits);
    expect(r.usable_vram_gb).toBeCloseTo(80, 5);
    expect(r.warnings.some((w) => /usable VRAM/i.test(w))).toBe(false);
  });

  it('catalog: H100 uses 0.85, RTX 4090 uses 0.5', () => {
    const h100 = findGpu('H100 80GB')!;
    const rtx = findGpu('RTX 4090')!;
    expect(h100.usable_memory_fraction).toBe(0.85);
    expect(rtx.usable_memory_fraction).toBe(0.5);
  });

  it('catalog: Apple M4 Pro and iPhone A18 both use 0.5', () => {
    expect(findGpu('Apple M4 Pro')!.usable_memory_fraction).toBe(0.5);
    expect(findGpu('Apple A18 Pro (iPhone 16 Pro)')!.usable_memory_fraction).toBe(0.5);
  });
});
