import type { CalculationResult, GPU, InferenceConfig } from './types.js';
import { computeMemory } from './memory.js';
import { computeThroughput } from './throughput.js';

const APPLE_SILICON_NAMES = /M[234]\s*Ultra|Apple Silicon/i;

export function calculate(config: InferenceConfig, gpu: GPU): CalculationResult {
  const memory = computeMemory(config);
  const throughput = computeThroughput(config, memory, gpu);

  const warnings = [...memory.warnings];
  if (APPLE_SILICON_NAMES.test(gpu.name)) {
    warnings.push(
      'Apple Silicon: bandwidth modeled at peak; sustained throughput typically 60–80% of theoretical, and unified memory must be shared with the OS.',
    );
  }
  if (config.tensor_parallel > 1 && gpu.name.toLowerCase().includes('apple')) {
    warnings.push('Tensor parallelism is not applicable to single Apple Silicon devices.');
  }

  const usable_fraction = gpu.usable_memory_fraction ?? 1.0;
  const usable_vram_gb = gpu.vram_gb * usable_fraction;
  const fits = memory.breakdown.per_gpu_gb <= gpu.vram_gb;
  const fits_usable = memory.breakdown.per_gpu_gb <= usable_vram_gb;
  const utilization_pct = (memory.breakdown.per_gpu_gb / gpu.vram_gb) * 100;

  if (fits && !fits_usable) {
    warnings.push(
      `Exceeds usable VRAM (~${(usable_fraction * 100).toFixed(0)}% of ${gpu.name}'s ${gpu.vram_gb.toFixed(0)} GB). Physically fits but may degrade or OOM under real OS/driver overhead.`,
    );
  }

  return {
    memory: memory.breakdown,
    throughput,
    fits,
    fits_usable,
    usable_vram_gb,
    utilization_pct,
    warnings,
  };
}
