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

  const fits = memory.breakdown.per_gpu_gb <= gpu.vram_gb;
  const utilization_pct = (memory.breakdown.per_gpu_gb / gpu.vram_gb) * 100;

  return {
    memory: memory.breakdown,
    throughput,
    fits,
    utilization_pct,
    warnings,
  };
}
