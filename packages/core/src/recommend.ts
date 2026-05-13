import type {
  CalculationResult,
  GPU,
  InferenceConfig,
  KVQuant,
  MemoryBreakdown,
  ModelConfig,
  TensorParallel,
  WeightQuant,
} from './types.js';
import { calculate } from './calculate.js';

export interface MaxContextResult {
  max_context: number;
  breakdown_at_max: MemoryBreakdown;
  warnings: string[];
}

export function findMaxContext(args: {
  model: ModelConfig;
  weight_quant: WeightQuant;
  kv_quant: KVQuant;
  gpu: GPU;
  tensor_parallel: TensorParallel;
  batch_size?: number;
  target_utilization?: number;
}): MaxContextResult {
  const {
    model,
    weight_quant,
    kv_quant,
    gpu,
    tensor_parallel,
    batch_size = 1,
    target_utilization = 0.9,
  } = args;

  const capacityGb = gpu.vram_gb * target_utilization;

  const probe = (context_length: number): CalculationResult =>
    calculate(
      { model, weight_quant, kv_quant, context_length, batch_size, tensor_parallel },
      gpu,
    );

  const baseline = probe(1);
  if (baseline.memory.per_gpu_gb > capacityGb) {
    return {
      max_context: 0,
      breakdown_at_max: baseline.memory,
      warnings: [...baseline.warnings, 'Weights + framework overhead already exceed target utilization'],
    };
  }

  let lo = 1;
  let hi = 1;
  let lastFit = baseline;
  while (hi < 8_388_608) {
    const r = probe(hi);
    if (r.memory.per_gpu_gb > capacityGb) break;
    lastFit = r;
    lo = hi;
    hi *= 2;
  }
  hi = Math.min(hi, 8_388_608);

  while (lo + 1 < hi) {
    const mid = Math.floor((lo + hi) / 2);
    const r = probe(mid);
    if (r.memory.per_gpu_gb <= capacityGb) {
      lo = mid;
      lastFit = r;
    } else {
      hi = mid;
    }
  }

  return {
    max_context: lo,
    breakdown_at_max: lastFit.memory,
    warnings: lastFit.warnings,
  };
}

export interface HardwareCandidate {
  gpu: GPU;
  tensor_parallel: TensorParallel;
  fits: boolean;
  headroom_gb: number;
  per_gpu_gb: number;
  total_vram_gb: number;
  utilization_pct: number;
  est_tps: number;
  bottleneck: 'memory' | 'compute';
  warnings: string[];
}

const TP_OPTIONS: TensorParallel[] = [1, 2, 4, 8];

export function recommendHardware(args: {
  model: ModelConfig;
  weight_quant: WeightQuant;
  kv_quant: KVQuant;
  context_length: number;
  batch_size?: number;
  gpus: GPU[];
}): HardwareCandidate[] {
  const { model, weight_quant, kv_quant, context_length, batch_size = 1, gpus } = args;
  const candidates: HardwareCandidate[] = [];

  for (const gpu of gpus) {
    for (const tp of TP_OPTIONS) {
      const r = calculate(
        {
          model,
          weight_quant,
          kv_quant,
          context_length,
          batch_size,
          tensor_parallel: tp,
        },
        gpu,
      );
      candidates.push({
        gpu,
        tensor_parallel: tp,
        fits: r.fits,
        headroom_gb: gpu.vram_gb - r.memory.per_gpu_gb,
        per_gpu_gb: r.memory.per_gpu_gb,
        total_vram_gb: gpu.vram_gb * tp,
        utilization_pct: r.utilization_pct,
        est_tps: r.throughput.estimated_tps,
        bottleneck: r.throughput.bottleneck,
        warnings: r.warnings,
      });
    }
  }

  candidates.sort((a, b) => {
    if (a.fits !== b.fits) return a.fits ? -1 : 1;
    if (a.fits) {
      if (a.total_vram_gb !== b.total_vram_gb) return a.total_vram_gb - b.total_vram_gb;
      if (a.tensor_parallel !== b.tensor_parallel) return a.tensor_parallel - b.tensor_parallel;
      return b.est_tps - a.est_tps;
    }
    return a.per_gpu_gb - b.per_gpu_gb;
  });

  return candidates;
}
