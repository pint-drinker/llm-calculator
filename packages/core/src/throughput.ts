import type {
  AttentionLayer,
  GPU,
  InferenceConfig,
  ThroughputEstimate,
  WeightQuant,
} from './types.js';
import type { MemoryComputation } from './memory.js';

const TP_COMM_EFFICIENCY = 0.85;

function effectiveTflops(gpu: GPU, weight_quant: WeightQuant, tp: number): number {
  let base = gpu.fp16_tflops;
  if (weight_quant === 'fp8' && gpu.fp8_tflops) {
    base = gpu.fp8_tflops;
  } else if (
    (weight_quant === 'awq_int4' ||
      weight_quant === 'gptq_int4' ||
      weight_quant === 'q4_k_m' ||
      weight_quant === 'q3_k_m') &&
    gpu.int4_tflops
  ) {
    base = gpu.int4_tflops;
  }
  const eff = tp === 1 ? 1 : TP_COMM_EFFICIENCY;
  return base * tp * eff;
}

function effectiveBandwidth(gpu: GPU, tp: number): number {
  const eff = tp === 1 ? 1 : TP_COMM_EFFICIENCY;
  return gpu.memory_bandwidth_gbs * tp * eff;
}

function attentionFlops(layers: AttentionLayer[], newTokens: number, totalContext: number): number {
  let flops = 0;
  for (const layer of layers) {
    if (layer.kind === 'full') {
      flops += 2 * newTokens * totalContext * layer.n_kv_heads * layer.head_dim;
    }
  }
  return flops;
}

/**
 * Compute prefill (TTFT) latency using the roofline model.
 * Accounts for linear matmul FLOPs, quadratic attention FLOPs for full-attention
 * layers, and memory-bandwidth floor for weight loading.
 *
 * @param promptTokens - Number of new tokens to prefill
 * @param cachedTokens - Tokens already in KV cache (cross-attention cost only)
 */
export function computePrefillTime(
  config: InferenceConfig,
  memory: MemoryComputation,
  gpu: GPU,
  promptTokens: number,
  cachedTokens = 0,
): number {
  const { model, weight_quant, tensor_parallel } = config;
  const tp = tensor_parallel;
  const effFlops = effectiveTflops(gpu, weight_quant, tp);
  const effBw = effectiveBandwidth(gpu, tp);

  const linearFlops = promptTokens * 2 * (model.active_params ?? model.params);
  const totalContext = promptTokens + cachedTokens;
  const attnFlops = attentionFlops(model.layers, promptTokens, totalContext);

  const computeTime = (linearFlops + attnFlops) / (effFlops * 1e12);
  const memoryTime = memory.raw.weights_bytes / (effBw * 1e9);

  return Math.max(computeTime, memoryTime);
}

export function computeThroughput(
  config: InferenceConfig,
  memory: MemoryComputation,
  gpu: GPU,
): ThroughputEstimate {
  const { model, weight_quant, context_length, tensor_parallel } = config;
  const tp = tensor_parallel;

  const bytesPerToken =
    memory.raw.weights_bytes +
    memory.raw.kv_cache_bytes / Math.max(1, context_length);
  const effBw = effectiveBandwidth(gpu, tp);
  const memory_bound_tps = (effBw * 1e9) / bytesPerToken;

  const flopsPerToken = 2 * (model.active_params ?? model.params);
  const effFlops = effectiveTflops(gpu, weight_quant, tp);
  const compute_bound_tps = (effFlops * 1e12) / flopsPerToken;

  const estimated_tps = Math.min(memory_bound_tps, compute_bound_tps);
  const bottleneck: 'memory' | 'compute' =
    memory_bound_tps < compute_bound_tps ? 'memory' : 'compute';

  const ttft_seconds = computePrefillTime(config, memory, gpu, context_length);

  return {
    memory_bound_tps,
    compute_bound_tps,
    estimated_tps,
    bottleneck,
    ttft_seconds,
  };
}

export const THROUGHPUT_CONSTANTS = {
  TP_COMM_EFFICIENCY,
};

export { effectiveBandwidth, effectiveTflops };
