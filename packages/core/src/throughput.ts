import type {
  AttentionLayer,
  GPU,
  InferenceConfig,
  InferenceEngine,
  ModelConfig,
  ThroughputEstimate,
  WeightQuant,
} from './types.js';
import type { MemoryComputation } from './memory.js';
import { bytesPerParam } from './quantization.js';

const TP_COMM_EFFICIENCY = 0.85;

// llama.cpp achieves a lower fraction of theoretical memory bandwidth than
// sglang-style serving stacks. Calibrated against measured decode throughput
// on DGX Spark (June 2026): Qwen3.6-27B q8_0 ≈ 7.5 t/s, q4_k_m ≈ 12 t/s,
// Qwen3.6-35B-A3B q4_k_m ≈ 57 t/s. Lower-bit GGUF quants pay more dequant
// kernel overhead per byte, so efficiency drops as bits shrink.
const LLAMACPP_QUANT_EFFICIENCY: Record<WeightQuant, number> = {
  bf16: 0.8,
  fp16: 0.8,
  q8_0: 0.79, // measured (7.5 / 9.51)
  fp8: 0.74,
  int8: 0.74,
  q5_k_m: 0.72, // interpolated
  awq_int4: 0.67,
  gptq_int4: 0.67,
  q4_k_m: 0.67, // measured (12 / 17.97)
  q3_k_m: 0.62, // extrapolated
  q2_k: 0.56, // extrapolated
};
// llama.cpp's expert matmuls run as many small kernels per token, so MoE
// models land well below their active-param roofline (measured 0.35 overall
// for Qwen3.6-35B-A3B q4_k_m → 0.35 / 0.67 ≈ 0.52).
const LLAMACPP_MOE_EFFICIENCY = 0.52;
// Prefill default: ~2× slower than sglang. Uncalibrated — tune with real
// llama.cpp TTFT measurements when available.
const LLAMACPP_PREFILL_COMPUTE_EFFICIENCY = 0.5;

function isMoE(model: ModelConfig): boolean {
  return model.active_params != null && model.active_params < model.params;
}

export interface EngineEfficiency {
  decode: number;
  prefill: number;
}

export function engineEfficiency(
  engine: InferenceEngine | undefined,
  weight_quant: WeightQuant,
  model: ModelConfig,
): EngineEfficiency {
  if (engine !== 'llama_cpp') {
    return { decode: 1, prefill: 1 };
  }
  const moe = isMoE(model) ? LLAMACPP_MOE_EFFICIENCY : 1;
  return {
    decode: LLAMACPP_QUANT_EFFICIENCY[weight_quant] * moe,
    prefill: LLAMACPP_PREFILL_COMPUTE_EFFICIENCY * moe,
  };
}

function effectiveTflops(gpu: GPU, weight_quant: WeightQuant, tp: number): number {
  let base = gpu.fp16_tflops;
  if (weight_quant === 'fp8' && gpu.fp8_tflops) {
    base = gpu.fp8_tflops;
  } else if (
    (weight_quant === 'awq_int4' ||
      weight_quant === 'gptq_int4' ||
      weight_quant === 'q4_k_m' ||
      weight_quant === 'q3_k_m' ||
      weight_quant === 'q2_k') &&
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
  const { model, weight_quant, tensor_parallel, engine } = config;
  const tp = tensor_parallel;
  const engineEff = engineEfficiency(engine, weight_quant, model);
  const effFlops = effectiveTflops(gpu, weight_quant, tp) * engineEff.prefill;
  const effBw = effectiveBandwidth(gpu, tp) * engineEff.prefill;

  const linearFlops = promptTokens * 2 * (model.active_params ?? model.params);
  const totalContext = promptTokens + cachedTokens;
  const attnFlops = attentionFlops(model.layers, promptTokens, totalContext);

  const activeWeightsBytes = (model.active_params ?? model.params) * bytesPerParam(weight_quant);
  const computeTime = (linearFlops + attnFlops) / (effFlops * 1e12);
  const memoryTime = activeWeightsBytes / (effBw * 1e9);

  return Math.max(computeTime, memoryTime);
}

export function computeThroughput(
  config: InferenceConfig,
  memory: MemoryComputation,
  gpu: GPU,
): ThroughputEstimate {
  const { model, weight_quant, context_length, tensor_parallel, engine } = config;
  const tp = tensor_parallel;
  const engineEff = engineEfficiency(engine, weight_quant, model);

  const activeWeightsBytes = (model.active_params ?? model.params) * bytesPerParam(weight_quant);
  const bytesPerToken =
    activeWeightsBytes +
    memory.raw.kv_cache_bytes / Math.max(1, context_length);
  const effBw = effectiveBandwidth(gpu, tp) * engineEff.decode;
  const memory_bound_tps = (effBw * 1e9) / bytesPerToken;

  const flopsPerToken = 2 * (model.active_params ?? model.params);
  const effFlops = effectiveTflops(gpu, weight_quant, tp) * engineEff.decode;
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
  LLAMACPP_QUANT_EFFICIENCY,
  LLAMACPP_MOE_EFFICIENCY,
  LLAMACPP_PREFILL_COMPUTE_EFFICIENCY,
};

export { effectiveBandwidth, effectiveTflops };
