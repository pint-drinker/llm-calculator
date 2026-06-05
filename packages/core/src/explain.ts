import type {
  ExplainStep,
  ExplainTrace,
  GPU,
  InferenceConfig,
} from './types.js';
import { bytesPerParam, kvBytesPerElement } from './quantization.js';
import { MEMORY_CONSTANTS, computeMemory } from './memory.js';
import {
  THROUGHPUT_CONSTANTS,
  effectiveBandwidth,
  effectiveTflops,
  engineEfficiency,
} from './throughput.js';

const GB = MEMORY_CONSTANTS.GB;

function num(n: number, digits = 6): string {
  if (!Number.isFinite(n)) return String(n);
  if (Math.abs(n) >= 1e9) return n.toExponential(3);
  return Number(n.toPrecision(digits)).toString();
}

export function explain(config: InferenceConfig, gpu: GPU): ExplainTrace {
  const { model, weight_quant, kv_quant, context_length, batch_size, tensor_parallel, engine } = config;
  const tp = tensor_parallel;
  const engineEff = engineEfficiency(engine, weight_quant, model);
  const steps: ExplainStep[] = [];

  const bpp = bytesPerParam(weight_quant);
  const weights_bytes = model.params * bpp;
  steps.push({
    name: 'weights_bytes',
    formula: 'params × bytes_per_param(weight_quant)',
    inputs: { params: model.params, weight_quant, bytes_per_param: bpp },
    substituted: `${num(model.params)} × ${bpp}`,
    result: weights_bytes,
    units: 'bytes',
  });

  const kvBpe = kvBytesPerElement(kv_quant);
  let kvPerToken = 0;
  let fullLayers = 0;
  for (const layer of model.layers) {
    if (layer.kind === 'full') {
      kvPerToken += 2 * layer.n_kv_heads * layer.head_dim * kvBpe;
      fullLayers++;
    }
  }
  steps.push({
    name: 'kv_bytes_per_token',
    formula: 'Σ_full_layers(2 × n_kv_heads × head_dim × kv_bytes_per_element)',
    inputs: {
      full_layer_count: fullLayers,
      kv_quant,
      kv_bytes_per_element: kvBpe,
    },
    substituted: `Σ over ${fullLayers} full-attention layers`,
    result: kvPerToken,
    units: 'bytes/token',
  });

  const kv_cache_bytes = batch_size * context_length * kvPerToken;
  steps.push({
    name: 'kv_cache_bytes',
    formula: 'batch × context × kv_bytes_per_token',
    inputs: { batch_size, context_length, kv_bytes_per_token: kvPerToken },
    substituted: `${batch_size} × ${num(context_length)} × ${num(kvPerToken)}`,
    result: kv_cache_bytes,
    units: 'bytes',
  });

  let linearTotal = 0;
  let linearLayers = 0;
  for (const layer of model.layers) {
    if (layer.kind === 'linear') {
      linearTotal += layer.state_size_bytes;
      linearLayers++;
    }
  }
  const linear_state_bytes = batch_size * linearTotal;
  steps.push({
    name: 'linear_state_bytes',
    formula: 'batch × Σ_linear_layers(state_size_bytes)',
    inputs: { batch_size, linear_layer_count: linearLayers },
    substituted: `${batch_size} × ${num(linearTotal)} (context-independent)`,
    result: linear_state_bytes,
    units: 'bytes',
  });

  const chunk = Math.min(context_length, MEMORY_CONSTANTS.ACTIVATION_CHUNK_TOKENS);
  const activations_bytes =
    batch_size * chunk * model.hidden_dim * MEMORY_CONSTANTS.ACTIVATION_BYTES_PER_HIDDEN;
  steps.push({
    name: 'activations_bytes',
    formula: 'batch × min(context, 4096) × hidden_dim × 8',
    inputs: { batch_size, chunk_tokens: chunk, hidden_dim: model.hidden_dim },
    substituted: `${batch_size} × ${chunk} × ${model.hidden_dim} × 8`,
    result: activations_bytes,
    units: 'bytes',
  });

  const framework_overhead_bytes = MEMORY_CONSTANTS.FRAMEWORK_OVERHEAD_GB * GB;
  steps.push({
    name: 'framework_overhead_bytes',
    formula: `${MEMORY_CONSTANTS.FRAMEWORK_OVERHEAD_GB} GB constant`,
    inputs: {},
    substituted: `${MEMORY_CONSTANTS.FRAMEWORK_OVERHEAD_GB} × 2^30`,
    result: framework_overhead_bytes,
    units: 'bytes',
  });

  const shardable = weights_bytes + kv_cache_bytes + linear_state_bytes + activations_bytes;
  const totalBytes = shardable + framework_overhead_bytes;
  steps.push({
    name: 'total_bytes',
    formula: 'weights + kv + linear + activations + overhead',
    inputs: {
      weights_bytes,
      kv_cache_bytes,
      linear_state_bytes,
      activations_bytes,
      framework_overhead_bytes,
    },
    substituted: `${num(weights_bytes)} + ${num(kv_cache_bytes)} + ${num(linear_state_bytes)} + ${num(activations_bytes)} + ${num(framework_overhead_bytes)}`,
    result: totalBytes,
    units: 'bytes',
  });

  const perGpuBytes = shardable / tp + framework_overhead_bytes;
  steps.push({
    name: 'per_gpu_bytes',
    formula: '(weights + kv + linear + activations) / TP + overhead',
    inputs: {
      shardable_bytes: shardable,
      tensor_parallel: tp,
      framework_overhead_bytes,
    },
    substituted: `${num(shardable)} / ${tp} + ${num(framework_overhead_bytes)}`,
    result: perGpuBytes,
    units: 'bytes',
  });

  steps.push({
    name: 'per_gpu_gb',
    formula: 'per_gpu_bytes / 2^30',
    inputs: { per_gpu_bytes: perGpuBytes, gb: GB },
    substituted: `${num(perGpuBytes)} / ${GB}`,
    result: perGpuBytes / GB,
    units: 'GB',
  });

  // throughput section
  const activeParams = model.active_params ?? model.params;
  const activeWeightsBytes = activeParams * bpp;
  const bytesPerToken = activeWeightsBytes + kv_cache_bytes / Math.max(1, context_length);
  steps.push({
    name: 'bytes_per_decode_token',
    formula: 'active_weights_bytes + kv_cache_bytes / context',
    inputs: { active_params: activeParams, bytes_per_param: bpp, active_weights_bytes: activeWeightsBytes, kv_cache_bytes, context_length },
    substituted: `${num(activeWeightsBytes)} + ${num(kv_cache_bytes)} / ${num(context_length)}`,
    result: bytesPerToken,
    units: 'bytes/token',
  });

  const effBw = effectiveBandwidth(gpu, tp);
  steps.push({
    name: 'effective_bandwidth',
    formula: 'gpu.memory_bandwidth × TP × (TP==1 ? 1 : 0.85)',
    inputs: {
      memory_bandwidth_gbs: gpu.memory_bandwidth_gbs,
      tensor_parallel: tp,
      tp_efficiency: tp === 1 ? 1 : THROUGHPUT_CONSTANTS.TP_COMM_EFFICIENCY,
    },
    substituted: `${gpu.memory_bandwidth_gbs} × ${tp} × ${tp === 1 ? 1 : THROUGHPUT_CONSTANTS.TP_COMM_EFFICIENCY}`,
    result: effBw,
    units: 'GB/s',
  });

  const isMoE = (model.active_params ?? model.params) < model.params;
  steps.push({
    name: 'engine_decode_efficiency',
    formula: 'engine == llama.cpp ? quant_efficiency × (MoE ? 0.52 : 1) : 1',
    inputs: {
      engine: engine ?? 'sglang',
      weight_quant,
      moe: isMoE ? 'yes' : 'no',
    },
    substituted:
      engine === 'llama_cpp'
        ? `${num(THROUGHPUT_CONSTANTS.LLAMACPP_QUANT_EFFICIENCY[weight_quant])} × ${isMoE ? THROUGHPUT_CONSTANTS.LLAMACPP_MOE_EFFICIENCY : 1}`
        : '1 (sglang baseline)',
    result: engineEff.decode,
    units: '×',
  });

  const memTps = (effBw * engineEff.decode * 1e9) / bytesPerToken;
  steps.push({
    name: 'memory_bound_tps',
    formula: 'effective_bandwidth × engine_efficiency × 1e9 / bytes_per_decode_token',
    inputs: {
      effective_bandwidth_gbs: effBw,
      engine_decode_efficiency: engineEff.decode,
      bytes_per_decode_token: bytesPerToken,
    },
    substituted: `${num(effBw)} × ${num(engineEff.decode)} × 1e9 / ${num(bytesPerToken)}`,
    result: memTps,
    units: 'tokens/s',
  });

  const flopsPerToken = 2 * (model.active_params ?? model.params);
  steps.push({
    name: 'flops_per_token',
    formula: '2 × (active_params ?? params)',
    inputs: {
      params: model.params,
      active_params: model.active_params ?? model.params,
    },
    substituted: `2 × ${num(model.active_params ?? model.params)}`,
    result: flopsPerToken,
    units: 'FLOPs/token',
  });

  const effFlops = effectiveTflops(gpu, weight_quant, tp);
  steps.push({
    name: 'effective_tflops',
    formula: 'gpu.tflops_for(weight_quant) × TP × (TP==1 ? 1 : 0.85)',
    inputs: {
      fp16_tflops: gpu.fp16_tflops,
      fp8_tflops: gpu.fp8_tflops ?? 0,
      int4_tflops: gpu.int4_tflops ?? 0,
      tensor_parallel: tp,
    },
    substituted: `tflops × ${tp} × ${tp === 1 ? 1 : THROUGHPUT_CONSTANTS.TP_COMM_EFFICIENCY}`,
    result: effFlops,
    units: 'TFLOPs/s',
  });

  const compTps = (effFlops * engineEff.decode * 1e12) / flopsPerToken;
  steps.push({
    name: 'compute_bound_tps',
    formula: 'effective_tflops × engine_efficiency × 1e12 / flops_per_token',
    inputs: {
      effective_tflops: effFlops,
      engine_decode_efficiency: engineEff.decode,
      flops_per_token: flopsPerToken,
    },
    substituted: `${num(effFlops)} × ${num(engineEff.decode)} × 1e12 / ${num(flopsPerToken)}`,
    result: compTps,
    units: 'tokens/s',
  });

  steps.push({
    name: 'estimated_tps',
    formula: 'min(memory_bound_tps, compute_bound_tps)',
    inputs: { memory_bound_tps: memTps, compute_bound_tps: compTps },
    substituted: `min(${num(memTps)}, ${num(compTps)})`,
    result: Math.min(memTps, compTps),
    units: 'tokens/s',
  });

  const linearFlops = context_length * flopsPerToken;
  steps.push({
    name: 'prefill_linear_flops',
    formula: 'context × flops_per_token',
    inputs: { context_length, flops_per_token: flopsPerToken },
    substituted: `${num(context_length)} × ${num(flopsPerToken)}`,
    result: linearFlops,
    units: 'FLOPs',
  });

  let attnFlops = 0;
  for (const layer of model.layers) {
    if (layer.kind === 'full') {
      attnFlops += 2 * context_length * context_length * layer.n_kv_heads * layer.head_dim;
    }
  }
  steps.push({
    name: 'prefill_attention_flops',
    formula: 'Σ_full_layers(2 × context² × n_kv_heads × head_dim)',
    inputs: { context_length, full_layer_count: fullLayers },
    substituted: `2 × ${num(context_length)}² × (Σ n_kv_heads×head_dim over ${fullLayers} layers)`,
    result: attnFlops,
    units: 'FLOPs',
  });

  const prefillComputeS = (linearFlops + attnFlops) / (effFlops * engineEff.prefill * 1e12);
  steps.push({
    name: 'prefill_compute_time',
    formula: '(linear_flops + attention_flops) / (effective_tflops × engine_prefill_efficiency × 1e12)',
    inputs: {
      linear_flops: linearFlops,
      attention_flops: attnFlops,
      effective_tflops: effFlops,
      engine_prefill_efficiency: engineEff.prefill,
    },
    substituted: `(${num(linearFlops)} + ${num(attnFlops)}) / (${num(effFlops)} × ${num(engineEff.prefill)} × 1e12)`,
    result: prefillComputeS,
    units: 's',
  });

  const prefillMemoryS = activeWeightsBytes / (effBw * engineEff.prefill * 1e9);
  steps.push({
    name: 'prefill_memory_floor',
    formula: 'active_weights_bytes / (effective_bandwidth × engine_prefill_efficiency × 1e9)',
    inputs: {
      active_weights_bytes: activeWeightsBytes,
      effective_bandwidth_gbs: effBw,
      engine_prefill_efficiency: engineEff.prefill,
    },
    substituted: `${num(activeWeightsBytes)} / (${num(effBw)} × ${num(engineEff.prefill)} × 1e9)`,
    result: prefillMemoryS,
    units: 's',
  });

  const ttft = Math.max(prefillComputeS, prefillMemoryS);
  steps.push({
    name: 'ttft_seconds',
    formula: 'max(prefill_compute_time, prefill_memory_floor)',
    inputs: { prefill_compute_time: prefillComputeS, prefill_memory_floor: prefillMemoryS },
    substituted: `max(${num(prefillComputeS)}, ${num(prefillMemoryS)})`,
    result: ttft,
    units: 's',
  });

  // sanity reference to keep computeMemory imported (parity check)
  void computeMemory;
  return { steps };
}
