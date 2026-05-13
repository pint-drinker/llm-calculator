import type {
  AttentionLayer,
  InferenceConfig,
  MemoryBreakdown,
  ModelConfig,
} from './types.js';
import { bytesPerParam, kvBytesPerElement } from './quantization.js';

const GB = 1e9;
const FRAMEWORK_OVERHEAD_GB = 1;
const ACTIVATION_CHUNK_TOKENS = 4096;
const ACTIVATION_BYTES_PER_HIDDEN = 8;

export interface MemoryComputation {
  breakdown: MemoryBreakdown;
  warnings: string[];
  raw: {
    weights_bytes: number;
    kv_cache_bytes: number;
    linear_state_bytes: number;
    activations_bytes: number;
    framework_overhead_bytes: number;
  };
}

export function computeMemory(config: InferenceConfig): MemoryComputation {
  const { model, weight_quant, kv_quant, context_length, batch_size, tensor_parallel } = config;
  const warnings: string[] = [];

  const weights_bytes = model.params * bytesPerParam(weight_quant);

  const kvBpe = kvBytesPerElement(kv_quant);
  let kvPerTokenBytes = 0;
  let linearStateBytes = 0;
  let hasLinear = false;
  for (const layer of model.layers) {
    if (layer.kind === 'full') {
      kvPerTokenBytes += 2 * layer.n_kv_heads * layer.head_dim * kvBpe;
    } else if (layer.kind === 'linear') {
      linearStateBytes += layer.state_size_bytes;
      hasLinear = true;
    }
  }
  const kv_cache_bytes = batch_size * context_length * kvPerTokenBytes;
  const linear_state_bytes = batch_size * linearStateBytes;

  const activations_bytes =
    batch_size *
    Math.min(context_length, ACTIVATION_CHUNK_TOKENS) *
    model.hidden_dim *
    ACTIVATION_BYTES_PER_HIDDEN;

  const framework_overhead_bytes = FRAMEWORK_OVERHEAD_GB * GB;

  const tp = tensor_parallel;
  const shardableBytes =
    weights_bytes + kv_cache_bytes + linear_state_bytes + activations_bytes;
  const totalBytes = shardableBytes + framework_overhead_bytes;
  const perGpuBytes = shardableBytes / tp + framework_overhead_bytes;

  if (kv_cache_bytes > weights_bytes && kvPerTokenBytes > 0) {
    warnings.push('KV cache exceeds weights at this context');
  }
  if (hasLinear && tp > 1) {
    warnings.push('Linear attention state ignored for TP > 1');
  }
  if (kvPerTokenBytes === 0 && !hasLinear) {
    warnings.push('Model has no full or linear attention layers — KV cache is zero');
  }

  return {
    breakdown: {
      weights_gb: weights_bytes / GB,
      kv_cache_gb: kv_cache_bytes / GB,
      linear_state_gb: linear_state_bytes / GB,
      activations_gb: activations_bytes / GB,
      framework_overhead_gb: FRAMEWORK_OVERHEAD_GB,
      total_gb: totalBytes / GB,
      per_gpu_gb: perGpuBytes / GB,
    },
    warnings,
    raw: {
      weights_bytes,
      kv_cache_bytes,
      linear_state_bytes,
      activations_bytes,
      framework_overhead_bytes,
    },
  };
}

export function kvBytesPerToken(model: ModelConfig, kvBpe: number): number {
  let bytes = 0;
  for (const layer of model.layers as AttentionLayer[]) {
    if (layer.kind === 'full') {
      bytes += 2 * layer.n_kv_heads * layer.head_dim * kvBpe;
    }
  }
  return bytes;
}

export const MEMORY_CONSTANTS = {
  GB,
  FRAMEWORK_OVERHEAD_GB,
  ACTIVATION_CHUNK_TOKENS,
  ACTIVATION_BYTES_PER_HIDDEN,
};
