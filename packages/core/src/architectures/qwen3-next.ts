import type { AttentionLayer, ModelConfig } from '../types.js';
import {
  type ArchitectureAdapter,
  type AdapterParseResult,
  paramsFromMetadata,
} from './types.js';

const SUPPORTED = ['Qwen3NextForCausalLM', 'Qwen3-NextForCausalLM'];

function deltaNetStateBytes(config: {
  linear_num_value_heads?: number;
  linear_value_head_dim?: number;
  linear_num_key_heads?: number;
  linear_key_head_dim?: number;
  hidden_size?: number;
}): number {
  const vh = config.linear_num_value_heads ?? 0;
  const vd = config.linear_value_head_dim ?? 0;
  const kh = config.linear_num_key_heads ?? vh;
  const kd = config.linear_key_head_dim ?? vd;
  if (vh && vd && kh && kd) {
    // DeltaNet recurrent state: per-head K×V outer product, kept in bf16
    return vh * vd * kh * kd * 2;
  }
  // Fallback: shape based on hidden_size — gives ~1.25 MB at hidden 2048
  const hidden = config.hidden_size ?? 0;
  return hidden * hidden * 0.3125 * 2;
}

export const qwen3NextAdapter: ArchitectureAdapter = {
  name: 'qwen3-next',
  matches(config) {
    const arches = config.architectures ?? [];
    if (arches.some((a) => SUPPORTED.includes(a))) return true;
    if (typeof config.model_type === 'string' && config.model_type.startsWith('qwen3_next')) {
      return true;
    }
    return false;
  },
  parse(modelId, config, metadata): AdapterParseResult {
    const warnings: string[] = [];
    const hidden = config.hidden_size ?? 0;
    const layers_n = config.num_hidden_layers ?? 0;
    const heads = config.num_attention_heads ?? Math.max(1, hidden / 128);
    const kvHeads = config.num_key_value_heads ?? heads;
    const headDim = config.head_dim ?? Math.floor(hidden / Math.max(1, heads));
    const vocab = config.vocab_size ?? 0;
    const stateBytes = deltaNetStateBytes(config);

    const layer_types = config.layer_types;
    const fullIdx = new Set(config.full_attn_layer_idx ?? []);
    const linearIdx = new Set(config.linear_attn_layer_idx ?? []);

    const layers: AttentionLayer[] = [];
    for (let i = 0; i < layers_n; i++) {
      let kind: 'full' | 'linear';
      if (layer_types && layer_types[i]) {
        const t = String(layer_types[i]).toLowerCase();
        kind = t.includes('full') || t === 'attention' ? 'full' : 'linear';
      } else if (fullIdx.size > 0 || linearIdx.size > 0) {
        kind = fullIdx.has(i) ? 'full' : 'linear';
      } else {
        // Default Qwen3-Next pattern: 1 full per group of 4
        kind = i % 4 === 3 ? 'full' : 'linear';
        if (i === 0)
          warnings.push(
            'No explicit layer_types in config — defaulted to 1-full-per-4 hybrid pattern',
          );
      }
      if (kind === 'full') {
        layers.push({ kind: 'full', n_kv_heads: kvHeads, head_dim: headDim });
      } else {
        layers.push({ kind: 'linear', state_size_bytes: stateBytes });
      }
    }

    let params = paramsFromMetadata(metadata);
    if (!params) {
      warnings.push('safetensors.total missing — param count not estimated for Qwen3-Next');
      params = config.num_hidden_layers ? config.num_hidden_layers * hidden * 1e6 : 0;
    }

    return {
      model: {
        name: modelId,
        params,
        hidden_dim: hidden,
        vocab_size: vocab,
        layers,
        architecture: 'qwen3-next',
      },
      warnings,
    };
  },
};
