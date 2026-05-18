import type { AttentionLayer, ModelConfig } from '../types.js';
import {
  type ArchitectureAdapter,
  type AdapterParseResult,
  type HFConfig,
  paramsFromMetadata,
} from './types.js';

const SUPPORTED_ARCHITECTURES = [
  'Qwen3NextForCausalLM',
  'Qwen3-NextForCausalLM',
  'Qwen3_5ForCausalLM',
  'Qwen3_5ForConditionalGeneration',
  'Qwen3_6ForCausalLM',
  'Qwen3_6ForConditionalGeneration',
];

const SUPPORTED_MODEL_TYPE_PREFIXES = ['qwen3_next', 'qwen3_5', 'qwen3_6'];

function flattenTextConfig(config: HFConfig): HFConfig {
  // Multimodal Qwen3.5/3.6 nest text-decoder fields under `text_config`.
  // Top-level fields take precedence so we don't clobber e.g. architectures.
  const text = (config as Record<string, unknown>).text_config;
  if (text && typeof text === 'object') {
    return { ...(text as HFConfig), ...config };
  }
  return config;
}

function modelTypeMatches(mt: unknown): boolean {
  return (
    typeof mt === 'string' &&
    SUPPORTED_MODEL_TYPE_PREFIXES.some((p) => mt.startsWith(p))
  );
}

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

function architectureLabel(modelType: unknown): string {
  if (typeof modelType === 'string') {
    if (modelType.startsWith('qwen3_5')) return 'qwen3.5';
    if (modelType.startsWith('qwen3_6')) return 'qwen3.6';
  }
  return 'qwen3-next';
}

export const qwen3NextAdapter: ArchitectureAdapter = {
  name: 'qwen3-next',
  matches(config) {
    const arches = config.architectures ?? [];
    if (arches.some((a) => SUPPORTED_ARCHITECTURES.includes(a))) return true;
    if (modelTypeMatches(config.model_type)) return true;
    const text = (config as Record<string, unknown>).text_config as HFConfig | undefined;
    if (text) {
      const nestedArches = text.architectures ?? [];
      if (nestedArches.some((a) => SUPPORTED_ARCHITECTURES.includes(a))) return true;
      if (modelTypeMatches(text.model_type)) return true;
    }
    return false;
  },
  parse(modelId, rawConfig, metadata): AdapterParseResult {
    const warnings: string[] = [];
    const config = flattenTextConfig(rawConfig);
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
        architecture: architectureLabel(config.model_type),
      },
      warnings,
    };
  },
};
