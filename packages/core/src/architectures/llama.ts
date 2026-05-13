import type { ModelConfig, AttentionLayer } from '../types.js';
import {
  type ArchitectureAdapter,
  type HFConfig,
  type HFModelInfo,
  type AdapterParseResult,
  estimateDenseGqaParams,
  paramsFromMetadata,
} from './types.js';

const SUPPORTED_ARCHES = [
  'LlamaForCausalLM',
  'Qwen2ForCausalLM',
  'Qwen2_5ForCausalLM',
  'MistralForCausalLM',
  'Phi3ForCausalLM',
  'Phi4ForCausalLM',
  'GemmaForCausalLM',
  'Gemma2ForCausalLM',
  'Qwen3ForCausalLM',
];

export const llamaAdapter: ArchitectureAdapter = {
  name: 'llama',
  matches(config) {
    const arches = config.architectures ?? [];
    return arches.some((a) => SUPPORTED_ARCHES.includes(a));
  },
  parse(modelId, config, metadata): AdapterParseResult {
    const warnings: string[] = [];
    const hidden = config.hidden_size ?? 0;
    const layers_n = config.num_hidden_layers ?? 0;
    const heads = config.num_attention_heads ?? Math.max(1, hidden / 128);
    const kvHeads = config.num_key_value_heads ?? heads;
    const headDim = config.head_dim ?? Math.floor(hidden / Math.max(1, heads));
    const vocab = config.vocab_size ?? 0;

    if (!hidden || !layers_n || !vocab) {
      warnings.push('Config missing one of hidden_size/num_hidden_layers/vocab_size');
    }

    const layers: AttentionLayer[] = Array.from({ length: layers_n }, () => ({
      kind: 'full' as const,
      n_kv_heads: kvHeads,
      head_dim: headDim,
    }));

    let params = paramsFromMetadata(metadata);
    if (!params) {
      params = estimateDenseGqaParams(config);
      warnings.push(
        'safetensors.total missing — param count estimated from architecture (may differ by 5–10%)',
      );
    }

    const model: ModelConfig = {
      name: modelId,
      params,
      hidden_dim: hidden,
      vocab_size: vocab,
      layers,
      architecture: 'llama',
    };
    return { model, warnings };
  },
};
