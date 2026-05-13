import type { AttentionLayer, ModelConfig } from '../types.js';
import {
  type ArchitectureAdapter,
  type AdapterParseResult,
  paramsFromMetadata,
} from './types.js';

const SUPPORTED = ['JambaForCausalLM'];

export const jambaAdapter: ArchitectureAdapter = {
  name: 'jamba',
  matches(config) {
    const arches = config.architectures ?? [];
    if (arches.some((a) => SUPPORTED.includes(a))) return true;
    if (typeof config.model_type === 'string' && config.model_type === 'jamba') return true;
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
    const dState = config.mamba_d_state ?? 16;
    const expand = config.mamba_expand ?? 2;
    // K+V style mamba state, bf16
    const mambaStateBytes = dState * expand * hidden * 2 * 2;
    const blockTypes = config.layers_block_type ?? [];

    const layers: AttentionLayer[] = [];
    for (let i = 0; i < layers_n; i++) {
      const blockType = String(blockTypes[i] ?? 'mamba').toLowerCase();
      if (blockType.includes('attention')) {
        layers.push({ kind: 'full', n_kv_heads: kvHeads, head_dim: headDim });
      } else if (blockType === 'mamba' || blockType.includes('ssm')) {
        layers.push({ kind: 'linear', state_size_bytes: mambaStateBytes });
      } else {
        layers.push({ kind: 'none' });
      }
    }

    let params = paramsFromMetadata(metadata);
    if (!params) {
      warnings.push('safetensors.total missing — param count unknown for Jamba');
      params = layers_n * hidden * 1e6;
    }

    return {
      model: {
        name: modelId,
        params,
        hidden_dim: hidden,
        vocab_size: vocab,
        layers,
        architecture: 'jamba',
      },
      warnings,
    };
  },
};
