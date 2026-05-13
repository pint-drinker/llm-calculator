import type { AttentionLayer, ModelConfig } from '../types.js';
import {
  type ArchitectureAdapter,
  type AdapterParseResult,
  estimateDenseGqaParams,
  paramsFromMetadata,
} from './types.js';

export const fallbackAdapter: ArchitectureAdapter = {
  name: 'fallback',
  matches() {
    return true;
  },
  parse(modelId, config, metadata): AdapterParseResult {
    const warnings: string[] = [];
    const archName = config.architectures?.[0] ?? config.model_type ?? 'unknown';
    warnings.push(
      `Unknown architecture \`${archName}\`, treating as dense GQA — KV cache estimate may be incorrect.`,
    );

    const hidden = config.hidden_size ?? 4096;
    const layers_n = config.num_hidden_layers ?? 32;
    const heads = config.num_attention_heads ?? Math.max(1, hidden / 128);
    const kvHeads = config.num_key_value_heads ?? heads;
    const headDim = config.head_dim ?? Math.floor(hidden / Math.max(1, heads));
    const vocab = config.vocab_size ?? 32000;

    const layers: AttentionLayer[] = Array.from({ length: layers_n }, () => ({
      kind: 'full' as const,
      n_kv_heads: kvHeads,
      head_dim: headDim,
    }));

    let params = paramsFromMetadata(metadata);
    if (!params) {
      params = estimateDenseGqaParams(config);
      warnings.push('safetensors.total missing — fallback param estimate used');
    }

    return {
      model: {
        name: modelId,
        params,
        hidden_dim: hidden,
        vocab_size: vocab,
        layers,
        architecture: String(archName),
      },
      warnings,
    };
  },
};
