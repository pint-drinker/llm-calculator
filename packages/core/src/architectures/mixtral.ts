import type { AttentionLayer, ModelConfig } from '../types.js';
import {
  type ArchitectureAdapter,
  type HFConfig,
  type HFModelInfo,
  type AdapterParseResult,
  paramsFromMetadata,
} from './types.js';

const SUPPORTED = [
  'MixtralForCausalLM',
  'DeepseekV2ForCausalLM',
  'DeepseekV3ForCausalLM',
  'Qwen2MoeForCausalLM',
  'Qwen3MoeForCausalLM',
  'OlmoeForCausalLM',
];

function gatedMlpParams(hidden: number, inter: number): number {
  return 3 * hidden * inter;
}

export const mixtralAdapter: ArchitectureAdapter = {
  name: 'mixtral',
  matches(config) {
    const arches = config.architectures ?? [];
    return arches.some((a) => SUPPORTED.includes(a));
  },
  parse(modelId, config, metadata): AdapterParseResult {
    const warnings: string[] = [];
    const hidden = config.hidden_size ?? 0;
    const layers_n = config.num_hidden_layers ?? 0;
    const heads = config.num_attention_heads ?? Math.max(1, hidden / 128);
    const kvHeads = config.num_key_value_heads ?? heads;
    const headDim = config.head_dim ?? Math.floor(hidden / Math.max(1, heads));
    const vocab = config.vocab_size ?? 0;
    const numExperts = config.num_local_experts ?? 0;
    const topK = config.num_experts_per_tok ?? 0;
    const expertInter = config.moe_intermediate_size ?? config.intermediate_size ?? 0;
    const sharedInter = config.shared_expert_intermediate_size ?? 0;

    const layers: AttentionLayer[] = Array.from({ length: layers_n }, () => ({
      kind: 'full' as const,
      n_kv_heads: kvHeads,
      head_dim: headDim,
    }));

    let params = paramsFromMetadata(metadata);
    if (!params) {
      const attnPerLayer =
        hidden * heads * headDim + 2 * hidden * kvHeads * headDim + heads * headDim * hidden;
      const expertParamsPerLayer = numExperts * gatedMlpParams(hidden, expertInter);
      const sharedParamsPerLayer = sharedInter ? gatedMlpParams(hidden, sharedInter) : 0;
      const routerPerLayer = hidden * numExperts;
      const normPerLayer = 2 * hidden;
      const perLayer =
        attnPerLayer + expertParamsPerLayer + sharedParamsPerLayer + routerPerLayer + normPerLayer;
      const embed = hidden * vocab;
      params = embed * 2 + hidden + perLayer * layers_n;
      warnings.push(
        'safetensors.total missing — MoE param count estimated from architecture, may differ',
      );
    }

    let active_params: number | undefined;
    if (numExperts > 0 && topK > 0 && expertInter > 0) {
      const expertParam = gatedMlpParams(hidden, expertInter);
      const sharedParam = sharedInter ? gatedMlpParams(hidden, sharedInter) : 0;
      const attnPerLayer =
        hidden * heads * headDim + 2 * hidden * kvHeads * headDim + heads * headDim * hidden;
      const normPerLayer = 2 * hidden;
      const perLayerActive =
        attnPerLayer + sharedParam + topK * expertParam + hidden * numExperts + normPerLayer;
      const embed = hidden * vocab;
      active_params = embed * 2 + hidden + perLayerActive * layers_n;
    }

    const model: ModelConfig = {
      name: modelId,
      params,
      ...(active_params ? { active_params } : {}),
      hidden_dim: hidden,
      vocab_size: vocab,
      layers,
      architecture: 'mixtral',
    };
    return { model, warnings };
  },
};
