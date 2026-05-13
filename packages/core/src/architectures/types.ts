import type { ModelConfig } from '../types.js';

export interface HFConfig {
  architectures?: string[];
  model_type?: string;
  hidden_size?: number;
  num_hidden_layers?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  head_dim?: number;
  vocab_size?: number;
  // MoE
  num_local_experts?: number;
  num_experts_per_tok?: number;
  intermediate_size?: number;
  moe_intermediate_size?: number;
  shared_expert_intermediate_size?: number;
  // Hybrid
  layer_types?: string[];
  layers_block_type?: string[];
  linear_attn_layer_idx?: number[];
  full_attn_layer_idx?: number[];
  // Mamba/DeltaNet
  mamba_d_state?: number;
  mamba_d_conv?: number;
  mamba_expand?: number;
  mamba_dt_rank?: number | string;
  state_size?: number;
  expand?: number;
  linear_num_value_heads?: number;
  linear_value_head_dim?: number;
  linear_num_key_heads?: number;
  linear_key_head_dim?: number;
  // generic catch-all
  [key: string]: unknown;
}

export interface HFModelInfo {
  modelId: string;
  safetensors?: {
    total?: number;
    parameters?: Record<string, number>;
  };
  pipeline_tag?: string;
  tags?: string[];
  [key: string]: unknown;
}

export interface AdapterParseResult {
  model: ModelConfig;
  warnings: string[];
}

export interface ArchitectureAdapter {
  name: string;
  matches(config: HFConfig): boolean;
  parse(modelId: string, config: HFConfig, metadata: HFModelInfo): AdapterParseResult;
}

export function paramsFromMetadata(metadata: HFModelInfo): number | undefined {
  const total = metadata.safetensors?.total;
  if (typeof total === 'number' && total > 0) return total;
  const parameters = metadata.safetensors?.parameters;
  if (parameters && typeof parameters === 'object') {
    const sum = Object.values(parameters).reduce(
      (acc, v) => acc + (typeof v === 'number' ? v : 0),
      0,
    );
    if (sum > 0) return sum;
  }
  return undefined;
}

export function estimateDenseGqaParams(config: HFConfig): number {
  // Best-effort param count for dense transformers when safetensors metadata is missing.
  const hidden = config.hidden_size ?? 0;
  const layers = config.num_hidden_layers ?? 0;
  const inter = config.intermediate_size ?? hidden * 4;
  const vocab = config.vocab_size ?? 0;
  const heads = config.num_attention_heads ?? Math.max(1, hidden / 128);
  const kvHeads = config.num_key_value_heads ?? heads;
  const headDim = config.head_dim ?? hidden / Math.max(1, heads);

  const qProj = hidden * heads * headDim;
  const kProj = hidden * kvHeads * headDim;
  const vProj = hidden * kvHeads * headDim;
  const oProj = heads * headDim * hidden;
  const attnPerLayer = qProj + kProj + vProj + oProj;
  const mlpPerLayer = 3 * hidden * inter; // gated MLP
  const normPerLayer = 2 * hidden;
  const perLayer = attnPerLayer + mlpPerLayer + normPerLayer;
  const embed = hidden * vocab;
  const finalNorm = hidden;
  return embed * 2 + finalNorm + perLayer * layers;
}
