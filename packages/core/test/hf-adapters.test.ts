import { describe, expect, it } from 'vitest';
import { dispatchArchitecture } from '../src/index.js';

describe('llama adapter', () => {
  it('parses Llama 3.1 8B config shape', () => {
    const { model, adapter } = dispatchArchitecture(
      'meta-llama/Llama-3.1-8B',
      {
        architectures: ['LlamaForCausalLM'],
        hidden_size: 4096,
        num_hidden_layers: 32,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        head_dim: 128,
        vocab_size: 128256,
      },
      { modelId: 'meta-llama/Llama-3.1-8B', safetensors: { total: 8_030_000_000 } },
    );
    expect(adapter).toBe('llama');
    expect(model.params).toBe(8_030_000_000);
    expect(model.layers).toHaveLength(32);
    expect(model.layers[0]).toEqual({ kind: 'full', n_kv_heads: 8, head_dim: 128 });
  });
});

describe('mixtral adapter', () => {
  it('computes active_params for MoE', () => {
    const { model, adapter } = dispatchArchitecture(
      'mistralai/Mixtral-8x7B-v0.1',
      {
        architectures: ['MixtralForCausalLM'],
        hidden_size: 4096,
        num_hidden_layers: 32,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        head_dim: 128,
        vocab_size: 32000,
        num_local_experts: 8,
        num_experts_per_tok: 2,
        intermediate_size: 14336,
      },
      { modelId: 'mistralai/Mixtral-8x7B-v0.1', safetensors: { total: 46_700_000_000 } },
    );
    expect(adapter).toBe('mixtral');
    expect(model.params).toBe(46_700_000_000);
    expect(model.active_params).toBeGreaterThan(8_000_000_000);
    expect(model.active_params).toBeLessThan(20_000_000_000);
  });
});

describe('qwen3-next adapter', () => {
  it('emits hybrid layers from layer_types', () => {
    const layer_types = Array.from({ length: 8 }, (_, i) =>
      i % 4 === 3 ? 'full_attention' : 'linear_attention',
    );
    const { model, adapter } = dispatchArchitecture(
      'Qwen/Qwen3.6-27B',
      {
        architectures: ['Qwen3NextForCausalLM'],
        hidden_size: 2048,
        num_hidden_layers: 8,
        num_attention_heads: 16,
        num_key_value_heads: 4,
        head_dim: 256,
        vocab_size: 152064,
        layer_types,
        linear_num_value_heads: 16,
        linear_value_head_dim: 128,
        linear_num_key_heads: 16,
        linear_key_head_dim: 128,
      },
      { modelId: 'Qwen/Qwen3.6-27B', safetensors: { total: 27_000_000_000 } },
    );
    expect(adapter).toBe('qwen3-next');
    expect(model.layers.filter((l) => l.kind === 'full')).toHaveLength(2);
    expect(model.layers.filter((l) => l.kind === 'linear')).toHaveLength(6);
  });

  it('parses multimodal Qwen3.5 with nested text_config', () => {
    const layer_types = Array.from({ length: 24 }, (_, i) =>
      i % 4 === 3 ? 'full_attention' : 'linear_attention',
    );
    const { model, adapter, warnings } = dispatchArchitecture(
      'Qwen/Qwen3.5-2B',
      {
        architectures: ['Qwen3_5ForConditionalGeneration'],
        model_type: 'qwen3_5',
        text_config: {
          model_type: 'qwen3_5_text',
          hidden_size: 2048,
          num_hidden_layers: 24,
          num_attention_heads: 8,
          num_key_value_heads: 2,
          head_dim: 256,
          vocab_size: 248320,
          layer_types,
          linear_num_value_heads: 16,
          linear_value_head_dim: 128,
          linear_num_key_heads: 16,
          linear_key_head_dim: 128,
        },
      } as never,
      { modelId: 'Qwen/Qwen3.5-2B', safetensors: { total: 2_274_069_824 } },
    );
    expect(adapter).toBe('qwen3-next');
    expect(model.architecture).toBe('qwen3.5');
    expect(model.params).toBe(2_274_069_824);
    expect(model.hidden_dim).toBe(2048);
    expect(model.vocab_size).toBe(248320);
    expect(model.layers).toHaveLength(24);
    expect(model.layers.filter((l) => l.kind === 'full')).toHaveLength(6);
    expect(model.layers.filter((l) => l.kind === 'linear')).toHaveLength(18);
    expect(warnings).toHaveLength(0);
  });
});

describe('jamba adapter', () => {
  it('reads layers_block_type', () => {
    const layers_block_type = ['attention', 'mamba', 'mamba', 'mamba'];
    const { model, adapter } = dispatchArchitecture(
      'ai21labs/Jamba-toy',
      {
        architectures: ['JambaForCausalLM'],
        hidden_size: 4096,
        num_hidden_layers: 4,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        head_dim: 128,
        vocab_size: 65536,
        mamba_d_state: 16,
        mamba_expand: 2,
        layers_block_type,
      },
      { modelId: 'ai21labs/Jamba-toy', safetensors: { total: 1_000_000_000 } },
    );
    expect(adapter).toBe('jamba');
    expect(model.layers[0].kind).toBe('full');
    expect(model.layers[1].kind).toBe('linear');
  });
});

describe('fallback adapter', () => {
  it('emits warning for unknown architecture', () => {
    const { warnings, adapter } = dispatchArchitecture(
      'foo/Bar',
      {
        architectures: ['FooBarForCausalLM'],
        hidden_size: 4096,
        num_hidden_layers: 32,
        num_attention_heads: 32,
        vocab_size: 50000,
      },
      { modelId: 'foo/Bar', safetensors: { total: 5_000_000_000 } },
    );
    expect(adapter).toBe('fallback');
    expect(warnings.some((w) => /Unknown architecture/.test(w))).toBe(true);
  });
});
