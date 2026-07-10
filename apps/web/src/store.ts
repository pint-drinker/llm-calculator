import { create } from 'zustand';
import type {
  GPU,
  InferenceConfig,
  InferenceEngine,
  KVQuant,
  ModelConfig,
  TensorParallel,
  WeightQuant,
} from '@llm-calc/core';
import { builtInGpus, builtInModels, mapQuantToEngine } from '@llm-calc/core';
import { decodeStateFromUrl } from './url.js';

export interface AppState {
  model: ModelConfig;
  gpu: GPU;
  weight_quant: WeightQuant;
  kv_quant: KVQuant;
  context_length: number;
  batch_size: number;
  tensor_parallel: TensorParallel;
  include_mmproj: boolean;
  inference_engine: InferenceEngine;
  initial_prompt_tokens: number;
  sci_enabled: boolean;
  sci_context_limit: number;
  ttft_threshold_s: number;
  throughput_threshold_tps: number;
  usable_memory_fraction: number;
  highlightedValue: string | null;
  setModel: (m: ModelConfig) => void;
  setGpu: (g: GPU) => void;
  setWeightQuant: (q: WeightQuant) => void;
  setKvQuant: (q: KVQuant) => void;
  setContext: (n: number) => void;
  setBatch: (n: number) => void;
  setTp: (n: TensorParallel) => void;
  setIncludeMmproj: (b: boolean) => void;
  setInferenceEngine: (e: InferenceEngine) => void;
  setInitialPrompt: (n: number) => void;
  setSciEnabled: (b: boolean) => void;
  setSciContextLimit: (n: number) => void;
  setTtftThreshold: (n: number) => void;
  setThroughputThreshold: (n: number) => void;
  setUsableMemoryFraction: (n: number) => void;
  setHighlighted: (k: string | null) => void;
}

function clampUsableMemoryFraction(n: number): number {
  return Math.min(0.95, Math.max(0.1, n));
}

const defaultModel =
  builtInModels.find((m) => m.name === 'Qwen3.6-27B-Hybrid') ?? builtInModels[0];
const defaultGpu = builtInGpus.find((g) => g.name === 'RTX 4090') ?? builtInGpus[0];

const initial = decodeStateFromUrl();

export const useStore = create<AppState>((set) => ({
  model: initial?.model ?? defaultModel,
  gpu: initial?.gpu ?? defaultGpu,
  weight_quant: initial?.weight_quant ?? 'awq_int4',
  kv_quant: initial?.kv_quant ?? 'fp8',
  context_length: initial?.context_length ?? 262144,
  batch_size: initial?.batch_size ?? 1,
  tensor_parallel: initial?.tensor_parallel ?? 1,
  include_mmproj: initial?.include_mmproj ?? false,
  inference_engine: initial?.inference_engine ?? 'sglang',
  initial_prompt_tokens: 2000,
  sci_enabled: false,
  sci_context_limit: 40960,
  ttft_threshold_s: 5,
  throughput_threshold_tps: 15,
  usable_memory_fraction: 0.5,
  highlightedValue: null,
  setModel: (m) => set({ model: m }),
  setGpu: (g) => set({ gpu: g }),
  setWeightQuant: (q) => set({ weight_quant: q }),
  setKvQuant: (q) => set({ kv_quant: q }),
  setContext: (n) => set({ context_length: n }),
  setBatch: (n) => set({ batch_size: n }),
  setTp: (n) => set({ tensor_parallel: n }),
  setIncludeMmproj: (b) => set({ include_mmproj: b }),
  setInferenceEngine: (e) =>
    set((s) => ({
      inference_engine: e,
      weight_quant: mapQuantToEngine(s.weight_quant, e),
    })),
  setInitialPrompt: (n) => set({ initial_prompt_tokens: n }),
  setSciEnabled: (b) => set({ sci_enabled: b }),
  setSciContextLimit: (n) => set({ sci_context_limit: n }),
  setTtftThreshold: (n) => set({ ttft_threshold_s: n }),
  setThroughputThreshold: (n) => set({ throughput_threshold_tps: n }),
  setUsableMemoryFraction: (n) =>
    set({ usable_memory_fraction: clampUsableMemoryFraction(n) }),
  setHighlighted: (k) => set({ highlightedValue: k }),
}));

export function selectEffectiveGpu(s: AppState): GPU {
  return { ...s.gpu, usable_memory_fraction: s.usable_memory_fraction };
}

export function selectConfig(s: AppState): InferenceConfig {
  return {
    model: s.model,
    weight_quant: s.weight_quant,
    kv_quant: s.kv_quant,
    context_length: s.sci_enabled
      ? Math.min(s.context_length, s.sci_context_limit)
      : s.context_length,
    batch_size: s.batch_size,
    tensor_parallel: s.tensor_parallel,
    include_mmproj: s.include_mmproj,
    engine: s.inference_engine,
  };
}
