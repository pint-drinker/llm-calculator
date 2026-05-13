import { create } from 'zustand';
import type {
  GPU,
  InferenceConfig,
  KVQuant,
  ModelConfig,
  TensorParallel,
  WeightQuant,
} from '@llm-calc/core';
import { builtInGpus, builtInModels } from '@llm-calc/core';
import { decodeStateFromUrl } from './url.js';

export interface AppState {
  model: ModelConfig;
  gpu: GPU;
  weight_quant: WeightQuant;
  kv_quant: KVQuant;
  context_length: number;
  batch_size: number;
  tensor_parallel: TensorParallel;
  customModels: ModelConfig[];
  highlightedValue: string | null;
  setModel: (m: ModelConfig) => void;
  setGpu: (g: GPU) => void;
  setWeightQuant: (q: WeightQuant) => void;
  setKvQuant: (q: KVQuant) => void;
  setContext: (n: number) => void;
  setBatch: (n: number) => void;
  setTp: (n: TensorParallel) => void;
  addCustomModel: (m: ModelConfig) => void;
  setHighlighted: (k: string | null) => void;
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
  customModels: initial?.customModels ?? [],
  highlightedValue: null,
  setModel: (m) => set({ model: m }),
  setGpu: (g) => set({ gpu: g }),
  setWeightQuant: (q) => set({ weight_quant: q }),
  setKvQuant: (q) => set({ kv_quant: q }),
  setContext: (n) => set({ context_length: n }),
  setBatch: (n) => set({ batch_size: n }),
  setTp: (n) => set({ tensor_parallel: n }),
  addCustomModel: (m) =>
    set((s) => {
      const filtered = s.customModels.filter((c) => c.name !== m.name);
      return { customModels: [...filtered, m], model: m };
    }),
  setHighlighted: (k) => set({ highlightedValue: k }),
}));

export function selectConfig(s: AppState): InferenceConfig {
  return {
    model: s.model,
    weight_quant: s.weight_quant,
    kv_quant: s.kv_quant,
    context_length: s.context_length,
    batch_size: s.batch_size,
    tensor_parallel: s.tensor_parallel,
  };
}
