import LZString from 'lz-string';
import type {
  GPU,
  KVQuant,
  ModelConfig,
  TensorParallel,
  WeightQuant,
} from '@llm-calc/core';
import { builtInGpus, builtInModels } from '@llm-calc/core';

export interface UrlState {
  model: ModelConfig;
  gpu: GPU;
  weight_quant: WeightQuant;
  kv_quant: KVQuant;
  context_length: number;
  batch_size: number;
  tensor_parallel: TensorParallel;
}

interface Encoded {
  m?: string;
  g?: string;
  wq?: WeightQuant;
  kq?: KVQuant;
  c?: number;
  b?: number;
  tp?: TensorParallel;
}

export function encodeState(state: UrlState): string {
  const e: Encoded = {
    m: state.model.name,
    g: state.gpu.name,
    wq: state.weight_quant,
    kq: state.kv_quant,
    c: state.context_length,
    b: state.batch_size,
    tp: state.tensor_parallel,
  };
  return LZString.compressToEncodedURIComponent(JSON.stringify(e));
}

export function decodeStateFromUrl(): UrlState | null {
  if (typeof window === 'undefined') return null;
  const qs = new URLSearchParams(window.location.search).get('s');
  if (!qs) return null;
  try {
    const raw = LZString.decompressFromEncodedURIComponent(qs);
    if (!raw) return null;
    const e = JSON.parse(raw) as Encoded;
    const model = builtInModels.find((m) => m.name === e.m) ?? builtInModels[0];
    const gpu = builtInGpus.find((g) => g.name === e.g) ?? builtInGpus[0];
    return {
      model,
      gpu,
      weight_quant: e.wq ?? 'bf16',
      kv_quant: e.kq ?? 'bf16',
      context_length: e.c ?? 8192,
      batch_size: e.b ?? 1,
      tensor_parallel: e.tp ?? 1,
    };
  } catch {
    return null;
  }
}

export function writeUrl(state: UrlState): void {
  if (typeof window === 'undefined') return;
  const encoded = encodeState(state);
  const url = new URL(window.location.href);
  url.searchParams.set('s', encoded);
  window.history.replaceState({}, '', url.toString());
}
