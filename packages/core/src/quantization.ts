import type { WeightQuant, KVQuant, InferenceEngine } from './types.js';

const WEIGHT_BYTES_PER_PARAM: Record<WeightQuant, number> = {
  bf16: 2.0,
  fp16: 2.0,
  fp8: 1.0,
  int8: 1.0,
  awq_int4: 0.5625,
  gptq_int4: 0.5625,
  q8_0: 1.0625,
  q5_k_m: 0.6875,
  q4_k_m: 0.5625,
  q3_k_m: 0.4375,
};

const KV_BYTES_PER_ELEMENT: Record<KVQuant, number> = {
  bf16: 2,
  fp8: 1,
  int4: 0.5,
};

export function bytesPerParam(q: WeightQuant): number {
  return WEIGHT_BYTES_PER_PARAM[q];
}

export function kvBytesPerElement(q: KVQuant): number {
  return KV_BYTES_PER_ELEMENT[q];
}

export const WEIGHT_QUANTS: WeightQuant[] = Object.keys(
  WEIGHT_BYTES_PER_PARAM,
) as WeightQuant[];

export const KV_QUANTS: KVQuant[] = Object.keys(
  KV_BYTES_PER_ELEMENT,
) as KVQuant[];

// Weight formats each engine realistically serves: sglang loads HF-style
// checkpoints (bf16/fp8/AWQ/GPTQ...), llama.cpp loads GGUF quants.
export const ENGINE_WEIGHT_QUANTS: Record<InferenceEngine, WeightQuant[]> = {
  sglang: ['bf16', 'fp16', 'fp8', 'int8', 'awq_int4', 'gptq_int4'],
  llama_cpp: ['fp16', 'q8_0', 'q5_k_m', 'q4_k_m', 'q3_k_m'],
};

// Nearest-equivalent quant when switching engines.
const TO_LLAMACPP_QUANT: Partial<Record<WeightQuant, WeightQuant>> = {
  bf16: 'fp16',
  fp8: 'q8_0',
  int8: 'q8_0',
  awq_int4: 'q4_k_m',
  gptq_int4: 'q4_k_m',
};
const TO_SGLANG_QUANT: Partial<Record<WeightQuant, WeightQuant>> = {
  q8_0: 'int8',
  q5_k_m: 'awq_int4',
  q4_k_m: 'awq_int4',
  q3_k_m: 'awq_int4',
};

/** Map a weight quant to the closest equivalent supported by `engine`. */
export function mapQuantToEngine(q: WeightQuant, engine: InferenceEngine): WeightQuant {
  if (ENGINE_WEIGHT_QUANTS[engine].includes(q)) return q;
  const mapped = engine === 'llama_cpp' ? TO_LLAMACPP_QUANT[q] : TO_SGLANG_QUANT[q];
  return mapped ?? ENGINE_WEIGHT_QUANTS[engine][0];
}
