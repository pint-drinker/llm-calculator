import type { WeightQuant, KVQuant } from './types.js';

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
