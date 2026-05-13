import {
  calculate,
  explain,
  findMaxContext,
  recommendHardware,
  builtInGpus,
  type GPU,
  type ModelConfig,
} from '@llm-calc/core';
import { importFromHF, GatedModelError } from '@llm-calc/core/hf';
import { findModel, registerModel, listModels, findGpu } from './registry.js';

export interface ResolveModelInput {
  model_name?: string;
  model?: ModelConfig;
}

export function resolveModel(input: ResolveModelInput): ModelConfig {
  if (input.model) return input.model;
  if (input.model_name) {
    const m = findModel(input.model_name);
    if (!m) throw new HttpError(404, 'model_not_found', { model_name: input.model_name });
    return m;
  }
  throw new HttpError(400, 'missing_model', { detail: 'Provide model_name or model object' });
}

export function resolveGpu(name: string): GPU {
  const g = findGpu(name);
  if (!g) throw new HttpError(404, 'gpu_not_found', { gpu_name: name });
  return g;
}

export class HttpError extends Error {
  constructor(public status: number, message: string, public details?: unknown) {
    super(message);
  }
}

export interface CalcArgs {
  model_name?: string;
  model?: ModelConfig;
  weight_quant: 'bf16' | 'fp16' | 'fp8' | 'int8' | 'awq_int4' | 'gptq_int4' | 'q8_0' | 'q5_k_m' | 'q4_k_m' | 'q3_k_m';
  kv_quant: 'bf16' | 'fp8' | 'int4';
  context_length: number;
  batch_size: number;
  tensor_parallel: 1 | 2 | 4 | 8;
  gpu_name: string;
}

export function doCalculate(args: CalcArgs) {
  const model = resolveModel(args);
  const gpu = resolveGpu(args.gpu_name);
  return calculate(
    {
      model,
      weight_quant: args.weight_quant,
      kv_quant: args.kv_quant,
      context_length: args.context_length,
      batch_size: args.batch_size,
      tensor_parallel: args.tensor_parallel,
    },
    gpu,
  );
}

export function doExplain(args: CalcArgs) {
  const model = resolveModel(args);
  const gpu = resolveGpu(args.gpu_name);
  return explain(
    {
      model,
      weight_quant: args.weight_quant,
      kv_quant: args.kv_quant,
      context_length: args.context_length,
      batch_size: args.batch_size,
      tensor_parallel: args.tensor_parallel,
    },
    gpu,
  );
}

export interface MaxContextArgs {
  model_name?: string;
  model?: ModelConfig;
  weight_quant: CalcArgs['weight_quant'];
  kv_quant: CalcArgs['kv_quant'];
  gpu_name: string;
  tensor_parallel: 1 | 2 | 4 | 8;
  batch_size: number;
  target_utilization: number;
}

export function doMaxContext(args: MaxContextArgs) {
  const model = resolveModel(args);
  const gpu = resolveGpu(args.gpu_name);
  return findMaxContext({
    model,
    weight_quant: args.weight_quant,
    kv_quant: args.kv_quant,
    gpu,
    tensor_parallel: args.tensor_parallel,
    batch_size: args.batch_size,
    target_utilization: args.target_utilization,
  });
}

export interface RecommendArgs {
  model_name?: string;
  model?: ModelConfig;
  weight_quant: CalcArgs['weight_quant'];
  kv_quant: CalcArgs['kv_quant'];
  context_length: number;
  batch_size: number;
}

export function doRecommend(args: RecommendArgs) {
  const model = resolveModel(args);
  return recommendHardware({
    model,
    weight_quant: args.weight_quant,
    kv_quant: args.kv_quant,
    context_length: args.context_length,
    batch_size: args.batch_size,
    gpus: builtInGpus,
  });
}

export function doCompare(configs: CalcArgs[]) {
  return configs.map((c) => ({ input: c, result: doCalculate(c) }));
}

export async function doImportHF(args: { model_id: string; token?: string; force_refresh?: boolean }) {
  try {
    const result = await importFromHF(args.model_id, {
      token: args.token,
      force_refresh: args.force_refresh,
    });
    registerModel(result.model);
    return result;
  } catch (err: unknown) {
    if (err instanceof GatedModelError) {
      throw new HttpError(401, 'gated_model', { model_id: err.modelId });
    }
    throw err;
  }
}

export function doListModels(filter?: {
  min_params?: number;
  max_params?: number;
  architecture?: string;
}): ModelConfig[] {
  let models = listModels();
  if (filter?.min_params) models = models.filter((m) => m.params >= filter.min_params!);
  if (filter?.max_params) models = models.filter((m) => m.params <= filter.max_params!);
  if (filter?.architecture) {
    const arch = filter.architecture.toLowerCase();
    models = models.filter((m) => (m.architecture ?? '').toLowerCase() === arch);
  }
  return models;
}

export function doListGpus(): GPU[] {
  return builtInGpus;
}
