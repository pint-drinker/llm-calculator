import { z } from 'zod';

export const attentionLayerSchema = z.discriminatedUnion('kind', [
  z.object({
    kind: z.literal('full'),
    n_kv_heads: z.number().int().positive(),
    head_dim: z.number().int().positive(),
  }),
  z.object({
    kind: z.literal('linear'),
    state_size_bytes: z.number().positive(),
  }),
  z.object({ kind: z.literal('none') }),
]);

export const modelConfigSchema = z.object({
  name: z.string().min(1),
  params: z.number().positive(),
  active_params: z.number().positive().optional(),
  hidden_dim: z.number().int().positive(),
  layers: z.array(attentionLayerSchema).min(1),
  vocab_size: z.number().int().positive(),
  architecture: z.string().optional(),
});

export const weightQuantSchema = z.enum([
  'bf16',
  'fp16',
  'fp8',
  'int8',
  'awq_int4',
  'gptq_int4',
  'q8_0',
  'q5_k_m',
  'q4_k_m',
  'q3_k_m',
]);

export const kvQuantSchema = z.enum(['bf16', 'fp8', 'int4']);

export const tensorParallelSchema = z.union([
  z.literal(1),
  z.literal(2),
  z.literal(4),
  z.literal(8),
]);

export const inferenceConfigSchema = z.object({
  model: modelConfigSchema,
  weight_quant: weightQuantSchema,
  kv_quant: kvQuantSchema,
  context_length: z.number().int().positive(),
  batch_size: z.number().int().positive(),
  tensor_parallel: tensorParallelSchema,
});

export const gpuSchema = z.object({
  name: z.string().min(1),
  vram_gb: z.number().positive(),
  memory_bandwidth_gbs: z.number().positive(),
  fp16_tflops: z.number().nonnegative(),
  fp8_tflops: z.number().nonnegative().optional(),
  int4_tflops: z.number().nonnegative().optional(),
});

export const calculateRequestSchema = z.object({
  config: inferenceConfigSchema,
  gpu_name: z.string().optional(),
  gpu: gpuSchema.optional(),
});

export const calculateByNameRequestSchema = z.object({
  model_name: z.string().optional(),
  model: modelConfigSchema.optional(),
  weight_quant: weightQuantSchema,
  kv_quant: kvQuantSchema,
  context_length: z.number().int().positive(),
  batch_size: z.number().int().positive().default(1),
  tensor_parallel: tensorParallelSchema.default(1),
  gpu_name: z.string(),
});

export const maxContextRequestSchema = z.object({
  model_name: z.string().optional(),
  model: modelConfigSchema.optional(),
  weight_quant: weightQuantSchema,
  kv_quant: kvQuantSchema,
  gpu_name: z.string(),
  tensor_parallel: tensorParallelSchema.default(1),
  batch_size: z.number().int().positive().default(1),
  target_utilization: z.number().positive().max(1).default(0.9),
});

export const recommendRequestSchema = z.object({
  model_name: z.string().optional(),
  model: modelConfigSchema.optional(),
  weight_quant: weightQuantSchema,
  kv_quant: kvQuantSchema,
  context_length: z.number().int().positive(),
  batch_size: z.number().int().positive().default(1),
});

export const compareRequestSchema = z.object({
  configs: z.array(calculateByNameRequestSchema).min(1),
});

export const explainRequestSchema = calculateByNameRequestSchema;

export const listModelsFilterSchema = z
  .object({
    min_params: z.number().nonnegative().optional(),
    max_params: z.number().positive().optional(),
    architecture: z.string().optional(),
  })
  .partial()
  .optional();
