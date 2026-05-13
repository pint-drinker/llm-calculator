import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { schemas } from '@llm-calc/core';
import {
  doCalculate,
  doExplain,
  doMaxContext,
  doRecommend,
  doCompare,
  doImportHF,
  doListModels,
  doListGpus,
} from '../handlers.js';

function jsonResult(value: unknown) {
  return {
    content: [{ type: 'text' as const, text: JSON.stringify(value, null, 2) }],
  };
}

const calcInput = {
  model_name: z.string().optional(),
  model: schemas.modelConfigSchema.optional(),
  weight_quant: schemas.weightQuantSchema,
  kv_quant: schemas.kvQuantSchema,
  context_length: z.number().int().positive(),
  batch_size: z.number().int().positive().default(1),
  tensor_parallel: schemas.tensorParallelSchema.default(1),
  gpu_name: z.string(),
};

export function createMcpServer(): McpServer {
  const server = new McpServer({ name: 'llm-calc', version: '0.1.0' });

  server.registerTool(
    'calculate_inference_memory',
    {
      title: 'Calculate inference VRAM & throughput',
      description:
        "Estimate VRAM and throughput for a model on specific hardware at a given context length and quantization. Use this for the direct question: 'will this fit?'",
      inputSchema: calcInput,
    },
    async (args) => jsonResult(doCalculate(args as Parameters<typeof doCalculate>[0])),
  );

  server.registerTool(
    'find_max_context',
    {
      title: 'Find max context length that fits',
      description:
        "Find the largest context length that fits in VRAM at target utilization. Use when asked 'how long a context can I run?'",
      inputSchema: {
        model_name: z.string().optional(),
        model: schemas.modelConfigSchema.optional(),
        weight_quant: schemas.weightQuantSchema,
        kv_quant: schemas.kvQuantSchema,
        gpu_name: z.string(),
        tensor_parallel: schemas.tensorParallelSchema.default(1),
        batch_size: z.number().int().positive().default(1),
        target_utilization: z.number().positive().max(1).default(0.9),
      },
    },
    async (args) => jsonResult(doMaxContext(args as Parameters<typeof doMaxContext>[0])),
  );

  server.registerTool(
    'recommend_hardware',
    {
      title: 'Recommend GPUs',
      description:
        "Rank GPUs by smallest-that-fits, including multi-GPU TP configurations. Use when asked 'what hardware do I need?'",
      inputSchema: {
        model_name: z.string().optional(),
        model: schemas.modelConfigSchema.optional(),
        weight_quant: schemas.weightQuantSchema,
        kv_quant: schemas.kvQuantSchema,
        context_length: z.number().int().positive(),
        batch_size: z.number().int().positive().default(1),
      },
    },
    async (args) =>
      jsonResult({ candidates: doRecommend(args as Parameters<typeof doRecommend>[0]) }),
  );

  server.registerTool(
    'compare_configurations',
    {
      title: 'Compare multiple inference configurations',
      description:
        "Calculate results for multiple configurations side-by-side for comparison. Use to answer 'bf16 vs int4', 'A100 vs H100', or 'how does this scale with context.'",
      inputSchema: {
        configs: z.array(z.object(calcInput)).min(1),
      },
    },
    async (args) => jsonResult(doCompare(args.configs as Parameters<typeof doCompare>[0])),
  );

  server.registerTool(
    'list_models',
    {
      title: 'List models in catalog',
      description:
        'List models in the catalog. Call before other tools to discover valid model names.',
      inputSchema: {
        filter: z
          .object({
            min_params: z.number().nonnegative().optional(),
            max_params: z.number().positive().optional(),
            architecture: z.string().optional(),
          })
          .partial()
          .optional(),
      },
    },
    async (args) => jsonResult(doListModels(args.filter)),
  );

  server.registerTool(
    'list_gpus',
    {
      title: 'List GPUs in catalog',
      description: 'List GPUs in the catalog.',
      inputSchema: {},
    },
    async () => jsonResult(doListGpus()),
  );

  server.registerTool(
    'import_model_from_hf',
    {
      title: 'Import a model from HuggingFace Hub',
      description:
        "Fetch a model's architecture from HuggingFace Hub by ID (e.g. 'meta-llama/Llama-3.1-70B') and return a ModelConfig ready for use in other tools. Use when the user names a model not in the catalog, or wants to check a brand-new release. Warnings include cases where the architecture wasn't recognized and a fallback was used.",
      inputSchema: {
        model_id: z.string().min(1),
        token: z.string().optional(),
        force_refresh: z.boolean().optional(),
      },
    },
    async (args) => jsonResult(await doImportHF(args)),
  );

  server.registerTool(
    'explain_calculation',
    {
      title: 'Trace a calculation step-by-step',
      description:
        "Return a step-by-step trace of the calculation showing each formula with substituted values. Use when the user asks 'why' or 'how was that computed' or wants to verify the math.",
      inputSchema: calcInput,
    },
    async (args) => jsonResult(doExplain(args as Parameters<typeof doExplain>[0])),
  );

  return server;
}

export const MCP_TOOL_NAMES = [
  'calculate_inference_memory',
  'find_max_context',
  'recommend_hardware',
  'compare_configurations',
  'list_models',
  'list_gpus',
  'import_model_from_hf',
  'explain_calculation',
] as const;
