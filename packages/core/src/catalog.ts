import type { GPU, ModelConfig } from './types.js';
import modelsJson from '../data/models.json' with { type: 'json' };
import gpusJson from '../data/gpus.json' with { type: 'json' };

export const builtInModels: ModelConfig[] = modelsJson as ModelConfig[];
export const builtInGpus: GPU[] = gpusJson as GPU[];

export function findModel(name: string, extra: ModelConfig[] = []): ModelConfig | undefined {
  const lc = name.toLowerCase();
  return (
    extra.find((m) => m.name.toLowerCase() === lc) ??
    builtInModels.find((m) => m.name.toLowerCase() === lc)
  );
}

export function findGpu(name: string): GPU | undefined {
  const lc = name.toLowerCase();
  return builtInGpus.find((g) => g.name.toLowerCase() === lc);
}
