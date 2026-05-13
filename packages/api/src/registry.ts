import type { ModelConfig } from '@llm-calc/core';
import { builtInModels, findGpu as coreFindGpu } from '@llm-calc/core';

const sessionModels = new Map<string, ModelConfig>();

export function registerModel(model: ModelConfig): void {
  sessionModels.set(model.name.toLowerCase(), model);
}

export function listModels(): ModelConfig[] {
  const merged = new Map<string, ModelConfig>();
  for (const m of builtInModels) merged.set(m.name.toLowerCase(), m);
  for (const [k, v] of sessionModels) merged.set(k, v);
  return Array.from(merged.values());
}

export function findModel(name: string): ModelConfig | undefined {
  const lc = name.toLowerCase();
  return sessionModels.get(lc) ?? builtInModels.find((m) => m.name.toLowerCase() === lc);
}

export { coreFindGpu as findGpu };
