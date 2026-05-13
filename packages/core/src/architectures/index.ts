import { llamaAdapter } from './llama.js';
import { mixtralAdapter } from './mixtral.js';
import { qwen3NextAdapter } from './qwen3-next.js';
import { jambaAdapter } from './jamba.js';
import { fallbackAdapter } from './fallback.js';
import type {
  AdapterParseResult,
  ArchitectureAdapter,
  HFConfig,
  HFModelInfo,
} from './types.js';

export const adapters: ArchitectureAdapter[] = [
  llamaAdapter,
  mixtralAdapter,
  qwen3NextAdapter,
  jambaAdapter,
  fallbackAdapter,
];

export function dispatch(
  modelId: string,
  config: HFConfig,
  metadata: HFModelInfo,
): AdapterParseResult & { adapter: string } {
  for (const adapter of adapters) {
    if (adapter.matches(config)) {
      const out = adapter.parse(modelId, config, metadata);
      return { ...out, adapter: adapter.name };
    }
  }
  const out = fallbackAdapter.parse(modelId, config, metadata);
  return { ...out, adapter: fallbackAdapter.name };
}

export { llamaAdapter, mixtralAdapter, qwen3NextAdapter, jambaAdapter, fallbackAdapter };
export type { ArchitectureAdapter, AdapterParseResult, HFConfig, HFModelInfo } from './types.js';
