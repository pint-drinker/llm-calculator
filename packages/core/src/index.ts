export * from './types.js';
export * from './quantization.js';
export * as schemas from './schemas.js';
export { computeMemory, kvBytesPerToken, MEMORY_CONSTANTS } from './memory.js';
export { computeThroughput, computePrefillTime, THROUGHPUT_CONSTANTS } from './throughput.js';
export { calculate } from './calculate.js';
export { findMaxContext, recommendHardware } from './recommend.js';
export type { MaxContextResult, HardwareCandidate } from './recommend.js';
export { explain } from './explain.js';
export { builtInModels, builtInGpus, findModel, findGpu } from './catalog.js';
// HF import lives in @llm-calc/core/hf (Node-only) so the browser bundle
// stays free of node:fs / node:os imports.
export type { ImportResult, ImportOptions } from './hf/import.js';
export {
  adapters,
  dispatch as dispatchArchitecture,
  llamaAdapter,
  mixtralAdapter,
  qwen3NextAdapter,
  jambaAdapter,
  fallbackAdapter,
} from './architectures/index.js';
export type {
  ArchitectureAdapter,
  AdapterParseResult,
  HFConfig,
  HFModelInfo,
} from './architectures/index.js';
