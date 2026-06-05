export type AttentionLayer =
  | { kind: 'full'; n_kv_heads: number; head_dim: number }
  | { kind: 'linear'; state_size_bytes: number }
  | { kind: 'none' };

export interface ModelConfig {
  name: string;
  params: number;
  active_params?: number;
  hidden_dim: number;
  layers: AttentionLayer[];
  vocab_size: number;
  architecture?: string;
}

export type WeightQuant =
  | 'bf16'
  | 'fp16'
  | 'fp8'
  | 'int8'
  | 'awq_int4'
  | 'gptq_int4'
  | 'q8_0'
  | 'q5_k_m'
  | 'q4_k_m'
  | 'q3_k_m';

export type KVQuant = 'bf16' | 'fp8' | 'int4';

export type TensorParallel = 1 | 2 | 4 | 8;

export type InferenceEngine = 'sglang' | 'llama_cpp';

export interface InferenceConfig {
  model: ModelConfig;
  weight_quant: WeightQuant;
  kv_quant: KVQuant;
  context_length: number;
  batch_size: number;
  tensor_parallel: TensorParallel;
  /** Inference engine the estimate is calibrated for. Defaults to 'sglang'. */
  engine?: InferenceEngine;
}

export interface GPU {
  name: string;
  vram_gb: number;
  memory_bandwidth_gbs: number;
  fp16_tflops: number;
  fp8_tflops?: number;
  int4_tflops?: number;
  usable_memory_fraction?: number;
  unified_memory?: boolean;
}

export interface MemoryBreakdown {
  weights_gb: number;
  kv_cache_gb: number;
  linear_state_gb: number;
  activations_gb: number;
  framework_overhead_gb: number;
  total_gb: number;
  per_gpu_gb: number;
}

export interface ThroughputEstimate {
  memory_bound_tps: number;
  compute_bound_tps: number;
  estimated_tps: number;
  bottleneck: 'memory' | 'compute';
  ttft_seconds: number;
}

export interface CalculationResult {
  memory: MemoryBreakdown;
  throughput: ThroughputEstimate;
  fits: boolean;
  fits_usable: boolean;
  usable_vram_gb: number;
  utilization_pct: number;
  warnings: string[];
}

export interface ExplainStep {
  name: string;
  formula: string;
  inputs: Record<string, number | string>;
  substituted: string;
  result: number;
  units: string;
}

export interface ExplainTrace {
  steps: ExplainStep[];
}
