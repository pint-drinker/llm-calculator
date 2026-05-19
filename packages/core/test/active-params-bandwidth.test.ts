import { describe, expect, it } from 'vitest';
import { calculate, findGpu, findModel, computeMemory, computeThroughput, computePrefillTime } from '../src/index.js';
import { bytesPerParam } from '../src/quantization.js';
import type { InferenceConfig } from '../src/index.js';

describe('active_params bandwidth calculation', () => {
  const mixtral = findModel('Mixtral-8x7B')!;
  const deepseek = findModel('DeepSeek-V3')!;
  const llama8b = findModel('Llama-3.1-8B')!;
  const dgxSpark = findGpu('DGX Spark')!;
  const h100 = findGpu('H100 80GB')!;
  const m4max = findGpu('Apple M4 Max')!;

  it('DGX Spark is present in catalog with unified_memory', () => {
    expect(dgxSpark).toBeDefined();
    expect(dgxSpark.unified_memory).toBe(true);
    expect(dgxSpark.vram_gb).toBe(128);
    expect(dgxSpark.memory_bandwidth_gbs).toBe(273);
  });

  it('Apple M4 Max has unified_memory flag', () => {
    expect(m4max.unified_memory).toBe(true);
  });

  it('H100 does not have unified_memory flag', () => {
    expect(h100.unified_memory).toBeUndefined();
  });

  it('memory_bound_tps uses active_params for MoE (Mixtral on DGX Spark)', () => {
    const config: InferenceConfig = {
      model: mixtral,
      weight_quant: 'q4_k_m',
      kv_quant: 'fp8',
      context_length: 8192,
      batch_size: 1,
      tensor_parallel: 1,
    };
    const memory = computeMemory(config);
    const throughput = computeThroughput(config, memory, dgxSpark);

    // active_params = 12.9B, q4_k_m = 0.5625 bytes/param
    // active_weights_bytes = 12.9e9 * 0.5625 ≈ 7.26 GB
    // memory_bound_tps ≈ 273e9 / 7.26e9 ≈ 37.6 tok/s
    const expectedActiveBytes = mixtral.active_params! * bytesPerParam('q4_k_m');
    const expectedTps = (273 * 1e9) / (expectedActiveBytes + memory.raw.kv_cache_bytes / 8192);

    expect(throughput.memory_bound_tps).toBeCloseTo(expectedTps, 0);
    expect(throughput.memory_bound_tps).toBeGreaterThan(30);
    expect(throughput.memory_bound_tps).toBeLessThan(45);
  });

  it('memory_bound_tps would be much lower if using total params (verifies fix)', () => {
    const config: InferenceConfig = {
      model: mixtral,
      weight_quant: 'q4_k_m',
      kv_quant: 'fp8',
      context_length: 8192,
      batch_size: 1,
      tensor_parallel: 1,
    };
    const memory = computeMemory(config);
    const throughput = computeThroughput(config, memory, dgxSpark);

    // If we used total params (46.7B) instead of active (12.9B):
    // total_weights_bytes = 46.7e9 * 0.5625 ≈ 26.3 GB
    // wrong_tps ≈ 273e9 / 26.3e9 ≈ 10.4 tok/s
    const wrongBytesPerToken = mixtral.params * bytesPerParam('q4_k_m');
    const wrongTps = (273 * 1e9) / wrongBytesPerToken;

    // Actual TPS should be ~3.6x higher than the wrong calculation
    expect(throughput.memory_bound_tps).toBeGreaterThan(wrongTps * 3);
  });

  it('dense model memory_bound_tps unchanged (active_params == params)', () => {
    const config: InferenceConfig = {
      model: llama8b,
      weight_quant: 'q4_k_m',
      kv_quant: 'fp8',
      context_length: 8192,
      batch_size: 1,
      tensor_parallel: 1,
    };
    const memory = computeMemory(config);
    const throughput = computeThroughput(config, memory, dgxSpark);

    // For dense model: active_params == params, so calculation should match
    const expectedBytes = llama8b.params * bytesPerParam('q4_k_m');
    const kvPerToken = memory.raw.kv_cache_bytes / 8192;
    const expectedTps = (273 * 1e9) / (expectedBytes + kvPerToken);

    expect(throughput.memory_bound_tps).toBeCloseTo(expectedTps, 0);
  });

  it('prefill memory floor uses active_params for MoE', () => {
    const config: InferenceConfig = {
      model: mixtral,
      weight_quant: 'q4_k_m',
      kv_quant: 'fp8',
      context_length: 8192,
      batch_size: 1,
      tensor_parallel: 1,
    };
    const memory = computeMemory(config);
    const ttft = computePrefillTime(config, memory, dgxSpark, 2000);

    // Memory floor = active_weights_bytes / bandwidth
    // = (12.9e9 * 0.5625) / (273e9) ≈ 0.027s
    const activeWeightsBytes = mixtral.active_params! * bytesPerParam('q4_k_m');
    const expectedMemFloor = activeWeightsBytes / (273 * 1e9);

    // TTFT should be >= memory floor (could be higher if compute-bound)
    expect(ttft).toBeGreaterThanOrEqual(expectedMemFloor * 0.99);
  });

  it('DeepSeek-V3 on DGX Spark: 37B active gives reasonable TPS', () => {
    const config: InferenceConfig = {
      model: deepseek,
      weight_quant: 'q4_k_m',
      kv_quant: 'fp8',
      context_length: 8192,
      batch_size: 1,
      tensor_parallel: 1,
    };
    const memory = computeMemory(config);
    const throughput = computeThroughput(config, memory, deepseek.active_params ? dgxSpark : dgxSpark);

    // active_params = 37B, q4_k_m = 0.5625 bytes/param
    // active_weights_bytes = 37e9 * 0.5625 ≈ 20.8 GB
    // memory_bound_tps ≈ 273e9 / 20.8e9 ≈ 13.1 tok/s
    const activeBytes = deepseek.active_params! * bytesPerParam('q4_k_m');
    const expectedTps = (273 * 1e9) / (activeBytes + memory.raw.kv_cache_bytes / 8192);

    expect(throughput.memory_bound_tps).toBeCloseTo(expectedTps, 0);
    expect(throughput.memory_bound_tps).toBeGreaterThan(10);
    expect(throughput.memory_bound_tps).toBeLessThan(20);
  });

  it('unified memory devices are always memory-bound at batch=1', () => {
    const config: InferenceConfig = {
      model: llama8b,
      weight_quant: 'q4_k_m',
      kv_quant: 'fp8',
      context_length: 8192,
      batch_size: 1,
      tensor_parallel: 1,
    };
    const memory = computeMemory(config);
    const throughput = computeThroughput(config, memory, dgxSpark);

    expect(throughput.bottleneck).toBe('memory');
  });
});
