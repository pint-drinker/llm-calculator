import { builtInGpus, builtInModels } from '@llm-calc/core';
import type { KVQuant, TensorParallel, WeightQuant } from '@llm-calc/core';
import { useStore } from '../store.js';

interface Preset {
  label: string;
  modelName: string;
  gpuName: string;
  weight_quant: WeightQuant;
  kv_quant: KVQuant;
  context_length: number;
  tensor_parallel: TensorParallel;
}

const PRESETS: Preset[] = [
  {
    label: 'Qwen3.6-27B on RTX 4090',
    modelName: 'Qwen3.6-27B-Hybrid',
    gpuName: 'RTX 4090',
    weight_quant: 'awq_int4',
    kv_quant: 'fp8',
    context_length: 262144,
    tensor_parallel: 1,
  },
  {
    label: 'Qwen3.6-27B on Apple M4 Pro',
    modelName: 'Qwen3.6-27B-Hybrid',
    gpuName: 'Apple M4 Pro',
    weight_quant: 'awq_int4',
    kv_quant: 'fp8',
    context_length: 262144,
    tensor_parallel: 1,
  },
  {
    label: 'Qwen3.6-35B on DGX Spark',
    modelName: 'Qwen3.6-35B-A3B',
    gpuName: 'DGX Spark',
    weight_quant: 'awq_int4',
    kv_quant: 'fp8',
    context_length: 262144,
    tensor_parallel: 1,
  },
  {
    label: 'Qwen3.5-2B on iPhone A17 Pro',
    modelName: 'Qwen3.5-2B',
    gpuName: 'Apple A17 Pro (iPhone 15 Pro)',
    weight_quant: 'awq_int4',
    kv_quant: 'fp8',
    context_length: 32768,
    tensor_parallel: 1,
  },
  {
    label: 'Qwen3.5-4B on iPhone A17 Pro',
    modelName: 'Qwen3.5-4B',
    gpuName: 'Apple A17 Pro (iPhone 15 Pro)',
    weight_quant: 'awq_int4',
    kv_quant: 'fp8',
    context_length: 32768,
    tensor_parallel: 1,
  },
];

export function Presets() {
  const { setModel, setGpu, setWeightQuant, setKvQuant, setContext, setTp } = useStore();
  return (
    <div className="flex flex-col gap-1">
      {PRESETS.map((p) => (
        <button
          key={p.label}
          className="btn text-left"
          onClick={() => {
            const m = builtInModels.find((x) => x.name === p.modelName);
            const g = builtInGpus.find((x) => x.name === p.gpuName);
            if (m) setModel(m);
            if (g) setGpu(g);
            setWeightQuant(p.weight_quant);
            setKvQuant(p.kv_quant);
            setContext(p.context_length);
            setTp(p.tensor_parallel);
          }}
        >
          {p.label}
        </button>
      ))}
    </div>
  );
}
