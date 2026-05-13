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
    label: 'Llama 70B on 2×H100',
    modelName: 'Llama-3.1-70B',
    gpuName: 'H100 80GB',
    weight_quant: 'bf16',
    kv_quant: 'bf16',
    context_length: 32768,
    tensor_parallel: 2,
  },
  {
    label: 'Mixtral 8x7B on A100-80',
    modelName: 'Mixtral-8x7B',
    gpuName: 'A100 80GB',
    weight_quant: 'bf16',
    kv_quant: 'bf16',
    context_length: 8192,
    tensor_parallel: 1,
  },
  {
    label: 'Llama 8B on RTX 3090',
    modelName: 'Llama-3.1-8B',
    gpuName: 'RTX 3090',
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
