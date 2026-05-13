import type { HFConfig, HFModelInfo } from '../architectures/types.js';

export class GatedModelError extends Error {
  override name = 'GatedModelError';
  constructor(public modelId: string) {
    super(`Model ${modelId} is gated — provide a HuggingFace token.`);
  }
}

export class HFFetchError extends Error {
  override name = 'HFFetchError';
  constructor(message: string, public status?: number) {
    super(message);
  }
}

const CONFIG_URL = (id: string) => `https://huggingface.co/${id}/resolve/main/config.json`;
const INFO_URL = (id: string) => `https://huggingface.co/api/models/${id}`;

async function fetchJson<T>(url: string, token?: string): Promise<T> {
  const headers: Record<string, string> = { accept: 'application/json' };
  if (token) headers.authorization = `Bearer ${token}`;
  const res = await fetch(url, { headers });
  if (res.status === 401 || res.status === 403) {
    throw new GatedModelError(url);
  }
  if (!res.ok) {
    throw new HFFetchError(`HF fetch failed: ${res.status} ${res.statusText} (${url})`, res.status);
  }
  return (await res.json()) as T;
}

export async function fetchHF(
  modelId: string,
  token?: string,
): Promise<{ config: HFConfig; metadata: HFModelInfo }> {
  try {
    const [config, metadata] = await Promise.all([
      fetchJson<HFConfig>(CONFIG_URL(modelId), token),
      fetchJson<HFModelInfo>(INFO_URL(modelId), token),
    ]);
    return { config, metadata };
  } catch (err) {
    if (err instanceof GatedModelError) {
      throw new GatedModelError(modelId);
    }
    throw err;
  }
}
