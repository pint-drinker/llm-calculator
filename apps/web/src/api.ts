import type { ImportResult } from '@llm-calc/core';

export async function importHF(
  model_id: string,
  opts: { token?: string; force_refresh?: boolean } = {},
): Promise<ImportResult> {
  const res = await fetch('/api/models/import-hf', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model_id, ...opts }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: 'unknown' }));
    throw new Error(body.error === 'gated_model' ? `Gated model — provide a HuggingFace token.` : (body.error || 'import failed'));
  }
  return (await res.json()) as ImportResult;
}
