import type { ModelConfig } from '../types.js';
import { dispatch } from '../architectures/index.js';
import { fetchHF } from './client.js';
import { readCache, writeCache } from './cache.js';

export interface ImportOptions {
  token?: string;
  force_refresh?: boolean;
  cache_dir?: string;
}

export interface ImportResult {
  model: ModelConfig;
  warnings: string[];
  adapter: string;
  from_cache: boolean;
}

export async function importFromHF(
  modelId: string,
  opts: ImportOptions = {},
): Promise<ImportResult> {
  const cached = await readCache(modelId, {
    dir: opts.cache_dir,
    force_refresh: opts.force_refresh,
  });
  let config;
  let metadata;
  let from_cache = false;
  if (cached) {
    config = cached.config;
    metadata = cached.metadata;
    from_cache = true;
  } else {
    const fetched = await fetchHF(modelId, opts.token);
    config = fetched.config;
    metadata = fetched.metadata;
    await writeCache(modelId, { config, metadata }, { dir: opts.cache_dir });
  }
  const { model, warnings, adapter } = dispatch(modelId, config, metadata);
  return { model, warnings, adapter, from_cache };
}
