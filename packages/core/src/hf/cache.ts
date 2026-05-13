import { promises as fs } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import type { HFConfig, HFModelInfo } from '../architectures/types.js';

const TTL_MS = 24 * 60 * 60 * 1000;

export interface CacheEntry {
  fetched_at: number;
  config: HFConfig;
  metadata: HFModelInfo;
}

export function defaultCacheDir(): string {
  if (process.env.LLM_CALC_HF_CACHE_DIR) return process.env.LLM_CALC_HF_CACHE_DIR;
  return path.join(os.homedir(), '.cache', 'llm-calc', 'hf');
}

function safeKey(modelId: string): string {
  return modelId.replace(/[^a-zA-Z0-9._-]/g, '__') + '.json';
}

export async function readCache(
  modelId: string,
  opts: { dir?: string; force_refresh?: boolean } = {},
): Promise<CacheEntry | null> {
  if (opts.force_refresh) return null;
  const dir = opts.dir ?? defaultCacheDir();
  const file = path.join(dir, safeKey(modelId));
  try {
    const raw = await fs.readFile(file, 'utf8');
    const entry = JSON.parse(raw) as CacheEntry;
    if (Date.now() - entry.fetched_at > TTL_MS) return null;
    return entry;
  } catch {
    return null;
  }
}

export async function writeCache(
  modelId: string,
  entry: Omit<CacheEntry, 'fetched_at'>,
  opts: { dir?: string } = {},
): Promise<void> {
  const dir = opts.dir ?? defaultCacheDir();
  await fs.mkdir(dir, { recursive: true });
  const file = path.join(dir, safeKey(modelId));
  const payload: CacheEntry = { fetched_at: Date.now(), ...entry };
  await fs.writeFile(file, JSON.stringify(payload), 'utf8');
}

export const CACHE_CONSTANTS = { TTL_MS };
