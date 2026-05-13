import { describe, expect, it, beforeEach, afterEach } from 'vitest';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { readCache, writeCache, CACHE_CONSTANTS } from '../src/hf/cache.js';

describe('HF cache', () => {
  let dir: string;
  beforeEach(async () => {
    dir = await fs.mkdtemp(path.join(os.tmpdir(), 'llm-calc-cache-'));
  });
  afterEach(async () => {
    await fs.rm(dir, { recursive: true, force: true });
  });

  it('miss returns null', async () => {
    const r = await readCache('foo/bar', { dir });
    expect(r).toBeNull();
  });

  it('hit returns the entry', async () => {
    await writeCache(
      'foo/bar',
      {
        config: { architectures: ['LlamaForCausalLM'] },
        metadata: { modelId: 'foo/bar' },
      },
      { dir },
    );
    const r = await readCache('foo/bar', { dir });
    expect(r).not.toBeNull();
    expect(r!.config.architectures).toEqual(['LlamaForCausalLM']);
  });

  it('force_refresh bypasses', async () => {
    await writeCache('foo/bar', { config: {}, metadata: { modelId: 'foo/bar' } }, { dir });
    const r = await readCache('foo/bar', { dir, force_refresh: true });
    expect(r).toBeNull();
  });

  it('TTL eviction', async () => {
    const file = path.join(dir, 'foo__bar.json');
    await fs.writeFile(
      file,
      JSON.stringify({
        fetched_at: Date.now() - CACHE_CONSTANTS.TTL_MS - 1000,
        config: {},
        metadata: { modelId: 'foo/bar' },
      }),
    );
    const r = await readCache('foo/bar', { dir });
    expect(r).toBeNull();
  });
});
