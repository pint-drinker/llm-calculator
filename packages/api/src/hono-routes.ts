import type { Context, Hono } from 'hono';
import type { ZodTypeAny, output } from 'zod';
import { schemas } from '@llm-calc/core';
import {
  HttpError,
  doCalculate,
  doExplain,
  doMaxContext,
  doRecommend,
  doCompare,
  doListModels,
  doListGpus,
} from './handlers.js';
import { findModel, findGpu } from './registry.js';

type AppType = Hono<{ Bindings: Record<string, unknown> }>;

function parseOr400<S extends ZodTypeAny>(c: Context, schema: S, data: unknown): output<S> {
  const r = schema.safeParse(data);
  if (!r.success) {
    throw new HttpError(400, 'invalid_input', { issues: r.error.issues });
  }
  return r.data;
}

export function registerHonoRoutes(app: AppType): void {
  app.get('/api/health', (c) => c.json({ ok: true }));

  app.get('/api/models', (c) => {
    const q = c.req.query();
    const filter = schemas.listModelsFilterSchema.parse({
      min_params: q.min_params ? Number(q.min_params) : undefined,
      max_params: q.max_params ? Number(q.max_params) : undefined,
      architecture: q.architecture,
    });
    return c.json(doListModels(filter));
  });

  app.get('/api/models/:name', (c) => {
    const m = findModel(c.req.param('name'));
    if (!m) throw new HttpError(404, 'model_not_found', { name: c.req.param('name') });
    return c.json(m);
  });

  app.get('/api/gpus', (c) => c.json(doListGpus()));

  app.get('/api/gpus/:name', (c) => {
    const g = findGpu(c.req.param('name'));
    if (!g) throw new HttpError(404, 'gpu_not_found', { name: c.req.param('name') });
    return c.json(g);
  });

  app.post('/api/calculate', async (c) => {
    const args = parseOr400(c, schemas.calculateByNameRequestSchema, await c.req.json());
    return c.json(doCalculate(args));
  });

  app.post('/api/max-context', async (c) => {
    const args = parseOr400(c, schemas.maxContextRequestSchema, await c.req.json());
    return c.json(doMaxContext(args));
  });

  app.post('/api/recommend-gpus', async (c) => {
    const args = parseOr400(c, schemas.recommendRequestSchema, await c.req.json());
    return c.json({ candidates: doRecommend(args) });
  });

  app.post('/api/compare', async (c) => {
    const args = parseOr400(c, schemas.compareRequestSchema, await c.req.json());
    return c.json(doCompare(args.configs));
  });

  app.post('/api/explain', async (c) => {
    const args = parseOr400(c, schemas.explainRequestSchema, await c.req.json());
    return c.json(doExplain(args));
  });
}

export function handleError(err: unknown, c: Context): Response {
  if (err instanceof HttpError) {
    return c.json({ error: err.message, details: err.details }, err.status as never);
  }
  console.error('unhandled error', err);
  return c.json({ error: 'internal_error' }, 500);
}
