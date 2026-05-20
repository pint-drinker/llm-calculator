import { Router as createRouter, type Router } from 'express';
import { z } from 'zod';
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
import { findModel, registerModel, findGpu } from './registry.js';

function parseOr400<S extends z.ZodTypeAny>(schema: S, data: unknown): z.output<S> {
  const r = schema.safeParse(data);
  if (!r.success) {
    throw new HttpError(400, 'invalid_input', { issues: r.error.issues });
  }
  return r.data;
}

export const router: Router = createRouter();

router.get('/health', (_req, res) => {
  res.json({ ok: true });
});

router.get('/models', (req, res) => {
  const filter = schemas.listModelsFilterSchema.parse({
    min_params: req.query.min_params ? Number(req.query.min_params) : undefined,
    max_params: req.query.max_params ? Number(req.query.max_params) : undefined,
    architecture: req.query.architecture,
  });
  res.json(doListModels(filter));
});

router.get('/models/:name', (req, res, next) => {
  const m = findModel(req.params.name);
  if (!m) return next(new HttpError(404, 'model_not_found', { name: req.params.name }));
  res.json(m);
});

router.post('/models', (req, res) => {
  const model = parseOr400(schemas.modelConfigSchema, req.body);
  registerModel(model);
  res.status(201).json(model);
});

router.get('/gpus', (_req, res) => {
  res.json(doListGpus());
});

router.get('/gpus/:name', (req, res, next) => {
  const g = findGpu(req.params.name);
  if (!g) return next(new HttpError(404, 'gpu_not_found', { name: req.params.name }));
  res.json(g);
});

router.post('/calculate', (req, res) => {
  const args = parseOr400(schemas.calculateByNameRequestSchema, req.body);
  res.json(doCalculate(args));
});

router.post('/max-context', (req, res) => {
  const args = parseOr400(schemas.maxContextRequestSchema, req.body);
  res.json(doMaxContext(args));
});

router.post('/recommend-gpus', (req, res) => {
  const args = parseOr400(schemas.recommendRequestSchema, req.body);
  res.json({ candidates: doRecommend(args) });
});

router.post('/compare', (req, res) => {
  const args = parseOr400(schemas.compareRequestSchema, req.body);
  res.json(doCompare(args.configs));
});

router.post('/explain', (req, res) => {
  const args = parseOr400(schemas.explainRequestSchema, req.body);
  res.json(doExplain(args));
});
