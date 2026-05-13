import { describe, expect, it } from 'vitest';
import request from 'supertest';
import { createApp } from '../src/server.js';

const app = createApp();

describe('REST API', () => {
  it('GET /api/health', async () => {
    const r = await request(app).get('/api/health');
    expect(r.status).toBe(200);
    expect(r.body).toEqual({ ok: true });
  });

  it('GET /api/models lists models', async () => {
    const r = await request(app).get('/api/models');
    expect(r.status).toBe(200);
    expect(Array.isArray(r.body)).toBe(true);
    expect(r.body.length).toBeGreaterThan(5);
    expect(r.body.find((m: { name: string }) => m.name === 'Qwen3.6-27B-Hybrid')).toBeTruthy();
  });

  it('GET /api/gpus lists GPUs', async () => {
    const r = await request(app).get('/api/gpus');
    expect(r.status).toBe(200);
    expect(r.body.find((g: { name: string }) => g.name === 'RTX 4090')).toBeTruthy();
  });

  it('POST /api/calculate (Qwen3.6 4-bit + fp8 KV @ 256K on RTX 4090)', async () => {
    const r = await request(app)
      .post('/api/calculate')
      .send({
        model_name: 'Qwen3.6-27B-Hybrid',
        weight_quant: 'awq_int4',
        kv_quant: 'fp8',
        context_length: 262144,
        batch_size: 1,
        tensor_parallel: 1,
        gpu_name: 'RTX 4090',
      });
    expect(r.status).toBe(200);
    expect(r.body.fits).toBe(true);
    expect(r.body.memory.weights_gb).toBeGreaterThan(13);
    expect(r.body.memory.weights_gb).toBeLessThan(17);
  });

  it('POST /api/explain returns steps', async () => {
    const r = await request(app)
      .post('/api/explain')
      .send({
        model_name: 'Llama-3.1-8B',
        weight_quant: 'bf16',
        kv_quant: 'bf16',
        context_length: 8192,
        batch_size: 1,
        tensor_parallel: 1,
        gpu_name: 'RTX 4090',
      });
    expect(r.status).toBe(200);
    expect(Array.isArray(r.body.steps)).toBe(true);
    expect(r.body.steps.length).toBeGreaterThan(5);
  });

  it('POST /api/recommend-gpus returns ranked candidates', async () => {
    const r = await request(app)
      .post('/api/recommend-gpus')
      .send({
        model_name: 'Llama-3.1-8B',
        weight_quant: 'bf16',
        kv_quant: 'bf16',
        context_length: 8192,
        batch_size: 1,
      });
    expect(r.status).toBe(200);
    expect(Array.isArray(r.body.candidates)).toBe(true);
    expect(r.body.candidates[0].fits).toBe(true);
  });

  it('POST /api/calculate rejects invalid input', async () => {
    const r = await request(app)
      .post('/api/calculate')
      .send({ weight_quant: 'banana', kv_quant: 'bf16', gpu_name: 'RTX 4090' });
    expect(r.status).toBe(400);
    expect(r.body.error).toBe('invalid_input');
  });

  it('GET /api/models/:name 404s for unknown', async () => {
    const r = await request(app).get('/api/models/UnknownModel123');
    expect(r.status).toBe(404);
  });
});
