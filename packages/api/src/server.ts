import express, { type ErrorRequestHandler } from 'express';
import cors from 'cors';
import { router } from './routes.js';
import { HttpError } from './handlers.js';
import { attachMcpHttp } from './mcp/transport.js';

export function createApp(): express.Express {
  const app = express();
  app.use(cors());
  app.use(express.json({ limit: '1mb' }));
  app.use('/api', router);
  attachMcpHttp(app);

  const errorHandler: ErrorRequestHandler = (err, _req, res, _next) => {
    if (err instanceof HttpError) {
      res.status(err.status).json({ error: err.message, details: err.details });
      return;
    }
    // eslint-disable-next-line no-console
    console.error(err);
    res.status(500).json({ error: 'internal_error', details: { message: String(err?.message ?? err) } });
  };
  app.use(errorHandler);
  return app;
}
