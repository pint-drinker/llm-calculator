import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { WebStandardStreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js';
import { createMcpServer } from './mcp/server.js';
import { registerHonoRoutes, handleError } from './hono-routes.js';
import { rateLimit } from './middleware/ratelimit.js';

type Env = {
  MCP_RATE_LIMITER?: { limit: (opts: { key: string }) => Promise<{ success: boolean }> };
  ASSETS?: { fetch: (request: Request) => Promise<Response> };
};

const app = new Hono<{ Bindings: Env }>();

app.use('*', cors({ origin: '*', allowHeaders: ['content-type', 'authorization'] }));
app.use('/api/*', rateLimit());
app.use('/mcp', rateLimit());

registerHonoRoutes(app);

app.all('/mcp', async (c) => {
  const transport = new WebStandardStreamableHTTPServerTransport({
    sessionIdGenerator: undefined,
    enableJsonResponse: true,
  });
  const server = createMcpServer();
  await server.connect(transport);
  const response = await transport.handleRequest(c.req.raw);
  c.executionCtx.waitUntil(
    (async () => {
      await transport.close();
      await server.close();
    })(),
  );
  return response;
});

app.onError(handleError);

app.notFound((c) => {
  if (c.env.ASSETS) return c.env.ASSETS.fetch(c.req.raw);
  return c.json({ error: 'not_found' }, 404);
});

export default app;
