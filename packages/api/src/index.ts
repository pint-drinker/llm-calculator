import { createApp } from './server.js';

const PORT = Number(process.env.PORT ?? 3001);
const app = createApp();
app.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`@llm-calc/api listening on http://localhost:${PORT}`);
  // eslint-disable-next-line no-console
  console.log(`  REST:  http://localhost:${PORT}/api`);
  // eslint-disable-next-line no-console
  console.log(`  MCP:   POST http://localhost:${PORT}/mcp`);
});
