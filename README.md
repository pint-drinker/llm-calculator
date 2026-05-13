# llm-calculator

Local-only VRAM & throughput calculator for LLM inference. First-class support for hybrid attention (linear + quadratic per-layer), independent weight/KV quantization, tensor parallelism, and MoE. Three surfaces over one calculation engine: REST API, MCP server, React widget.

## Quick start

```bash
pnpm install
pnpm dev
```

- Vite at http://localhost:5173
- Express at http://localhost:3001
- Vite proxies `/api/*` to Express.
- MCP exposed at `POST http://localhost:3001/mcp` (Streamable HTTP) and via a stdio entry for Claude Desktop / Claude Code.

## Workspaces

```
packages/core    pure TS calc engine (zod only runtime dep)
packages/api     Express REST + MCP server
apps/web         React + Vite frontend
```

## Tests

```bash
pnpm test
```

The core engine validates against published model spec sheets — see `packages/core/test/qwen36-validation.test.ts`.

## MCP integration

For Claude Desktop / Claude Code, after `pnpm build`:

```json
{
  "mcpServers": {
    "llm-calc": {
      "command": "node",
      "args": ["/path/to/llm-calculator/packages/api/dist/mcp-stdio.js"]
    }
  }
}
```

For HTTP-based agents:

```
POST http://localhost:3001/mcp
```

## Goals

- **Inspectable math** — every number on screen is traceable to a formula with substituted values.
- **Hybrid attention** — correct numbers for models with mixed full + linear attention layers, not just dense GQA.
- **Independent quantization** — FP8 KV on int4 weights, etc.
- **MoE-aware** — active params drive compute, total params drive memory.
- **HuggingFace import** — paste a model ID, get a ready-to-use `ModelConfig` from `config.json` + safetensors metadata.

## Non-goals

- Training memory (no optimizer states / gradients / backward activations).
- Cost modeling.
- Multi-node distributed serving.
- Auth or persistence beyond a session-scoped in-memory custom model store.
