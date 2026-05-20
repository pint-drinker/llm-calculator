import type { MiddlewareHandler } from 'hono';

interface RateLimitBinding {
  limit: (opts: { key: string }) => Promise<{ success: boolean }>;
}

interface RateLimitEnv {
  MCP_RATE_LIMITER?: RateLimitBinding;
}

export function rateLimit(): MiddlewareHandler<{ Bindings: RateLimitEnv }> {
  return async (c, next) => {
    const limiter = c.env.MCP_RATE_LIMITER;
    if (!limiter) return next();
    const ip = c.req.header('CF-Connecting-IP') ?? c.req.header('X-Forwarded-For') ?? 'unknown';
    const { success } = await limiter.limit({ key: ip });
    if (!success) {
      return c.json(
        { error: 'rate_limited', detail: 'Too many requests. Try again shortly.' },
        429,
        { 'Retry-After': '60' },
      );
    }
    return next();
  };
}
