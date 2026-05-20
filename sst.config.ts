/// <reference path="./.sst/platform/config.d.ts" />

export default $config({
  app(input) {
    return {
      name: 'llm-calc',
      removal: input?.stage === 'production' ? 'retain' : 'remove',
      home: 'cloudflare',
      providers: {
        cloudflare: '6.15.0',
      },
    };
  },
  async run() {
    const limiter = new sst.cloudflare.RateLimit('McpLimit', {
      namespaceId: 1001,
      limit: 30,
      period: '1 minute',
    });

    const worker = new sst.cloudflare.Worker('LlmCalc', {
      handler: 'packages/api/src/worker.ts',
      url: true,
      link: [limiter],
      transform: {
        worker: {
          compatibilityDate: '2025-05-05',
          compatibilityFlags: ['nodejs_compat'],
          assets: {
            directory: '../../apps/web/dist',
            binding: 'ASSETS',
            notFoundHandling: 'single-page-application',
            runWorkerFirst: ['/api/*', '/mcp'],
          },
        },
      },
    });

    return {
      url: worker.url,
    };
  },
});
