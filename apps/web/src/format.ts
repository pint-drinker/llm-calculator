export function fmtGB(n: number): string {
  if (!Number.isFinite(n)) return '—';
  if (n < 0.001) return `${(n * 1000).toFixed(2)} MB`;
  if (n < 1) return `${(n * 1000).toFixed(0)} MB`;
  if (n < 10) return `${n.toFixed(2)} GB`;
  return `${n.toFixed(1)} GB`;
}

export function fmtTok(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return '—';
  if (n < 1) return `${n.toFixed(2)} tok/s`;
  if (n < 100) return `${n.toFixed(1)} tok/s`;
  return `${Math.round(n)} tok/s`;
}

export function fmtSeconds(n: number): string {
  if (!Number.isFinite(n)) return '—';
  if (n < 0.01) return `${(n * 1000).toFixed(1)} ms`;
  if (n < 1) return `${(n * 1000).toFixed(0)} ms`;
  if (n < 60) return `${n.toFixed(2)} s`;
  return `${Math.floor(n / 60)}m ${(n % 60).toFixed(0)}s`;
}

export function fmtContext(n: number): string {
  if (n >= 1_048_576) return `${(n / 1_048_576).toFixed(2)} M`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)} K`;
  return String(n);
}

export function fmtNumber(n: number): string {
  if (!Number.isFinite(n)) return '—';
  if (Math.abs(n) >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (Math.abs(n) >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (Math.abs(n) >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toFixed(2);
}
