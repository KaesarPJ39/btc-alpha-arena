// ── Indicadores técnicos sobre series de precios ─────────────────────────────

export function ema(values: number[], period: number): number[] {
  if (values.length === 0) return [];
  const k = 2 / (period + 1);
  const out: number[] = [values[0]];
  for (let i = 1; i < values.length; i++) {
    out.push(values[i] * k + out[i - 1] * (1 - k));
  }
  return out;
}

export function sma(values: number[], period: number): number[] {
  const out: number[] = [];
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= period) sum -= values[i - period];
    out.push(i >= period - 1 ? sum / period : values[i]);
  }
  return out;
}

export function rsi(closes: number[], period = 14): number[] {
  const out: number[] = new Array(closes.length).fill(50);
  let gain = 0;
  let loss = 0;
  for (let i = 1; i < closes.length; i++) {
    const d = closes[i] - closes[i - 1];
    const g = Math.max(d, 0);
    const l = Math.max(-d, 0);
    if (i <= period) {
      gain += g;
      loss += l;
      if (i === period) {
        gain /= period;
        loss /= period;
        out[i] = loss === 0 ? 100 : 100 - 100 / (1 + gain / loss);
      }
    } else {
      gain = (gain * (period - 1) + g) / period;
      loss = (loss * (period - 1) + l) / period;
      out[i] = loss === 0 ? 100 : 100 - 100 / (1 + gain / loss);
    }
  }
  return out;
}

export interface MacdResult {
  macd: number[];
  signal: number[];
  histogram: number[];
}

export function macd(closes: number[], fast = 12, slow = 26, sig = 9): MacdResult {
  const ef = ema(closes, fast);
  const es = ema(closes, slow);
  const m = closes.map((_, i) => ef[i] - es[i]);
  const s = ema(m, sig);
  return { macd: m, signal: s, histogram: m.map((v, i) => v - s[i]) };
}

/** Retornos simples (arithmetic) */
export function returns(closes: number[]): number[] {
  const out: number[] = [0];
  for (let i = 1; i < closes.length; i++) {
    out.push(closes[i] / closes[i - 1] - 1);
  }
  return out;
}

/** Volatilidad rodante (desviación típica de retornos) */
export function rollingVolatility(closes: number[], period = 20): number[] {
  const r = returns(closes);
  const out: number[] = new Array(closes.length).fill(0);
  for (let i = period; i < closes.length; i++) {
    const slice = r.slice(i - period, i);
    const mean = slice.reduce((a, b) => a + b, 0) / period;
    const varr = slice.reduce((a, b) => a + (b - mean) ** 2, 0) / period;
    out[i] = Math.sqrt(varr);
  }
  return out;
}

/** Average True Range simplificada (close-only) */
export function atr(closes: number[], period = 14): number[] {
  const out: number[] = new Array(closes.length).fill(0);
  const tr: number[] = [];
  for (let i = 1; i < closes.length; i++) {
    tr.push(Math.abs(closes[i] - closes[i - 1]));
  }
  for (let i = period; i < closes.length; i++) {
    const slice = tr.slice(i - period, i);
    out[i] = slice.reduce((a, b) => a + b, 0) / period;
  }
  return out;
}

/** Momentum: retorno en las últimas `period` barras */
export function momentum(closes: number[], period = 10): number[] {
  const out: number[] = new Array(closes.length).fill(0);
  for (let i = period; i < closes.length; i++) {
    out[i] = closes[i] / closes[i - period] - 1;
  }
  return out;
}

/** Vector de features para un índice dado (normalizados aprox. a [-1, 1]) */
export interface FeatureSet {
  names: string[];
  at: (i: number) => number[];
}

export function buildFeatures(closes: number[]): FeatureSet {
  const r = returns(closes);
  const rsiArr = rsi(closes, 14);
  const m = macd(closes);
  const vol = rollingVolatility(closes, 20);
  const mom5 = momentum(closes, 5);
  const mom15 = momentum(closes, 15);
  const e9 = ema(closes, 9);
  const e21 = ema(closes, 21);
  const e50 = ema(closes, 50);
  const atrArr = atr(closes, 14);

  const names = [
    "Retorno 1",
    "Retorno 3",
    "RSI-14",
    "MACD hist",
    "Volatilidad 20",
    "Momentum 5",
    "Momentum 15",
    "EMA9/EMA21",
    "EMA21/EMA50",
    "Precio/EMA50",
    "ATR-14",
  ];

  const at = (i: number): number[] => {
    const ret3 = i >= 3 ? closes[i] / closes[i - 3] - 1 : 0;
    const price = closes[i] || 1;
    return [
      clamp(r[i] * 80, -1, 1),
      clamp(ret3 * 40, -1, 1),
      (rsiArr[i] - 50) / 50,
      clamp((m.histogram[i] / price) * 400, -1, 1),
      clamp(vol[i] * 120, 0, 1),
      clamp(mom5[i] * 25, -1, 1),
      clamp(mom15[i] * 15, -1, 1),
      clamp(((e9[i] - e21[i]) / price) * 200, -1, 1),
      clamp(((e21[i] - e50[i]) / price) * 120, -1, 1),
      clamp(((price - e50[i]) / price) * 60, -1, 1),
      clamp(atrArr[i] / price * 1000, 0, 1),
    ];
  };

  return { names, at };
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

/** Sharpe anualizado a partir de equity muestreada cada `secondsPerStep` segundos */
export function annualizedSharpe(equity: number[], secondsPerStep: number): number {
  if (equity.length < 3) return 0;
  const r = returns(equity).slice(1).filter((x) => isFinite(x));
  if (r.length < 2) return 0;
  const mean = r.reduce((a, b) => a + b, 0) / r.length;
  const sd = Math.sqrt(r.reduce((a, b) => a + (b - mean) ** 2, 0) / (r.length - 1));
  if (sd === 0) return 0;
  const periodsPerYear = (365 * 24 * 3600) / secondsPerStep;
  return (mean / sd) * Math.sqrt(periodsPerYear);
}

export function maxDrawdown(equity: number[]): number {
  let peak = -Infinity;
  let mdd = 0;
  for (const v of equity) {
    if (v > peak) peak = v;
    const dd = peak > 0 ? (peak - v) / peak : 0;
    if (dd > mdd) mdd = dd;
  }
  return mdd;
}
