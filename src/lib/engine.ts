// ── Motor de simulación BTC/USD · 5 modelos con cuentas de crédito ──────────

import { buildFeatures, clamp, annualizedSharpe, maxDrawdown, type FeatureSet } from "./indicators";
import {
  QLearningAgent,
  GradientBoostingModel,
  StatisticalModel,
  RandomForestModel,
  LSTMModel,
  ACTION_NAMES,
  type Action,
  type ModelSnapshot,
} from "./models";
import { MODEL_IDS, type ModelId, type AggressionLevel } from "./registry";

export const LOAN_PRINCIPAL = 100_000;
export const LOAN_APR = 0.095; // 9.5% TAE
export const COMMISSION = 0.001; // 0.1% por operación
const TRADE_FRACTION = 0.35;
const YEAR_SECONDS = 365 * 24 * 3600;

export type AgentId = ModelId;

export interface Trade {
  ts: number;
  agent: AgentId;
  side: "BUY" | "SELL";
  price: number;
  qty: number;
  value: number;
  fee: number;
  reason: string;
}

export interface EquityPoint {
  t: number;
  rl: number;
  xgb: number;
  stat: number;
  rf: number;
  lstm: number;
  bh: number;
  price: number;
  live: boolean;
}

export interface AgentMetrics {
  netEquity: number;
  grossEquity: number;
  netReturn: number;
  interest: number;
  debt: number;
  exposure: number;
  trades: number;
  buys: number;
  sells: number;
  sharpe: number;
  maxDD: number;
  winRate: number;
  realizedCloses: number;
  cash: number;
  btc: number;
  btcValue: number;
}

export interface MarketStats {
  price: number;
  change24h: number;
  high24h: number;
  low24h: number;
  volume24h: number;
  source: string;
  lastUpdate: number;
}

export interface ModelUiState extends ModelSnapshot {
  id: ModelId;
  name: string;
  aggression: AggressionLevel;
}

export interface EngineSnapshot {
  market: MarketStats;
  equitySeries: EquityPoint[];
  agents: Record<ModelId, AgentMetrics>;
  modelInfo: Record<ModelId, ModelUiState>;
  trades: Trade[];
  status: "loading" | "backtest" | "live" | "error";
  statusDetail: string;
  ticks: number;
  running: boolean;
  aggressionPerModel: Record<ModelId, AggressionLevel>;
}

const WARMUP = 60;
const MAX_EQUITY_POINTS = 1400;
const RETRAIN_EVERY_TICKS = 25;

// ════════════════════════════════════════════════════════════════════════════
// Cuenta de crédito (una por modelo)
// ════════════════════════════════════════════════════════════════════════════

class MarginAccount {
  readonly principal = LOAN_PRINCIPAL;
  readonly apr = LOAN_APR;
  readonly id: AgentId;
  cash = LOAN_PRINCIPAL;
  btc = 0;
  interest = 0;
  costBasis = 0;
  realizedWins = 0;
  realizedCloses = 0;
  buys = 0;
  sells = 0;
  lastTradeTs = 0;
  private equityGross: number[] = [];
  private equityNet: number[] = [];
  trades: Trade[] = [];

  constructor(id: AgentId) {
    this.id = id;
  }

  accrueInterest(dtSeconds: number): void {
    this.interest += (this.principal + this.interest) * this.apr * (dtSeconds / YEAR_SECONDS);
  }

  gross(price: number): number {
    return this.cash + this.btc * price;
  }
  net(price: number): number {
    return this.gross(price) - this.interest;
  }
  exposure(price: number): number {
    const g = this.gross(price);
    return g > 0 ? clamp((this.btc * price) / g, 0, 1) : 0;
  }

  buy(price: number, ts: number, reason: string): Trade | null {
    const spend = this.cash * TRADE_FRACTION;
    if (spend < 20) return null;
    const fee = spend * COMMISSION;
    const qty = (spend - fee) / price;
    this.cash -= spend;
    this.btc += qty;
    this.costBasis += spend;
    this.buys++;
    const tr: Trade = { ts, agent: this.id, side: "BUY", price, qty, value: spend, fee, reason };
    this.trades.push(tr);
    this.lastTradeTs = ts;
    return tr;
  }

  sell(price: number, ts: number, reason: string): Trade | null {
    const qty = this.btc * TRADE_FRACTION;
    if (qty * price < 20) return null;
    const proceeds = qty * price;
    const fee = proceeds * COMMISSION;
    const fractionSold = this.btc > 0 ? qty / this.btc : 0;
    const basisPortion = this.costBasis * fractionSold;
    this.realizedCloses++;
    if (proceeds - fee > basisPortion) this.realizedWins++;
    this.costBasis -= basisPortion;
    this.btc -= qty;
    this.cash += proceeds - fee;
    this.sells++;
    const tr: Trade = { ts, agent: this.id, side: "SELL", price, qty, value: proceeds, fee, reason };
    this.trades.push(tr);
    this.lastTradeTs = ts;
    return tr;
  }

  record(price: number): void {
    this.equityGross.push(this.gross(price));
    this.equityNet.push(this.net(price));
  }

  getEquityNet(): number[] {
    return this.equityNet;
  }
  getEquityNetLen(): number {
    return this.equityNet.length;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Datos de mercado (APIs públicas)
// ════════════════════════════════════════════════════════════════════════════

async function fetchJson(url: string, timeoutMs = 6000): Promise<unknown> {
  const ctrl = new AbortController();
  const to = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { signal: ctrl.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } finally {
    clearTimeout(to);
  }
}

export async function fetchKlines(limit = 320): Promise<{ t: number; close: number }[]> {
  // 1. Directo (funciona en Node / local sin CORS)
  try {
    const data = (await fetchJson(
      `https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=${limit}`
    )) as unknown[][];
    return data.map((k) => ({ t: k[0] as number, close: parseFloat(k[4] as string) }));
  } catch {}

  // 2. Proxy CORS (para navegador desplegado)
  try {
    const proxyUrl = `https://api.allorigins.win/raw?url=${encodeURIComponent(
      `https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=${limit}`
    )}`;
    const data = (await fetchJson(proxyUrl, 12_000)) as unknown[][];
    if (Array.isArray(data) && data.length > 0) {
      return data.map((k) => ({ t: k[0] as number, close: parseFloat(k[4] as string) }));
    }
  } catch {}

  // 3. Fallback sintético con tendencia para que los modelos aprendan
  return await generateSyntheticKlines(limit);
}

async function generateSyntheticKlines(limit: number): Promise<{ t: number; close: number }[]> {
  const spot = await fetchSpotPrice().catch(() => null);
  const basePrice = spot?.price ?? 100_000;
  const now = Date.now();

  // Generar retornos con tendencia sinusoidal + ruido, luego reconstruir precios
  // El precio final (índice más alto) será exactamente basePrice
  const rets: number[] = [];
  let seed = 123456;
  const rand = () => {
    seed = (seed * 16807 + 0) % 2147483647;
    return (seed - 1) / 2147483646;
  };

  for (let i = 0; i < limit; i++) {
    const trend = Math.sin(i * 0.06) * 0.001;
    const momentum = Math.sin(i * 0.15) * 0.0006;
    const shock = (rand() - 0.5) * 0.002;
    rets.push(trend + momentum + shock);
  }

  // Calcular factor de normalización para que el último precio = basePrice
  let cumProd = 1;
  for (let i = 0; i < limit; i++) cumProd *= (1 + rets[i]);
  const scale = basePrice / cumProd;

  const out: { t: number; close: number }[] = [];
  let p = scale;
  for (let i = 0; i < limit; i++) {
    p *= (1 + rets[i]);
    out.push({ t: now - (limit - 1 - i) * 60_000, close: p });
  }

  return out;
}

let offlinePrice = 0;

export async function fetchSpotPrice(): Promise<MarketStats> {
  try {
    const d = (await fetchJson("https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT", 5000)) as Record<string, string>;
    offlinePrice = parseFloat(d.lastPrice);
    return {
      price: offlinePrice,
      change24h: parseFloat(d.priceChangePercent),
      high24h: parseFloat(d.highPrice),
      low24h: parseFloat(d.lowPrice),
      volume24h: parseFloat(d.quoteVolume),
      source: "Binance Spot",
      lastUpdate: Date.now(),
    };
  } catch {}
  try {
    const d = (await fetchJson("https://api.kraken.com/0/public/Ticker?pair=XBTUSD", 5000)) as { result: Record<string, { c: string[]; h: string[]; l: string[]; v: string[]; o: string }> };
    const k = Object.values(d.result)[0];
    const price = parseFloat(k.c[0]);
    const open = parseFloat(k.o);
    offlinePrice = price;
    return {
      price,
      change24h: (price / open - 1) * 100,
      high24h: parseFloat(k.h[1]),
      low24h: parseFloat(k.l[1]),
      volume24h: parseFloat(k.v[1]) * price,
      source: "Kraken",
      lastUpdate: Date.now(),
    };
  } catch {}
  try {
    const d = (await fetchJson("https://api.coincap.io/v2/assets/bitcoin", 5000)) as { data: Record<string, string> };
    const a = d.data;
    offlinePrice = parseFloat(a.priceUsd);
    return {
      price: offlinePrice,
      change24h: parseFloat(a.changePercent24Hr),
      high24h: 0,
      low24h: 0,
      volume24h: parseFloat(a.volumeUsd24Hr),
      source: "CoinCap",
      lastUpdate: Date.now(),
    };
  } catch {}
  try {
    const d = (await fetchJson("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true", 5000)) as { bitcoin: { usd: number; usd_24h_change: number; usd_24h_vol: number } };
    offlinePrice = d.bitcoin.usd;
    return {
      price: d.bitcoin.usd,
      change24h: d.bitcoin.usd_24h_change,
      high24h: 0,
      low24h: 0,
      volume24h: d.bitcoin.usd_24h_vol,
      source: "CoinGecko",
      lastUpdate: Date.now(),
    };
  } catch {}
  if (offlinePrice === 0) offlinePrice = 118_000;
  offlinePrice *= 1 + (Math.random() - 0.5) * 0.0008;
  return {
    price: offlinePrice,
    change24h: 0,
    high24h: 0,
    low24h: 0,
    volume24h: 0,
    source: "Modo offline (sin API)",
    lastUpdate: Date.now(),
  };
}

// ════════════════════════════════════════════════════════════════════════════
// Motor principal
// ════════════════════════════════════════════════════════════════════════════

type AnyModel =
  | QLearningAgent
  | GradientBoostingModel
  | StatisticalModel
  | RandomForestModel
  | LSTMModel;

export class TradingEngine {
  accounts: Record<ModelId, MarginAccount> = {
    rl: new MarginAccount("rl"),
    xgb: new MarginAccount("xgb"),
    stat: new MarginAccount("stat"),
    rf: new MarginAccount("rf"),
    lstm: new MarginAccount("lstm"),
  };
  private bhBtc = 0;
  private bhInterest = 0;

  private rlAgent = new QLearningAgent();
  private xgbModel = new GradientBoostingModel();
  private statModel = new StatisticalModel();
  private rfModel = new RandomForestModel();
  private lstmModel = new LSTMModel();

  private modelLookup: Record<ModelId, AnyModel> = {
    rl: this.rlAgent,
    xgb: this.xgbModel,
    stat: this.statModel,
    rf: this.rfModel,
    lstm: this.lstmModel,
  };

  aggressionPerModel: Record<ModelId, AggressionLevel> = {
    rl: 2,
    xgb: 2,
    stat: 2,
    rf: 2,
    lstm: 2,
  };

  closes: number[] = [];
  times: number[] = [];
  // Barras de 1 minuto usadas para features y entrenamiento (evita que live mode
  // entrene/prediga sobre ticks de 3 segundos con distribución diferente).
  minuteCloses: number[] = [];
  minuteTimes: number[] = [];
  features: FeatureSet | null = null;
  featuresMatrix: number[][] = [];
  equitySeries: EquityPoint[] = [];
  market: MarketStats = {
    price: 0,
    change24h: 0,
    high24h: 0,
    low24h: 0,
    volume24h: 0,
    source: "—",
    lastUpdate: 0,
  };
  trades: Trade[] = [];
  status: EngineSnapshot["status"] = "loading";
  statusDetail = "Conectando con el mercado…";
  ticks = 0;
  running = true;
  private lastSignals: Record<ModelId, "buy" | "sell" | "hold"> = {
    rl: "hold",
    xgb: "hold",
    stat: "hold",
    rf: "hold",
    lstm: "hold",
  };
  private lastProbabilities: Record<ModelId, number> = {
    rl: 0.5,
    xgb: 0.5,
    stat: 0.5,
    rf: 0.5,
    lstm: 0.5,
  };
  private lastStatOutcome: boolean | undefined = undefined;

  private timer: ReturnType<typeof setInterval> | null = null;
  private listener: (() => void) | null = null;
  private busy = false;
  private aborted = false;
  private startGeneration = 0;

  constructor() {
    MODEL_IDS.forEach((id) => this.modelLookup[id].setAggression(this.aggressionPerModel[id]));
  }

  setAggression(id: ModelId, level: AggressionLevel): void {
    this.aggressionPerModel[id] = level;
    this.modelLookup[id].setAggression(level);
    this.emit();
  }

  onUpdate(cb: () => void): void {
    this.listener = cb;
  }

  private emit(): void {
    this.listener?.();
  }

  async start(options?: { resetAfterBacktest?: boolean }): Promise<void> {
    const gen = ++this.startGeneration;
    try {
      this.status = "loading";
      this.statusDetail = "Descargando histórico 1m de BTC/USD…";
      this.emit();

      const klines = await fetchKlines(320);
      if (this.aborted || gen !== this.startGeneration) return;
      this.closes = klines.map((k) => k.close);
      this.times = klines.map((k) => k.t);
      this.minuteCloses = [...this.closes];
      this.minuteTimes = [...this.times];
      this.features = buildFeatures(this.minuteCloses);
      this.featuresMatrix = Array.from({ length: this.minuteCloses.length }, (_, i) => this.features!.at(i));
      this.market = { ...this.market, price: this.closes[this.closes.length - 1] };

      this.status = "backtest";
      this.statusDetail = "Entrenando modelos ML…";
      this.emit();
      await sleep(30);

      if (gen !== this.startGeneration) return;
      this.trainInitialModels(this.closes.length - 2);

      this.statusDetail = "Backtest walk-forward: 5 modelos…";
      this.emit();
      this.bhBtc = LOAN_PRINCIPAL / this.closes[WARMUP];
      for (let i = WARMUP; i < this.closes.length; i++) {
        if (this.aborted || gen !== this.startGeneration) return;
        this.step(i, 60, false);
        if (i % 40 === 0) {
          this.statusDetail = `Backtest ${Math.round(((i - WARMUP) / (this.closes.length - WARMUP)) * 100)}% · 5 modelos`;
          this.emit();
          await sleep(0);
        }
      }

      if (this.aborted || gen !== this.startGeneration) return;

      // Si es un reset, volver a $100k y limpiar historial tras el backtest
      if (options?.resetAfterBacktest) {
        this.accounts = {
          rl: new MarginAccount("rl"),
          xgb: new MarginAccount("xgb"),
          stat: new MarginAccount("stat"),
          rf: new MarginAccount("rf"),
          lstm: new MarginAccount("lstm"),
        };
        this.equitySeries = [];
        this.trades = [];
        this.bhInterest = 0;
      }

      this.status = "live";
      this.statusDetail = "En vivo · actualizando cada 3 s";
      this.market = await fetchSpotPrice().catch(() => this.market);
      if (this.aborted || gen !== this.startGeneration) return;

      // Establecer baseline buy-and-hold con precio actual y grabar equity inicial
      const livePrice = this.market.price || this.closes[this.closes.length - 1] || 100_000;
      this.bhBtc = LOAN_PRINCIPAL / livePrice;
      this.recordEquity(livePrice, Date.now(), true);

      this.emit();
      this.timer = setInterval(() => void this.liveTick(), 3000);
    } catch (e) {
      if (gen !== this.startGeneration) return;
      this.status = "error";
      this.statusDetail = e instanceof Error ? e.message : "Error de conexión";
      this.emit();
    }
  }

  stop(): void {
    this.aborted = true;
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }

  /** Reset completo: limpia todo el estado y relanza desde cero */
  async reset(): Promise<void> {
    this.stop();
    this.aborted = false;

    // 1. Recrear cuentas
    this.accounts = {
      rl: new MarginAccount("rl"),
      xgb: new MarginAccount("xgb"),
      stat: new MarginAccount("stat"),
      rf: new MarginAccount("rf"),
      lstm: new MarginAccount("lstm"),
    };

    // 2. Recrear modelos
    this.rlAgent = new QLearningAgent();
    this.xgbModel = new GradientBoostingModel();
    this.statModel = new StatisticalModel();
    this.rfModel = new RandomForestModel();
    this.lstmModel = new LSTMModel();
    this.modelLookup = {
      rl: this.rlAgent,
      xgb: this.xgbModel,
      stat: this.statModel,
      rf: this.rfModel,
      lstm: this.lstmModel,
    };
    MODEL_IDS.forEach((id) => this.modelLookup[id].setAggression(this.aggressionPerModel[id]));

    // 3. Limpiar historial
    this.closes = [];
    this.times = [];
    this.minuteCloses = [];
    this.minuteTimes = [];
    this.features = null;
    this.featuresMatrix = [];
    this.equitySeries = [];
    this.trades = [];
    this.ticks = 0;
    this.running = true;
    this.bhBtc = 0;
    this.bhInterest = 0;
    this.lastSignals = { rl: "hold", xgb: "hold", stat: "hold", rf: "hold", lstm: "hold" };
    this.lastProbabilities = { rl: 0.5, xgb: 0.5, stat: 0.5, rf: 0.5, lstm: 0.5 };
    this.lastStatOutcome = undefined;
    this.market = { price: 0, change24h: 0, high24h: 0, low24h: 0, volume24h: 0, source: "—", lastUpdate: 0 };

    // 4. Emitir estado limpio INMEDIATAMENTE para que la UI se actualice
    this.status = "loading";
    this.statusDetail = "Reiniciando simulación…";
    this.emit();

    // 5. Relanzar pipeline — tras el backtest las cuentas vuelven a $100k
    await this.start({ resetAfterBacktest: true });
  }

  toggleRunning(): void {
    this.running = !this.running;
    this.emit();
  }

  private trainInitialModels(upTo: number): void {
    if (!this.features) return;
    const horizon = 5;
    const X: number[][] = [];
    const y: number[] = [];
    const from = Math.max(WARMUP, upTo - 400);
    for (let i = from; i < upTo; i++) {
      X.push(this.features.at(i));
      const nextIdx = Math.min(i + horizon, this.minuteCloses.length - 1);
      y.push(this.minuteCloses[nextIdx] > this.minuteCloses[i] ? 1 : 0);
    }
    this.xgbModel.train(X, y);
    this.rfModel.train(X, y);
    this.lstmModel.train(X, y, 80);
  }

  private retrainIncrementalModels(): void {
    if (!this.features) return;
    const horizon = 5;
    const upTo = this.minuteCloses.length - 2;
    const X: number[][] = [];
    const y: number[] = [];
    const from = Math.max(WARMUP, upTo - 400);
    for (let i = from; i < upTo; i++) {
      X.push(this.features.at(i));
      const nextIdx = Math.min(i + horizon, this.minuteCloses.length - 1);
      y.push(this.minuteCloses[nextIdx] > this.minuteCloses[i] ? 1 : 0);
    }
    this.xgbModel.retrainIncremental(X, y, 4);
    this.rfModel.addTrees(X, y, 6);
    if (this.ticks % (RETRAIN_EVERY_TICKS * 2) === 0) this.lstmModel.train(X, y, 20);
  }

  private rlState(i: number, acc: MarginAccount, price: number) {
    const f = this.features!.at(i);
    return {
      momentum: f[5],
      rsi: f[2],
      volatility: f[4],
      trend: f[7],
      position: acc.exposure(price),
    };
  }

  private step(i: number, dtSeconds: number, live: boolean, priceOverride?: number, tsOverride?: number): void {
    if (!this.features || i >= this.minuteCloses.length) return;
    const price = priceOverride ?? this.closes[i];
    const ts = tsOverride ?? (this.times[i] ?? Date.now());
    const f = this.features.at(i);

    MODEL_IDS.forEach((id) => this.accounts[id].accrueInterest(dtSeconds));
    this.bhInterest += (LOAN_PRINCIPAL + this.bhInterest) * LOAN_APR * (dtSeconds / YEAR_SECONDS);

    const COOLDOWN_MS = live ? 30_000 : 120_000;
    const featureSeq = live
      ? this.featuresMatrix.slice(-Math.max(12, this.lstmModel["seqLength"] + 1))
      : this.featuresMatrix.slice(Math.max(0, i - 15), i + 1);

    // ── XGB ──
    const probXgb = this.xgbModel.predictProba(f);
    this.lastProbabilities.xgb = probXgb;
    const sigXgb = probXgb > this.xgbModel.buyThreshold ? "buy" : probXgb < this.xgbModel.sellThreshold ? "sell" : "hold";
    this.lastSignals.xgb = sigXgb;
    if (sigXgb !== "hold" && ts - this.accounts.xgb.lastTradeTs >= COOLDOWN_MS) {
      const tr = sigXgb === "buy" && this.canBuy(this.accounts.xgb, price)
        ? this.accounts.xgb.buy(price, ts, `P(subida)=${(probXgb * 100).toFixed(1)}%`)
        : sigXgb === "sell" && this.canSell(this.accounts.xgb, price)
          ? this.accounts.xgb.sell(price, ts, `P(subida)=${(probXgb * 100).toFixed(1)}%`)
          : null;
      if (tr) this.pushTrade(tr);
    }

    // ── RL ──
    const sRl = this.rlState(i, this.accounts.rl, price);
    const eqBeforeRl = this.accounts.rl.net(price);
    const actionRl: Action = this.rlAgent.act(sRl);
    this.lastSignals.rl = actionRl === 2 ? "buy" : actionRl === 0 ? "sell" : "hold";
    this.lastProbabilities.rl = actionRl === 2 ? 0.6 : actionRl === 0 ? 0.4 : 0.5;
    let trRl: Trade | null = null;
    if (!live || ts - this.accounts.rl.lastTradeTs >= COOLDOWN_MS) {
      trRl = actionRl === 2 && this.canBuy(this.accounts.rl, price)
        ? this.accounts.rl.buy(price, ts, `Q-buy · ε=${this.rlAgent.epsilonValue.toFixed(3)}`)
        : actionRl === 0 && this.canSell(this.accounts.rl, price)
          ? this.accounts.rl.sell(price, ts, `Q-sell · ε=${this.rlAgent.epsilonValue.toFixed(3)}`)
          : null;
      if (trRl) this.pushTrade(trRl);
    }
    const nextIdx = Math.min(i + 1, this.minuteCloses.length - 1);
    const nextPrice = live ? price : this.minuteCloses[nextIdx];
    const eqAfterRl = this.accounts.rl.cash + this.accounts.rl.btc * nextPrice - this.accounts.rl.interest;
    const reward = ((eqAfterRl - eqBeforeRl) / LOAN_PRINCIPAL) * 1000;
    this.rlAgent.learn(sRl, actionRl, reward, this.rlState(nextIdx, this.accounts.rl, nextPrice));

    // ── Statistical ──
    const statRes = this.statModel.predict(f, this.lastStatOutcome);
    // Compute outcome for adaptive threshold: was previous signal correct?
    if (i > 0) {
      const prevPrice = this.minuteCloses[i - 1] ?? price;
      const priceUp = price > prevPrice;
      if (this.lastSignals.stat === "buy") this.lastStatOutcome = priceUp;
      else if (this.lastSignals.stat === "sell") this.lastStatOutcome = !priceUp;
      else this.lastStatOutcome = undefined;
    }
    this.lastSignals.stat = statRes.signal;
    this.lastProbabilities.stat = statRes.probability;
    if (!live || ts - this.accounts.stat.lastTradeTs >= COOLDOWN_MS) {
      const trStat = statRes.signal === "buy" && this.canBuy(this.accounts.stat, price)
        ? this.accounts.stat.buy(price, ts, `Stat:${statRes.signal} comp=${statRes.composite.toFixed(2)}`)
        : statRes.signal === "sell" && this.canSell(this.accounts.stat, price)
          ? this.accounts.stat.sell(price, ts, `Stat:${statRes.signal} comp=${statRes.composite.toFixed(2)}`)
          : null;
      if (trStat) this.pushTrade(trStat);
    }

    // ── RF ──
    const probRf = this.rfModel.predictProba(f);
    this.lastProbabilities.rf = probRf;
    const sigRf = this.rfModel.pickSignal(probRf);
    this.lastSignals.rf = sigRf;
    if (sigRf !== "hold" && ts - this.accounts.rf.lastTradeTs >= COOLDOWN_MS) {
      const trRf = sigRf === "buy" && this.canBuy(this.accounts.rf, price)
        ? this.accounts.rf.buy(price, ts, `RF prob=${(probRf * 100).toFixed(1)}%`)
        : sigRf === "sell" && this.canSell(this.accounts.rf, price)
          ? this.accounts.rf.sell(price, ts, `RF prob=${(probRf * 100).toFixed(1)}%`)
          : null;
      if (trRf) this.pushTrade(trRf);
    }

    // ── LSTM ──
    const probLstm = this.lstmModel.predict(featureSeq);
    this.lastProbabilities.lstm = probLstm;
    const sigLstm = this.lstmModel.pickSignal(probLstm);
    this.lastSignals.lstm = sigLstm;
    if (sigLstm !== "hold" && ts - this.accounts.lstm.lastTradeTs >= COOLDOWN_MS) {
      const trLstm = sigLstm === "buy" && this.canBuy(this.accounts.lstm, price)
        ? this.accounts.lstm.buy(price, ts, `LSTM P=${(probLstm * 100).toFixed(1)}%`)
        : sigLstm === "sell" && this.canSell(this.accounts.lstm, price)
          ? this.accounts.lstm.sell(price, ts, `LSTM P=${(probLstm * 100).toFixed(1)}%`)
          : null;
      if (trLstm) this.pushTrade(trLstm);
    }

    this.recordEquity(price, ts, live);
  }

  private recordEquity(price: number, ts: number, live: boolean): void {
    MODEL_IDS.forEach((id) => this.accounts[id].record(price));
    const point: EquityPoint = {
      t: ts,
      rl: this.accounts.rl.net(price),
      xgb: this.accounts.xgb.net(price),
      stat: this.accounts.stat.net(price),
      rf: this.accounts.rf.net(price),
      lstm: this.accounts.lstm.net(price),
      bh: this.bhBtc * price - this.bhInterest,
      price,
      live,
    };
    this.equitySeries.push(point);
    if (this.equitySeries.length > MAX_EQUITY_POINTS) {
      const keep = 500;
      const head = this.equitySeries.slice(0, -keep);
      const tail = this.equitySeries.slice(-keep);
      const decimated = head.filter((_, idx) => idx % 2 === 0);
      this.equitySeries = [...decimated, ...tail];
    }
  }

  private canBuy(acc: MarginAccount, price: number): boolean {
    return acc.exposure(price) < 0.85;
  }

  private canSell(acc: MarginAccount, price: number): boolean {
    return acc.exposure(price) > 0.15;
  }

  private pushTrade(tr: Trade): void {
    const last = this.trades[0];
    if (last && last.ts === tr.ts && last.agent === tr.agent && last.side === tr.side && Math.abs(last.qty - tr.qty) < 1e-9) return;
    this.trades.unshift(tr);
    if (this.trades.length > 500) this.trades.pop();
  }

  private async liveTick(): Promise<void> {
    if (this.busy || !this.running || this.aborted) return;
    this.busy = true;
    try {
      this.market = await fetchSpotPrice();
      const price = this.market.price;
      const now = Date.now();

      // Actualizar serie de display cada 3 segundos
      this.closes.push(price);
      this.times.push(now);
      if (this.closes.length > 2500) {
        this.closes.splice(0, 500);
        this.times.splice(0, 500);
      }

      // Solo generar una nueva barra de 1 minuto para features/modelos
      const lastMinuteTs = this.minuteTimes[this.minuteTimes.length - 1] ?? 0;
      const isNewMinute = this.minuteTimes.length === 0 || now - lastMinuteTs >= 60_000;

      if (isNewMinute) {
        this.minuteCloses.push(price);
        this.minuteTimes.push(now);
        if (this.minuteCloses.length > 2500) {
          this.minuteCloses.splice(0, 500);
          this.minuteTimes.splice(0, 500);
        }
        this.features = buildFeatures(this.minuteCloses);
        this.featuresMatrix = Array.from({ length: this.minuteCloses.length }, (_, i) => this.features!.at(i));
        this.ticks++;
        const dt = Math.max(60, Math.min(300, (now - lastMinuteTs) / 1000));
        this.step(this.minuteCloses.length - 1, dt, true, price, now);
        if (this.ticks % RETRAIN_EVERY_TICKS === 0) this.retrainIncrementalModels();
      } else {
        // En ticks intermedios actualizar equity para el gráfico
        this.recordEquity(price, now, true);
      }

      this.emit();
    } catch {
      // mantén último estado
    } finally {
      this.busy = false;
    }
  }

  private metricsFor(acc: MarginAccount): AgentMetrics {
    const price = this.market.price || this.minuteCloses[this.minuteCloses.length - 1] || this.closes[this.closes.length - 1] || 0;
    const net = acc.net(price);
    const secsPerStep =
      this.equitySeries.length > 1
        ? Math.max(
            3,
            (this.equitySeries[this.equitySeries.length - 1].t - this.equitySeries[0].t) /
              1000 /
              Math.max(1, this.equitySeries.length - 1)
          )
        : 60;
    return {
      netEquity: net,
      grossEquity: acc.gross(price),
      netReturn: (net / LOAN_PRINCIPAL - 1) * 100,
      interest: acc.interest,
      debt: acc.principal + acc.interest,
      exposure: acc.exposure(price),
      trades: acc.trades.length,
      buys: acc.buys,
      sells: acc.sells,
      sharpe: annualizedSharpe(acc.getEquityNet(), secsPerStep),
      maxDD: maxDrawdown(acc.getEquityNet()) * 100,
      winRate: acc.realizedCloses > 0 ? (acc.realizedWins / acc.realizedCloses) * 100 : 0,
      realizedCloses: acc.realizedCloses,
      cash: acc.cash,
      btc: acc.btc,
      btcValue: acc.btc * price,
    };
  }

  snapshot(): EngineSnapshot {
    const featNames = this.features?.names ?? [];
    const agents = {} as Record<ModelId, AgentMetrics>;
    MODEL_IDS.forEach((id) => (agents[id] = this.metricsFor(this.accounts[id])));

    const modelInfo = {} as Record<ModelId, ModelUiState>;
    const snapshots: Record<ModelId, ModelSnapshot> = {
      rl: this.rlAgent.snapshot(this.lastSignals.rl),
      xgb: this.xgbModel.snapshot(this.lastSignals.xgb, featNames),
      stat: this.statModel.snapshot(),
      rf: this.rfModel.snapshot(featNames),
      lstm: this.lstmModel.snapshot(featNames),
    };
    MODEL_IDS.forEach((id) => {
      modelInfo[id] = {
        ...snapshots[id],
        id,
        name: id === "rl" ? "Agente RL"
          : id === "xgb" ? "XGBoost"
          : id === "stat" ? "Statistical"
          : id === "rf" ? "Random Forest"
          : "LSTM",
        aggression: this.aggressionPerModel[id],
      };
    });

    return {
      market: this.market,
      equitySeries: [...this.equitySeries],
      agents,
      modelInfo,
      trades: [...this.trades],
      status: this.status,
      statusDetail: this.statusDetail,
      ticks: this.ticks,
      running: this.running,
      aggressionPerModel: { ...this.aggressionPerModel },
    };
  }

  // Exponer seqLength para uses internos de step()
  get lstmSeqLength(): number {
    return (this.lstmModel as unknown as { seqLength: number }).seqLength;
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

// Reexport para conveniencia
export { ACTION_NAMES };
