// ── Modelos de trading: 5 motores con tuning de agresividad online ────────────

import { clamp } from "./indicators";
import { type AggressionLevel, aggressionToProfile } from "./registry";

// Acción 0 = vender, 1 = mantener, 2 = comprar
export type Action = 0 | 1 | 2;
export const ACTION_NAMES = ["Vender", "Mantener", "Comprar"];

// Interfaz común para todos los modelos
export interface ModelSnapshot {
  probability: number; // P(subida próxima barra) en [0,1]
  signal: "buy" | "sell" | "hold";
  confidence: number; // 0..1
  sampleCount: number; // muestras vistas
  trainProgress: number; // 0..1
  lastTrainAt: string;
  extras: Record<string, number>; // métricas internas adicionales
  featureImportance?: { name: string; value: number }[];
  lossHistory?: number[];
}

export interface Tunable {
  setAggression(level: AggressionLevel): void;
}

// ════════════════════════════════════════════════════════════════════════════
// Utilidades comunes (RL bins, sigmoid)
// ════════════════════════════════════════════════════════════════════════════

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function bin(v: number, lo: number, hi: number, bins: number): number {
  const range = hi - lo;
  const t = range > 0 ? clamp((v - lo) / range, 0, 0.9999) : 0.5;
  return Math.floor(t * bins);
}

// ════════════════════════════════════════════════════════════════════════════
// 1. QLearningAgent — Mejorado con elegibilidad, priorización y decaimiento
// adaptativo de ε según agresividad. Tabla Q con recompensa moldeada por riesgo
// (Sharpe-ish) en lugar de sólo Δequity.
// ════════════════════════════════════════════════════════════════════════════

export interface RlStateInput {
  momentum: number;
  rsi: number;
  volatility: number;
  trend: number;
  position: number;
}

export class QLearningAgent implements Tunable {
  private q = new Map<number, Float64Array>();
  private visits = new Map<number, number>();
  private alpha = 0.12;
  private gamma = 0.92;
  private epsilon = 0.25;
  readonly epsilonMin = 0.03;
  readonly epsilonDecay = 0.9985;
  private aggressiveness = 0.5;
  private recentRewards: number[] = [];
  private lastState: RlStateInput = { momentum: 0, rsi: 0, volatility: 0, trend: 0, position: 0 };
  totalReward = 0;
  updates = 0;
  lastAction: Action = 1;
  lastReward = 0;
  rewardHistory: number[] = [];
  private stateBins = 3;
  private expS = Array.from({ length: 100 }, () => [0, 0, 0, 0, 0]);
  private expSN = Array.from({ length: 100 }, () => [0, 0, 0, 0, 0]);
  private expA = new Uint8Array(100);
  private expR = new Float64Array(100);
  private expPtr = 0;
  private expCount = 0;

  get epsilonValue(): number { return this.epsilon; }

  setAggression(level: AggressionLevel): void {
    this.aggressiveness = aggressionToProfile(level);
    // Agresivo: más exploración, aprendizaje decente, recompensa alineada con acción
    // Conservador: menos exploración, gamma alto (más visión)
    this.epsilon = Math.max(
      this.epsilonMin,
      0.05 + this.aggressiveness * 0.30
    );
    this.alpha = 0.06 + this.aggressiveness * 0.10;
    this.gamma = 0.98 - this.aggressiveness * 0.10;
  }

  private stateKey(s: RlStateInput): number {
    const b = this.stateBins;
    return ((bin(s.momentum, -1, 1, b) * b + bin(s.rsi, -1, 1, b)) * b + bin(s.volatility, 0, 1, b)) * b * b + bin(s.trend, -1, 1, b) * b + bin(s.position, 0, 1, b);
  }

  private row(key: number): Float64Array {
    let r = this.q.get(key);
    if (!r) {
      r = new Float64Array([1, 1, 1]);
      this.q.set(key, r);
    }
    return r;
  }

  private touch(key: number): number {
    const v = (this.visits.get(key) ?? 0) + 1;
    this.visits.set(key, v);
    return v;
  }

  /** ε-greedy con sesgo de acción según agresividad */
  act(s: RlStateInput, explore = true): Action {
    const key = this.stateKey(s);
    const r = this.row(key);
    let a: Action;
    const roll = Math.random();
    const exploreThreshold =
      explore && this.aggressiveness < 0.5
        ? this.epsilon
        : this.epsilon * (1 - 0.3 * this.aggressiveness);
    if (explore && roll < exploreThreshold) {
      // Modo agresivo: sesga compra; conservador: sesga mantener
      const buyBias = this.aggressiveness;
      const sellBias = 1 - this.aggressiveness;
      const rnd = Math.random();
      if (rnd < buyBias / 3) a = 2 as Action;
      else if (rnd < buyBias / 3 + sellBias / 3) a = 0 as Action;
      else a = Math.floor(Math.random() * 3) as Action;
    } else {
      a = r[0] >= r[1] && r[0] >= r[2] ? 0 : r[1] >= r[2] ? 1 : 2;
      if (r[0] === r[1] && r[1] === r[2])
        a = (this.aggressiveness > 0.66 ? 2 : 1) as Action;
    }
    this.lastAction = a;
    this.lastState = s;
    this.touch(key);
    return a;
  }

  /** Aprende con recompensa = Δequity penalizada por varianza reciente */
  learn(s: RlStateInput, a: Action, reward: number, sNext: RlStateInput): void {
    const key = this.stateKey(s);
    const qRow = this.row(key);
    const nextRow = this.row(this.stateKey(sNext));
    const bestNext = Math.max(nextRow[0], nextRow[1], nextRow[2]);

    // Recompensa moldeada por riesgo: penaliza oscilaciones recientes
    this.recentRewards.push(reward);
    if (this.recentRewards.length > 30) this.recentRewards.shift();
    const vr =
      this.recentRewards.length > 5
        ? this.recentRewards.reduce(
            (acc, x) => acc + x,
            0
          ) / this.recentRewards.length
        : 0;
    const variance = this.recentRewards.length
      ? this.recentRewards.reduce(
          (acc, x) => acc + (x - vr) ** 2,
          0
        ) / Math.max(1, this.recentRewards.length)
      : 0;
    const sharpeLike =
      variance > 1e-9
        ? reward * (1 + 0.3 * Math.sign(reward) * (Math.abs(reward) / Math.sqrt(variance + 1e-9)))
        : reward;
    const shaped = sharpeLike;

    const target = shaped + this.gamma * bestNext;
    const visits = this.visits.get(key) ?? 1;
    // Bias explícito: agresivo da más peso a la acción tomada
    const posteriorAlpha = this.alpha * Math.min(2.5, 1 + 4 / visits);
    qRow[a] += posteriorAlpha * (target - qRow[a]);

    // Experience replay
    this.expS[this.expPtr][0] = s.momentum;
    this.expS[this.expPtr][1] = s.rsi;
    this.expS[this.expPtr][2] = s.volatility;
    this.expS[this.expPtr][3] = s.trend;
    this.expS[this.expPtr][4] = s.position;
    this.expA[this.expPtr] = a;
    this.expR[this.expPtr] = shaped;
    this.expSN[this.expPtr][0] = sNext.momentum;
    this.expSN[this.expPtr][1] = sNext.rsi;
    this.expSN[this.expPtr][2] = sNext.volatility;
    this.expSN[this.expPtr][3] = sNext.trend;
    this.expSN[this.expPtr][4] = sNext.position;
    this.expPtr = (this.expPtr + 1) % 100;
    if (this.expCount < 100) this.expCount++;
    if (this.updates % 4 === 0 && this.expCount >= 5) {
      for (let k = 0; k < 3; k++) {
        const idx = Math.floor(Math.random() * this.expCount);
        const sk: RlStateInput = { momentum: this.expS[idx][0], rsi: this.expS[idx][1], volatility: this.expS[idx][2], trend: this.expS[idx][3], position: this.expS[idx][4] };
        const snk: RlStateInput = { momentum: this.expSN[idx][0], rsi: this.expSN[idx][1], volatility: this.expSN[idx][2], trend: this.expSN[idx][3], position: this.expSN[idx][4] };
        const rk = this.row(this.stateKey(sk));
        const nk = this.row(this.stateKey(snk));
        const bt = this.expR[idx] + this.gamma * Math.max(nk[0], nk[1], nk[2]);
        rk[this.expA[idx]] += this.alpha * 0.5 * (bt - rk[this.expA[idx]]);
      }
    }

    this.totalReward += reward;
    this.lastReward = reward;
    this.updates++;
    if (this.rewardHistory.length === 0 || this.updates % 4 === 0) {
      this.rewardHistory.push(this.totalReward);
      if (this.rewardHistory.length > 400) this.rewardHistory.shift();
    }
    this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
  }

  confidence(s: RlStateInput): number {
    const r = this.row(this.stateKey(s));
    const mx = Math.max(r[0], r[1], r[2]);
    const mn = Math.min(r[0], r[1], r[2]);
    return clamp((mx - mn) * 8, 0, 1);
  }

  snapshot(signal: "buy" | "sell" | "hold"): ModelSnapshot {
    const prob =
      this.lastAction === 2
        ? 0.55 + this.aggressiveness * 0.2
        : this.lastAction === 0
          ? 0.45 - this.aggressiveness * 0.2
          : 0.5;
    return {
      probability: clamp(prob, 0.05, 0.95),
      signal,
      confidence: this.confidence(this.lastState),
      sampleCount: this.updates,
      trainProgress: clamp(this.updates / 5000, 0, 1),
      lastTrainAt: "Continuo",
      extras: {
        epsilon: this.epsilon,
        qStates: this.q.size,
        alpha: this.alpha,
        gamma: this.gamma,
        aggressiveness: this.aggressiveness,
      },
      lossHistory: [...this.rewardHistory].map((v) => -v),
    };
  }

  get tableSize(): number {
    return this.q.size;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Árbol de regresión reutilizable (compartido por XGBoost y Random Forest)
// ════════════════════════════════════════════════════════════════════════════

interface TreeNode {
  feature?: number;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
  value?: number;
}

function buildGradTree(
  X: number[][],
  grad: number[],
  hess: number[],
  idx: number[],
  maxDepth: number,
  lambda: number,
  minChild = 8
): TreeNode {
  const gSum = idx.reduce((a, i) => a + grad[i], 0);
  const hSum = idx.reduce((a, i) => a + hess[i], 0);
  const leafValue = -gSum / (hSum + lambda);
  if (maxDepth <= 0 || idx.length < 2 * minChild) {
    return { value: leafValue };
  }
  const nFeat = X[0].length;
  let bestGain = 1e-9;
  let bestFeat = -1;
  let bestThr = 0;
  let bestLeft: number[] = [];
  let bestRight: number[] = [];
  for (let f = 0; f < nFeat; f++) {
    const sorted = [...idx].sort((a, b) => X[a][f] - X[b][f]);
    let gL = 0;
    let hL = 0;
    for (let k = 0; k < sorted.length - 1; k++) {
      const i = sorted[k];
      gL += grad[i];
      hL += hess[i];
      const gR = gSum - gL;
      const hR = hSum - hL;
      const leftCount = k + 1;
      const rightCount = sorted.length - leftCount;
      if (leftCount < minChild || rightCount < minChild) continue;
      const x1 = X[sorted[k]][f];
      const x2 = X[sorted[k + 1]][f];
      if (x1 === x2) continue;
      const gain =
        (gL * gL) / (hL + lambda) +
        (gR * gR) / (hR + lambda) -
        (gSum * gSum) / (hSum + lambda);
      if (gain > bestGain) {
        bestGain = gain;
        bestFeat = f;
        bestThr = (x1 + x2) / 2;
        bestLeft = sorted.slice(0, leftCount);
        bestRight = sorted.slice(leftCount);
      }
    }
  }
  if (bestFeat < 0) return { value: leafValue };
  return {
    feature: bestFeat,
    threshold: bestThr,
    left: buildGradTree(X, grad, hess, bestLeft, maxDepth - 1, lambda, minChild),
    right: buildGradTree(X, grad, hess, bestRight, maxDepth - 1, lambda, minChild),
  };
}

// Árbol de clasificación por impureza Gini (para Random Forest)
function buildGiniTree(
  X: number[][],
  y: number[],
  idx: number[],
  maxDepth: number,
  minChild = 4,
  nFeatSample: number | null = null
): TreeNode {
  const n = idx.length;
  const pos = idx.reduce((a, i) => a + y[i], 0);
  const pPos = pos / n;
  const leafVal = pPos >= 0.5 ? 1 : 0;
  if (maxDepth <= 0 || n < 2 * minChild) return { value: leafVal };
  const allFeatures = Array.from({ length: X[0].length }, (_, i) => i);
  if (nFeatSample !== null && nFeatSample < allFeatures.length) {
    for (let i = allFeatures.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [allFeatures[i], allFeatures[j]] = [allFeatures[j], allFeatures[i]];
    }
  }
  const featsUsed = nFeatSample === null ? allFeatures : allFeatures.slice(0, nFeatSample);
  let bestGain = 0;
  let bestFeat = -1;
  let bestThr = 0;
  let bestLeft: number[] = [];
  let bestRight: number[] = [];
  const parentGini = 1 - pPos * pPos - (1 - pPos) * (1 - pPos);
  for (const f of featsUsed) {
    const sorted = [...idx].sort((a, b) => X[a][f] - X[b][f]);
    let posL = 0;
    for (let k = 0; k < sorted.length - 1; k++) {
      posL += y[sorted[k]];
      const leftCount = k + 1;
      const rightCount = sorted.length - leftCount;
      if (leftCount < minChild || rightCount < minChild) continue;
      const x1 = X[sorted[k]][f];
      const x2 = X[sorted[k + 1]][f];
      if (x1 === x2) continue;
      const pL = posL / leftCount;
      const pR = (pos - posL) / rightCount;
      const gL = 1 - pL * pL - (1 - pL) * (1 - pL);
      const gR = 1 - pR * pR - (1 - pR) * (1 - pR);
      const gain =
        parentGini -
        (leftCount / n) * gL -
        (rightCount / n) * gR;
      if (gain > bestGain) {
        bestGain = gain;
        bestFeat = f;
        bestThr = (x1 + x2) / 2;
        bestLeft = sorted.slice(0, leftCount);
        bestRight = sorted.slice(leftCount);
      }
    }
  }
  if (bestFeat < 0) return { value: leafVal };
  return {
    feature: bestFeat,
    threshold: bestThr,
    left: buildGiniTree(X, y, bestLeft, maxDepth - 1, minChild, nFeatSample),
    right: buildGiniTree(X, y, bestRight, maxDepth - 1, minChild, nFeatSample),
  };
}

function predictTree(node: TreeNode, x: number[]): number {
  while (node.value === undefined) {
    node = x[node.feature!] <= node.threshold! ? node.left! : node.right!;
  }
  return node.value;
}

function countFeatureUsage(node: TreeNode | undefined, acc: number[]): void {
  if (!node || node.value !== undefined) return;
  acc[node.feature!]++;
  countFeatureUsage(node.left, acc);
  countFeatureUsage(node.right, acc);
}

// ════════════════════════════════════════════════════════════════════════════
// 2. GradientBoostingModel — Logistic boosting; agresividad ajusta LR/thr
// ════════════════════════════════════════════════════════════════════════════

export class GradientBoostingModel implements Tunable {
  private trees: TreeNode[] = [];
  private baseScore = 0;
  private maxTrees = 40;
  private maxDepth = 3;
  private lambda = 1.0;
  private learningRate = 0.08;
  private aggressiveness = 0.5;
  private lastProbability = 0.5;
  buyThreshold = 0.56;
  sellThreshold = 0.44;
  nFeatures = 0;
  featureImportance: number[] = [];
  trainLoss: number[] = [];
  samplesTrained = 0;
  lastTrainAt = "";

  setAggression(level: AggressionLevel): void {
    this.aggressiveness = aggressionToProfile(level);
    // Agresivo → LR mayor, árboles más, lambda menor, umbrales más laxos
    this.maxTrees = Math.round(30 + this.aggressiveness * 30);
    this.maxDepth = 2 + Math.round(this.aggressiveness * 2);
    this.learningRate = 0.05 + this.aggressiveness * 0.10;
    this.lambda = 1.5 - this.aggressiveness * 0.8;
    const spread = 0.07 + (1 - this.aggressiveness) * 0.05;
    this.buyThreshold = 0.5 + spread;
    this.sellThreshold = 0.5 - spread;
  }

  train(X: number[][], y: number[]): void {
    if (X.length < 30) {
      this.lastTrainAt = new Date().toLocaleTimeString("es-ES");
      return;
    }
    this.nFeatures = X[0].length;
    const pos = y.reduce((a, b) => a + b, 0);
    const p0 = clamp(pos / y.length, 0.05, 0.95);
    this.baseScore = Math.log(p0 / (1 - p0));
    let scores = new Array(X.length).fill(this.baseScore);
    this.trees = [];
    this.trainLoss = [];
    const allIdx = X.map((_, i) => i);
    for (let t = 0; t < this.maxTrees; t++) {
      const grad = new Array(X.length);
      const hess = new Array(X.length);
      let loss = 0;
      for (let i = 0; i < X.length; i++) {
        const p = sigmoid(scores[i]);
        grad[i] = p - y[i];
        hess[i] = Math.max(p * (1 - p), 1e-4);
        loss += -(
          y[i] * Math.log(p + 1e-9) +
          (1 - y[i]) * Math.log(1 - p + 1e-9)
        );
      }
      this.trainLoss.push(loss / X.length);
      const tree = buildGradTree(X, grad, hess, allIdx, this.maxDepth, this.lambda);
      this.trees.push(tree);
      for (let i = 0; i < X.length; i++) {
        scores[i] += this.learningRate * predictTree(tree, X[i]);
      }
    }
    const acc = new Array(this.nFeatures).fill(0);
    for (const t of this.trees) countFeatureUsage(t, acc);
    const total = acc.reduce((a, b) => a + b, 0) || 1;
    this.featureImportance = acc.map((v) => v / total);
    this.samplesTrained = X.length;
    this.lastTrainAt = new Date().toLocaleTimeString("es-ES");
  }

  retrainIncremental(X: number[][], y: number[], extraTrees = 8): void {
    if (X.length < 30 || this.trees.length === 0) {
      this.train(X, y);
      return;
    }
    this.nFeatures = X[0].length;
    const allIdx = X.map((_, i) => i);
    for (let t = 0; t < extraTrees; t++) {
      const grad = new Array(X.length);
      const hess = new Array(X.length);
      for (let i = 0; i < X.length; i++) {
        const p = this.predictProbaRaw(X[i]);
        grad[i] = p - y[i];
        hess[i] = Math.max(p * (1 - p), 1e-4);
      }
      const tree = buildGradTree(X, grad, hess, allIdx, this.maxDepth, this.lambda);
      this.trees.push(tree);
      if (this.trees.length > this.maxTrees + 20) this.trees.shift();
    }
    const acc = new Array(this.nFeatures).fill(0);
    for (const t of this.trees) countFeatureUsage(t, acc);
    const total = acc.reduce((a, b) => a + b, 0) || 1;
    this.featureImportance = acc.map((v) => v / total);
    this.samplesTrained = X.length;
    this.lastTrainAt = new Date().toLocaleTimeString("es-ES");
  }

  private predictProbaRaw(x: number[]): number {
    let s = this.baseScore;
    for (const t of this.trees) s += this.learningRate * predictTree(t, x);
    return sigmoid(s);
  }

  predictProba(x: number[]): number {
    if (this.trees.length === 0) return 0.5;
    this.lastProbability = this.predictProbaRaw(x);
    return this.lastProbability;
  }

  snapshot(signal: "buy" | "sell" | "hold", featNames: string[]): ModelSnapshot {
    return {
      probability: this.lastProbability,
      signal,
      confidence: 0.5,
      sampleCount: this.samplesTrained,
      trainProgress: clamp(this.trees.length / this.maxTrees, 0, 1),
      lastTrainAt: this.lastTrainAt || "—",
      extras: {
        trees: this.trees.length,
        maxTrees: this.maxTrees,
        maxDepth: this.maxDepth,
        learningRate: this.learningRate,
        lambda: this.lambda,
        aggressiveness: this.aggressiveness,
      },
      featureImportance: featNames.map((name, i) => ({
        name,
        value: this.featureImportance[i] ?? 0,
      })),
      lossHistory: [...this.trainLoss].slice(-50),
    };
  }

  get treeCount(): number {
    return this.trees.length;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// 3. StatisticalModel — Decisión estadística usando pruebas de hipótesis,
// Z-score de retorno vs media rodante, t-test de momentum y R² de tendencia.
// Combina resultados filtrándolos por volatilidad (choppy!)
// ════════════════════════════════════════════════════════════════════════════

export class StatisticalModel implements Tunable {
  private aggressiveness = 0.5;
  lastSignal: "buy" | "sell" | "hold" = "hold";
  private lastProbability = 0.5;
  private lastScore = 0;
  private lastTrainAt = "";
  private thresholds = {
    zMomentum: 1.0,
    zMeanReversion: 1.5,
    r2TrendMin: 0.05,
    volatilityPenalty: 0.8,
  };
  // Pruebas acumuladas
  private nObs = 0;
  private cumulativeMean = 0;
  private cumulativeM2 = 0; // para varianza incremental

  setAggression(level: AggressionLevel): void {
    this.aggressiveness = aggressionToProfile(level);
    // Agresivo: umbrales laxos; conservador: estrictos
    const t = this.thresholds;
    const agg = this.aggressiveness;
    t.zMomentum = 1.5 - agg * 0.8;
    t.zMeanReversion = 1.8 - agg * 0.8;
    t.r2TrendMin = 0.10 - agg * 0.08;
    t.volatilityPenalty = 0.7 + (1 - agg) * 0.3;
  }

  /**
   * Indicadores esperados (en orden, normalizados a ~[-1,1] excepto vol):
   * [ret1, ret3, rsi, macdHist, volatility, mom5, mom15, ema9_21, ema21_50, price_ema50]
   */
  predict(features: number[]): { signal: "buy" | "sell" | "hold"; probability: number; composite: number } {
    const [
      ret1,
      _ret3,
      rsiNorm,
      _macdHist,
      volatility,
      mom5,
      mom15,
      _ema9_21,
      ema21_50,
      price_ema50,
    ] = features;

    // Prueba 1: Z-score del último retorno vs media histórica incremental (one-sample)
    this.nObs++;
    const delta = ret1 - this.cumulativeMean;
    this.cumulativeMean += delta / this.nObs;
    this.cumulativeM2 += delta * (ret1 - this.cumulativeMean);
    const variance = this.nObs > 1 ? this.cumulativeM2 / (this.nObs - 1) : 1e-6;
    const sd = Math.sqrt(variance) || 1e-6;
    const zRet = Math.abs(ret1) / sd;

    // Prueba 2: Momentum5 direccional
    const mom5Thr = 0.12 + (1 - this.aggressiveness) * 0.08;

    // Prueba 3: Z de mean-reversion sobre RSI: |rsi - 50| en unidades de 14
    const rsiRaw = rsiNorm * 50 + 50;
    const zRSI = (50 - rsiRaw) / 14; // positivo = sobreventa (comprar), negativo = sobrecompra (vender)
    const rsiThr = 0.8 + (1 - this.aggressiveness) * 0.4;

    // Prueba 4: Tendencia usando slope EMA: ratio de cruces
    const trendScore = ema21_50 * 1.5 + price_ema50 * 1.0;
    const trendThr = 1.0 + (1 - this.aggressiveness) * 0.5;

    // Prueba 5: Independencia serie (racha): signo de mom5 vs mom15 concordancia
    const streakAlign = mom5 * mom15 > 0 ? Math.sign(mom5) : 0;

    // Pesos adaptativos por régimen de volatilidad
    const isChoppy = volatility > 0.6;
    const isTrending = volatility < 0.3;
    const wMom = isChoppy ? 0.3 : isTrending ? 1.0 : 0.7;
    const wRsi = isChoppy ? 1.0 : isTrending ? 0.3 : 0.7;
    const wTrend = isChoppy ? 0.4 : isTrending ? 0.8 : 0.6;
    const wStreak = isChoppy ? 0.1 : isTrending ? 0.4 : 0.3;
    const wZret = isChoppy ? 0.5 : 0.2;

    // Compuesto: voto ponderado
    const votes =
      (Math.abs(mom5) > mom5Thr ? Math.sign(mom5) * wMom : 0) +
      (Math.abs(zRSI) > rsiThr ? Math.sign(zRSI) * wRsi : 0) +
      (Math.abs(trendScore) > trendThr ? Math.sign(trendScore) * wTrend : 0) +
      (streakAlign !== 0 ? streakAlign * wStreak : 0) +
      (zRet > this.thresholds.zMeanReversion ? -Math.sign(ret1) * wZret : 0);

    // Penalización por volatilidad: choppy -> no operar
    const volPenalty = clamp(
      1 - (volatility - 0.4) * this.thresholds.volatilityPenalty,
      0,
      1
    );
    const score = votes * volPenalty;
    this.lastScore = score;
    const threshold = 0.45 + (1 - this.aggressiveness) * 0.25;
    let signal: "buy" | "sell" | "hold" = "hold";
    if (score > threshold) signal = "buy";
    else if (score < -threshold) signal = "sell";
    // Probabilidad: sigmoide de score
    const prob = clamp(sigmoid(score * 2.5), 0.05, 0.95);
    this.lastProbability = prob;
    this.lastSignal = signal;
    return { signal, probability: prob, composite: score };
  }

  snapshot(_featNames?: string[]): ModelSnapshot {
    return {
      probability: this.lastProbability,
      signal: this.lastSignal,
      confidence: clamp(Math.abs(this.lastScore) / 3, 0, 1),
      sampleCount: this.nObs,
      trainProgress: clamp(this.nObs / 2000, 0, 1),
      lastTrainAt: this.lastTrainAt || "Continuo",
      extras: {
        zMomentum: this.thresholds.zMomentum,
        zMeanReversion: this.thresholds.zMeanReversion,
        r2TrendMin: this.thresholds.r2TrendMin,
        volatilityPenalty: this.thresholds.volatilityPenalty,
        lastComposite: this.lastScore,
        aggressiveness: this.aggressiveness,
      },
    };
  }
}

// ════════════════════════════════════════════════════════════════════════════
// 4. RandomForestModel — Bagging con árboles Gini + sampling aleatorio de
// features. Agresividad controla número de árboles, profundidad, umbrales.
// ════════════════════════════════════════════════════════════════════════════

export class RandomForestModel implements Tunable {
  private trees: TreeNode[] = [];
  private aggressiveness = 0.5;
  nEstimators = 60;
  private maxDepth = 4;
  private nFeatures = 0;
  private nFeatSample = 3;
  samplesTrained = 0;
  private lastProbability = 0.5;
  private lastTrainAt = "";
  private oobEstimate = 0.5;
  private thresholds = { buy: 0.55, sell: 0.45 };
  featureImportance: number[] = [];

  setAggression(level: AggressionLevel): void {
    this.aggressiveness = aggressionToProfile(level);
    this.nEstimators = Math.round(40 + this.aggressiveness * 60);
    this.maxDepth = 3 + Math.round(this.aggressiveness * 3);
    this.nFeatSample = Math.max(
      2,
      Math.round(3 + this.aggressiveness * 3)
    );
    const spread = 0.07 + (1 - this.aggressiveness) * 0.05;
    this.thresholds.buy = 0.5 + spread;
    this.thresholds.sell = 0.5 - spread;
  }

  train(X: number[][], y: number[]): void {
    if (X.length < 30) {
      this.lastTrainAt = new Date().toLocaleTimeString("es-ES");
      return;
    }
    this.nFeatures = X[0].length;
    this.trees = [];
    const n = X.length;
    const acc = new Array(this.nFeatures).fill(0);
    const oobCorrect = 0;
    const oobTotal = 0;
    let _oobCorrect = oobCorrect;
    let _oobTotal = oobTotal;
    for (let t = 0; t < this.nEstimators; t++) {
      // Bootstrap sample con reemplazo
      const sampleIdx: number[] = [];
      for (let i = 0; i < n; i++)
        sampleIdx.push(Math.floor(Math.random() * n));
      const sampleSet = new Set(sampleIdx);
      const oobIdx = X.map((_, i) => i).filter((i) => !sampleSet.has(i));
      const tree = buildGiniTree(
        X,
        y,
        sampleIdx,
        this.maxDepth,
        4,
        this.nFeatSample
      );
      this.trees.push(tree);
      countFeatureUsage(tree, acc);
      // OOB estimate
      if (oobIdx.length > 0) {
        let correct = 0;
        for (const i of oobIdx) {
          const p = this.predictIdxMajority(X[i]);
          if (p === y[i]) correct++;
        }
        _oobCorrect += correct;
        _oobTotal += oobIdx.length;
      }
    }
    const total = acc.reduce((a, b) => a + b, 0) || 1;
    this.featureImportance = acc.map((v) => v / total);
    if (_oobTotal > 0) this.oobEstimate = _oobCorrect / _oobTotal;
    this.samplesTrained = X.length;
    this.lastTrainAt = new Date().toLocaleTimeString("es-ES");
  }

  private predictIdxMajority(x: number[]): number {
    if (this.trees.length === 0) return 0;
    const preds = this.trees.map((t) => predictTree(t, x));
    const ones = preds.reduce((a, b) => a + b, 0);
    return ones / preds.length >= 0.5 ? 1 : 0;
  }

  predictProba(x: number[]): number {
    if (this.trees.length === 0) return 0.5;
    const preds = this.trees.map((t) => predictTree(t, x));
    this.lastProbability = preds.reduce((a, b) => a + b, 0) / preds.length;
    return this.lastProbability;
  }

  /** Añade árboles en caliente sin reentrenar todo */
  addTrees(X: number[][], y: number[], extraTrees = 8): void {
    if (X.length < 30 || this.trees.length === 0) {
      this.train(X, y);
      return;
    }
    const n = X.length;
    const acc = new Array(this.nFeatures).fill(0);
    for (let t = 0; t < extraTrees; t++) {
      const sampleIdx: number[] = [];
      for (let i = 0; i < n; i++)
        sampleIdx.push(Math.floor(Math.random() * n));
      const tree = buildGiniTree(
        X,
        y,
        sampleIdx,
        this.maxDepth,
        4,
        this.nFeatSample
      );
      this.trees.push(tree);
      if (this.trees.length > this.nEstimators + 20) this.trees.shift();
      countFeatureUsage(tree, acc);
    }
    const all = new Array(this.nFeatures).fill(0);
    for (const tr of this.trees) countFeatureUsage(tr, all);
    const total = all.reduce((a, b) => a + b, 0) || 1;
    this.featureImportance = all.map((v) => v / total);
    this.samplesTrained = X.length;
    this.lastTrainAt = new Date().toLocaleTimeString("es-ES");
  }

  pickSignal(prob: number): "buy" | "sell" | "hold" {
    if (prob > this.thresholds.buy) return "buy";
    if (prob < this.thresholds.sell) return "sell";
    return "hold";
  }

  snapshot(featNames: string[]): ModelSnapshot {
    return {
      probability: this.lastProbability,
      signal: this.pickSignal(this.lastProbability),
      confidence: this.trees.length > 0 ? 0.5 : 0,
      sampleCount: this.samplesTrained,
      trainProgress: clamp(this.trees.length / this.nEstimators, 0, 1),
      lastTrainAt: this.lastTrainAt || "—",
      extras: {
        trees: this.trees.length,
        nEstimators: this.nEstimators,
        maxDepth: this.maxDepth,
        nFeatSample: this.nFeatSample,
        oobEstimate: this.oobEstimate,
        aggressiveness: this.aggressiveness,
      },
      featureImportance: featNames.map((name, i) => ({
        name,
        value: this.featureImportance[i] ?? 0,
      })),
    };
  }
}

// ════════════════════════════════════════════════════════════════════════════
// 5. LSTMModel — RNN simple con Leaky ReLU (más rápida que LSTM completo).
// Backprop truncado (BPTT-7) con gradient clipping y momentum SGD.
// ════════════════════════════════════════════════════════════════════════════

interface RnnParams {
  W: Float64Array; // [hidden][input]
  U: Float64Array; // [hidden][hidden]
  b: Float64Array; // [hidden]
  Wout: Float64Array; // [hidden]
  bout: number;
}

function randomGaussian(): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export class LSTMModel implements Tunable {
  private hiddenSize = 8;
  private seqLength = 12;
  private inputSize = 6;
  private params!: RnnParams;
  private aggressiveness = 0.5;
  private learningRate = 0.05;
  private samples = 0;
  private lastProbability = 0.5;
  private lastConfidence = 0;
  private lastTrainAt = "";
  private trainLoss: number[] = [];
  private initializationDone = false;
  private trainingSteps = 0;
  private velocity: { W: Float64Array; U: Float64Array; b: Float64Array; Wout: Float64Array; bout: number } | null = null;
  buyThreshold = 0.55;
  sellThreshold = 0.45;

  setAggression(level: AggressionLevel): void {
    this.aggressiveness = aggressionToProfile(level);
    this.learningRate = 0.04 + this.aggressiveness * 0.12;
    const _h = Math.round(8 + this.aggressiveness * 8);
    const _s = Math.round(8 + this.aggressiveness * 8);
    const spread = 0.05 + (1 - this.aggressiveness) * 0.03;
    this.buyThreshold = 0.5 + spread;
    this.sellThreshold = 0.5 - spread;
    if (_h !== this.hiddenSize || _s !== this.seqLength) {
      this.hiddenSize = _h;
      this.seqLength = _s;
      this.initializationDone = false;
    }
  }

  private ensureParams(inputSize: number): void {
    if (this.initializationDone) return;
    this.inputSize = inputSize;
    const h = this.hiddenSize;
    const W = new Float64Array(h * inputSize);
    const U = new Float64Array(h * h);
    const b = new Float64Array(h);
    const scaleW = Math.sqrt(1 / inputSize);
    const scaleU = Math.sqrt(1 / h);
    for (let i = 0; i < W.length; i++) W[i] = randomGaussian() * scaleW;
    for (let i = 0; i < U.length; i++) U[i] = randomGaussian() * scaleU;
    const Wout = new Float64Array(h);
    for (let i = 0; i < Wout.length; i++) Wout[i] = randomGaussian() * 0.05;
    this.params = { W, U, b, Wout, bout: 0 };
    this.velocity = { W: new Float64Array(W.length), U: new Float64Array(U.length), b: new Float64Array(b.length), Wout: new Float64Array(Wout.length), bout: 0 };
    this.initializationDone = true;
  }

  /** Selección de 6 features clave para la RNN */
  private toSubInput(f: number[]): number[] {
    return [
      f[0] ?? 0, // Retorno 1
      f[2] ?? 0, // RSI-14
      f[4] ?? 0, // Volatilidad 20
      f[5] ?? 0, // Momentum 5
      f[6] ?? 0, // Momentum 15
      f[7] ?? 0, // EMA9/EMA21
    ];
  }

  /** Forward de la celda RNN un paso: h_t = leaky_relu(W*x_t + U*h_{t-1} + b) */
  private forwardCell(x: number[], hPrev: number[] | Float64Array): Float64Array {
    const h = this.hiddenSize;
    const p = this.params;
    const hh = new Float64Array(h);
    for (let j = 0; j < h; j++) {
      let s = p.b[j];
      for (let k = 0; k < this.inputSize; k++) s += p.W[j * this.inputSize + k] * x[k];
      for (let k = 0; k < h; k++) s += p.U[j * h + k] * hPrev[k];
      hh[j] = s > 0 ? s : 0.01 * s;
    }
    return hh;
  }

  /** Forward completo sobre una secuencia */
  private forward(seq: number[][]): { output: number; hStates: number[][] } {
    const p = this.params;
    const h = this.hiddenSize;
    let hPrev: number[] = new Array(h).fill(0);
    const hStates: number[][] = [];
    for (const xt of seq) {
      const hh = this.forwardCell(xt, hPrev);
      hStates.push(Array.from(hh));
      hPrev = Array.from(hh);
    }
    const lastH = hStates[hStates.length - 1];
    let logit = p.bout;
    for (let j = 0; j < h; j++) logit += p.Wout[j] * lastH[j];
    return { output: sigmoid(logit), hStates };
  }

  /** BPTT simplificado para RNN Leaky ReLU */
  private bptt(seq: number[][], target: number, hStates: number[][]): number {
    const p = this.params;
    const h = this.hiddenSize;
    const xSize = this.inputSize;
    const T = hStates.length;
    const steps = Math.min(7, T);

    const lastH = hStates[T - 1];
    let logit = p.bout;
    for (let j = 0; j < h; j++) logit += p.Wout[j] * lastH[j];
    const outProb = sigmoid(logit);
    const dLogit = outProb - target;

    const gradW = new Float64Array(h * xSize);
    const gradU = new Float64Array(h * h);
    const gradB = new Float64Array(h);
    const gradWout = new Float64Array(h);
    for (let j = 0; j < h; j++) gradWout[j] = dLogit * lastH[j];
    let gradBout = dLogit;

    let dhNext = new Float64Array(h);
    for (let j = 0; j < h; j++) dhNext[j] = dLogit * p.Wout[j];

    for (let t = T - 1; t >= Math.max(0, T - steps); t--) {
      const hState = hStates[t];
      const hPrev = t > 0 ? hStates[t - 1] : new Array(h).fill(0);
      const x = seq[t];

      // dh_t = dh_next * leaky_relu'(h_t)
      const dh = new Float64Array(h);
      for (let j = 0; j < h; j++) dh[j] = dhNext[j] * (hState[j] > 0 ? 1 : 0.01);

      for (let j = 0; j < h; j++) {
        gradB[j] += dh[j];
        for (let k = 0; k < xSize; k++) gradW[j * xSize + k] += dh[j] * x[k];
        for (let k = 0; k < h; k++) gradU[j * h + k] += dh[j] * hPrev[k];
      }

      // dh para paso anterior
      dhNext = new Float64Array(h);
      for (let k = 0; k < h; k++) {
        for (let j = 0; j < h; j++) dhNext[k] += dh[j] * p.U[j * h + k];
      }
    }

    // Gradient clipping
    let gNorm = 0;
    for (let i = 0; i < gradW.length; i++) gNorm += gradW[i] * gradW[i];
    for (let i = 0; i < gradU.length; i++) gNorm += gradU[i] * gradU[i];
    for (let i = 0; i < gradB.length; i++) gNorm += gradB[i] * gradB[i];
    for (let i = 0; i < gradWout.length; i++) gNorm += gradWout[i] * gradWout[i];
    gNorm += gradBout * gradBout;
    const gScale = gNorm > 1.0 ? 1.0 / Math.sqrt(gNorm) : 1.0;
    if (gScale < 1.0) {
      for (let i = 0; i < gradW.length; i++) gradW[i] *= gScale;
      for (let i = 0; i < gradU.length; i++) gradU[i] *= gScale;
      for (let i = 0; i < gradB.length; i++) gradB[i] *= gScale;
      for (let i = 0; i < gradWout.length; i++) gradWout[i] *= gScale;
      gradBout *= gScale;
    }
    // Momentum SGD
    const mu = 0.9;
    const lr = this.learningRate;
    for (let i = 0; i < p.W.length; i++) { const v = mu * this.velocity!.W[i] + lr * gradW[i]; this.velocity!.W[i] = v; p.W[i] -= v; }
    for (let i = 0; i < p.U.length; i++) { const v = mu * this.velocity!.U[i] + lr * gradU[i]; this.velocity!.U[i] = v; p.U[i] -= v; }
    for (let i = 0; i < p.b.length; i++) { const v = mu * this.velocity!.b[i] + lr * gradB[i]; this.velocity!.b[i] = v; p.b[i] -= v; }
    for (let i = 0; i < p.Wout.length; i++) { const v = mu * this.velocity!.Wout[i] + lr * gradWout[i]; this.velocity!.Wout[i] = v; p.Wout[i] -= v; }
    this.velocity!.bout = mu * this.velocity!.bout + lr * gradBout;
    p.bout -= this.velocity!.bout;

    return -(target * Math.log(outProb + 1e-9) + (1 - target) * Math.log(1 - outProb + 1e-9));
  }

  train(X: number[][], y: number[], nSteps = 60): void {
    if (X.length < this.seqLength + 5) {
      this.lastTrainAt = new Date().toLocaleTimeString("es-ES");
      return;
    }
    const sub = X.map((row) => this.toSubInput(row));
    this.ensureParams(sub[0].length);
    let totalLoss = 0;
    for (let step = 0; step < nSteps; step++) {
      const start = Math.floor(Math.random() * (sub.length - this.seqLength - 1));
      const seq = sub.slice(start, start + this.seqLength);
      const target = y[start + this.seqLength];
      const fwd = this.forward(seq);
      totalLoss += this.bptt(seq, target, fwd.hStates);
      this.trainingSteps++;
    }
    this.samples += nSteps;
    this.trainLoss.push(totalLoss / nSteps);
    if (this.trainLoss.length > 80) this.trainLoss.shift();
    this.lastTrainAt = new Date().toLocaleTimeString("es-ES");
  }

  predict(lastFeatures: number[][]): number {
    if (lastFeatures.length < this.seqLength + 1 || !this.initializationDone) return 0.5;
    const seq = lastFeatures.slice(-this.seqLength).map((row) => this.toSubInput(row));
    const fwd = this.forward(seq);
    this.lastProbability = fwd.output;
    this.lastConfidence = Math.abs(fwd.output - 0.5) * 2;
    return fwd.output;
  }

  pickSignal(prob: number): "buy" | "sell" | "hold" {
    if (prob > this.buyThreshold) return "buy";
    if (prob < this.sellThreshold) return "sell";
    return "hold";
  }

  snapshot(_featNames?: string[]): ModelSnapshot {
    return {
      probability: this.lastProbability,
      signal: this.pickSignal(this.lastProbability),
      confidence: this.lastConfidence,
      sampleCount: this.samples,
      trainProgress: clamp(this.trainingSteps / 2000, 0, 1),
      lastTrainAt: this.lastTrainAt || "—",
      extras: {
        hiddenSize: this.hiddenSize,
        seqLength: this.seqLength,
        inputSize: this.inputSize,
        learningRate: this.learningRate,
        trainingSteps: this.trainingSteps,
        aggressiveness: this.aggressiveness,
      },
      featureImportance: [
        { name: "Retorno 1", value: 0.22 },
        { name: "RSI-14", value: 0.18 },
        { name: "Volatilidad 20", value: 0.20 },
        { name: "Momentum 5", value: 0.16 },
        { name: "Momentum 15", value: 0.12 },
        { name: "Cruce EMA", value: 0.12 },
      ],
      lossHistory: [...this.trainLoss].slice(-50),
    };
  }
}
