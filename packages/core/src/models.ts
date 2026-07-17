// ── Modelos de trading: 5 motores con tuning de agresividad online ────────────
// Paquete core: sin dependencias de React/DOM. Uso de JS/TS estándar.

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
// Utilidades compartidas
// ════════════════════════════════════════════════════════════════════════════

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-Math.min(Math.max(x, -50), 50)));
}

function bin(v: number, lo: number, hi: number, bins: number): number {
  const range = hi - lo;
  const t = range > 0 ? clamp((v - lo) / range, 0, 0.9999) : 0.5;
  return Math.floor(t * bins);
}

function randomGaussian(): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Recompensa ajustada por riesgo para el agente RL. */
export function computeRiskAdjustedReward(
  eqBefore: number,
  eqAfter: number,
  volatility: number,
  loanPrincipal = 10000
): number {
  return ((eqAfter - eqBefore) / loanPrincipal) / Math.max(volatility, 0.01) * 100;
}

// ════════════════════════════════════════════════════════════════════════════
// 1. QLearningAgent — Tabular Q-learning con ε-greedy, elegibilidad,
// experience replay y recompensa ajustada por riesgo.
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
  private lambda = 0.7;
  private epsilon = 0.25;
  readonly epsilonMin = 0.01;
  readonly epsilonDecay = 0.9985;
  private aggressiveness = 0.5;
  private lastState: RlStateInput = { momentum: 0, rsi: 0, volatility: 0, trend: 0, position: 0 };
  totalReward = 0;
  updates = 0;
  lastAction: Action = 1;
  lastReward = 0;
  rewardHistory: number[] = [];
  private stateBins = 5;
  private expS = Array.from({ length: 100 }, () => [0, 0, 0, 0, 0]);
  private expSN = Array.from({ length: 100 }, () => [0, 0, 0, 0, 0]);
  private expA = new Uint8Array(100);
  private expR = new Float64Array(100);
  private expPtr = 0;
  private expCount = 0;
  // Eligibility traces: ring buffer of recent (stateKey, action) pairs
  private traceKeys: number[] = new Array(12).fill(0);
  private traceActions: number[] = new Array(12).fill(0);
  private traceLen = 0;
  private tracePtr = 0;
  private readonly TRACE_CAP = 12;

  get epsilonValue(): number {
    return this.epsilon;
  }

  setAggression(level: AggressionLevel): void {
    this.aggressiveness = aggressionToProfile(level);
    this.epsilon = Math.max(this.epsilonMin, 0.05 + this.aggressiveness * 0.30);
    this.alpha = 0.06 + this.aggressiveness * 0.10;
    this.gamma = 0.98 - this.aggressiveness * 0.10;
  }

  private stateKey(s: RlStateInput): number {
    const b = this.stateBins;
    return (
      (((bin(s.momentum, -1, 1, b) * b + bin(s.rsi, -1, 1, b)) * b +
        bin(s.volatility, 0, 1, b)) *
        b +
        bin(s.trend, -1, 1, b)) *
        b +
      bin(s.position, 0, 1, b)
    );
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

  /** Aprende con eligibility traces (λ-return), experience replay y decaimiento de ε. */
  learn(s: RlStateInput, a: Action, reward: number, sNext: RlStateInput): void {
    const key = this.stateKey(s);
    const qRow = this.row(key);
    const nextRow = this.row(this.stateKey(sNext));
    const bestNext = Math.max(nextRow[0], nextRow[1], nextRow[2]);

    const target = reward + this.gamma * bestNext;
    const delta = target - qRow[a];

    this.traceKeys[this.tracePtr] = key;
    this.traceActions[this.tracePtr] = a;
    this.tracePtr = (this.tracePtr + 1) % this.TRACE_CAP;
    if (this.traceLen < this.TRACE_CAP) this.traceLen++;

    const gammaLambda = this.gamma * this.lambda;
    for (let i = 0; i < this.traceLen; i++) {
      const age = this.traceLen - 1 - i;
      const influence = Math.pow(gammaLambda, age);
      const tKey =
        this.traceKeys[(this.tracePtr - this.traceLen + i + this.TRACE_CAP) % this.TRACE_CAP];
      const tAct =
        this.traceActions[(this.tracePtr - this.traceLen + i + this.TRACE_CAP) % this.TRACE_CAP];
      const tRow = this.row(tKey);
      tRow[tAct] += this.alpha * delta * influence;
    }

    this.expS[this.expPtr][0] = s.momentum;
    this.expS[this.expPtr][1] = s.rsi;
    this.expS[this.expPtr][2] = s.volatility;
    this.expS[this.expPtr][3] = s.trend;
    this.expS[this.expPtr][4] = s.position;
    this.expA[this.expPtr] = a;
    this.expR[this.expPtr] = reward;
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
        const sk: RlStateInput = {
          momentum: this.expS[idx][0],
          rsi: this.expS[idx][1],
          volatility: this.expS[idx][2],
          trend: this.expS[idx][3],
          position: this.expS[idx][4],
        };
        const snk: RlStateInput = {
          momentum: this.expSN[idx][0],
          rsi: this.expSN[idx][1],
          volatility: this.expSN[idx][2],
          trend: this.expSN[idx][3],
          position: this.expSN[idx][4],
        };
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

  toJSON(): unknown {
    const entries: Array<[number, number[]]> = [];
    for (const [k, v] of this.q.entries()) {
      entries.push([k, Array.from(v)]);
    }
    return {
      alpha: this.alpha,
      gamma: this.gamma,
      lambda: this.lambda,
      epsilon: this.epsilon,
      aggressiveness: this.aggressiveness,
      totalReward: this.totalReward,
      updates: this.updates,
      lastAction: this.lastAction,
      lastReward: this.lastReward,
      rewardHistory: [...this.rewardHistory],
      qEntries: entries,
      visits: Array.from(this.visits.entries()),
      lastState: this.lastState,
    };
  }

  static fromJSON(data: unknown): QLearningAgent {
    const d = data as Record<string, unknown>;
    const agent = new QLearningAgent();
    if (typeof d.alpha === "number") agent.alpha = d.alpha;
    if (typeof d.gamma === "number") agent.gamma = d.gamma;
    if (typeof d.lambda === "number") agent.lambda = d.lambda;
    if (typeof d.epsilon === "number") agent.epsilon = d.epsilon;
    if (typeof d.aggressiveness === "number") agent.aggressiveness = d.aggressiveness;
    if (typeof d.totalReward === "number") agent.totalReward = d.totalReward;
    if (typeof d.updates === "number") agent.updates = d.updates;
    if (typeof d.lastAction === "number") agent.lastAction = clamp(d.lastAction, 0, 2) as Action;
    if (typeof d.lastReward === "number") agent.lastReward = d.lastReward;
    if (Array.isArray(d.rewardHistory)) agent.rewardHistory = d.rewardHistory.map((v) => Number(v));
    if (Array.isArray(d.qEntries)) {
      for (const entry of d.qEntries) {
        if (!Array.isArray(entry) || entry.length !== 2) continue;
        const [k, v] = entry as [number, number[]];
        agent.q.set(k, new Float64Array(v.map((n) => Number(n))));
      }
    }
    if (Array.isArray(d.visits)) {
      for (const entry of d.visits) {
        if (!Array.isArray(entry) || entry.length !== 2) continue;
        const [k, v] = entry as [number, number];
        agent.visits.set(k, Number(v));
      }
    }
    if (d.lastState && typeof d.lastState === "object") {
      const s = d.lastState as Record<string, number>;
      agent.lastState = {
        momentum: s.momentum ?? 0,
        rsi: s.rsi ?? 0,
        volatility: s.volatility ?? 0,
        trend: s.trend ?? 0,
        position: s.position ?? 0,
      };
    }
    return agent;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Árboles de regresión y gradient boosting compartidos
// ════════════════════════════════════════════════════════════════════════════

export interface TreeNode {
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

function buildRegTree(
  X: number[][],
  y: number[],
  idx: number[],
  maxDepth: number,
  minChild = 4,
  nFeatSample: number | null = null
): TreeNode {
  const n = idx.length;
  const sum = idx.reduce((a, i) => a + y[i], 0);
  const mean = sum / n;
  if (maxDepth <= 0 || n < 2 * minChild) return { value: mean };
  const allFeatures = Array.from({ length: X[0].length }, (_, i) => i);
  if (nFeatSample !== null && nFeatSample < allFeatures.length) {
    for (let i = allFeatures.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [allFeatures[i], allFeatures[j]] = [allFeatures[j], allFeatures[i]];
    }
  }
  const featsUsed = nFeatSample === null ? allFeatures : allFeatures.slice(0, nFeatSample);
  const parentVar = idx.reduce((a, i) => a + (y[i] - mean) * (y[i] - mean), 0) / n;
  const totalSumSq = idx.reduce((a, i) => a + y[i] * y[i], 0);
  let bestGain = 0;
  let bestFeat = -1;
  let bestThr = 0;
  let bestLeft: number[] = [];
  let bestRight: number[] = [];
  for (const f of featsUsed) {
    const sorted = [...idx].sort((a, b) => X[a][f] - X[b][f]);
    let sumL = 0;
    let sumSqL = 0;
    for (let k = 0; k < sorted.length - 1; k++) {
      const i = sorted[k];
      sumL += y[i];
      sumSqL += y[i] * y[i];
      const leftCount = k + 1;
      const rightCount = sorted.length - leftCount;
      if (leftCount < minChild || rightCount < minChild) continue;
      const x1 = X[sorted[k]][f];
      const x2 = X[sorted[k + 1]][f];
      if (x1 === x2) continue;
      const meanL = sumL / leftCount;
      const sumR = sum - sumL;
      const sumSqR = totalSumSq - sumSqL;
      const meanR = sumR / rightCount;
      const varL = sumSqL / leftCount - meanL * meanL;
      const varR = sumSqR / rightCount - meanR * meanR;
      const gain = parentVar - (leftCount / n) * varL - (rightCount / n) * varR;
      if (gain > bestGain) {
        bestGain = gain;
        bestFeat = f;
        bestThr = (x1 + x2) / 2;
        bestLeft = sorted.slice(0, leftCount);
        bestRight = sorted.slice(leftCount);
      }
    }
  }
  if (bestFeat < 0) return { value: mean };
  return {
    feature: bestFeat,
    threshold: bestThr,
    left: buildRegTree(X, y, bestLeft, maxDepth - 1, minChild, nFeatSample),
    right: buildRegTree(X, y, bestRight, maxDepth - 1, minChild, nFeatSample),
  };
}

function predictTree(node: TreeNode, x: number[]): number {
  let current: TreeNode = node;
  while (current.value === undefined) {
    current = x[current.feature!] <= current.threshold! ? current.left! : current.right!;
  }
  return current.value;
}

function countFeatureUsage(node: TreeNode | undefined, acc: number[]): void {
  if (!node || node.value !== undefined) return;
  acc[node.feature!]++;
  countFeatureUsage(node.left, acc);
  countFeatureUsage(node.right, acc);
}

function treeToJson(node: TreeNode): unknown {
  if (node.value !== undefined) return { value: node.value };
  return {
    feature: node.feature,
    threshold: node.threshold,
    left: treeToJson(node.left!),
    right: treeToJson(node.right!),
  };
}

function treeFromJson(data: unknown): TreeNode | undefined {
  if (!data || typeof data !== "object") return undefined;
  const d = data as Record<string, unknown>;
  if (typeof d.value === "number") return { value: d.value };
  const left = treeFromJson(d.left);
  const right = treeFromJson(d.right);
  if (typeof d.feature === "number" && typeof d.threshold === "number" && left && right) {
    return { feature: d.feature, threshold: d.threshold, left, right };
  }
  return undefined;
}

// ════════════════════════════════════════════════════════════════════════════
// 2. GradientBoostingModel — Ensamble multi-horizonte (1, 5, 15 barras)
// ════════════════════════════════════════════════════════════════════════════

interface HorizonModel {
  horizon: number;
  trees: TreeNode[];
  baseScore: number;
  trainLoss: number[];
  treeLoss: number[]; // log-loss reciente por árbol, para pruning
}

function emptyHorizon(horizon: number): HorizonModel {
  return {
    horizon,
    trees: [],
    baseScore: 0,
    trainLoss: [],
    treeLoss: [],
  };
}

export class GradientBoostingModel implements Tunable {
  private horizons: HorizonModel[] = [emptyHorizon(1), emptyHorizon(5), emptyHorizon(15)];
  private maxTrees = 40;
  private maxDepth = 3;
  private lambda = 1.0;
  private learningRate = 0.08;
  private baseLearningRate = 0.08;
  private subsample = 0.8;
  private aggressiveness = 0.5;
  private lastProbability = 0.5;
  buyThreshold = 0.56;
  sellThreshold = 0.44;
  nFeatures = 0;
  featureImportance: number[] = [];
  samplesTrained = 0;
  lastTrainAt = "";

  setAggression(level: AggressionLevel): void {
    this.aggressiveness = aggressionToProfile(level);
    this.maxTrees = Math.round(30 + this.aggressiveness * 30);
    this.maxDepth = 2 + Math.round(this.aggressiveness * 2);
    this.learningRate = 0.05 + this.aggressiveness * 0.10;
    this.baseLearningRate = this.learningRate;
    this.lambda = 1.5 - this.aggressiveness * 0.8;
    const spread = 0.07 + (1 - this.aggressiveness) * 0.05;
    this.buyThreshold = 0.5 + spread;
    this.sellThreshold = 0.5 - spread;
  }

  private computeHorizonTargets(returns: number[], horizon: number): number[] {
    const y: number[] = new Array(returns.length).fill(0);
    for (let i = 0; i < returns.length - horizon; i++) {
      let sum = 0;
      for (let h = 1; h <= horizon; h++) sum += returns[i + h] ?? 0;
      y[i] = sum > 0 ? 1 : 0;
    }
    return y;
  }

  private fitHorizon(X: number[][], y: number[], horizon: number): HorizonModel {
    const pos = y.reduce((a, b) => a + b, 0);
    const p0 = clamp(pos / y.length, 0.05, 0.95);
    const model = emptyHorizon(horizon);
    model.baseScore = Math.log(p0 / (1 - p0));
    const scores = new Array(X.length).fill(model.baseScore);
    model.trees = [];
    model.trainLoss = [];
    model.treeLoss = [];
    const n = X.length;
    const subSize = Math.round(n * this.subsample);
    let bestLoss = Infinity;
    let noImprove = 0;
    const maxTrees = this.maxTrees;
    for (let t = 0; t < maxTrees; t++) {
      const subIdx: number[] = [];
      const used = new Set<number>();
      for (let k = 0; k < subSize; k++) {
        let r: number;
        do {
          r = Math.floor(Math.random() * n);
        } while (used.has(r));
        used.add(r);
        subIdx.push(r);
      }
      const grad: number[] = new Array(n);
      const hess: number[] = new Array(n);
      let loss = 0;
      for (let i = 0; i < n; i++) {
        const p = sigmoid(scores[i]);
        grad[i] = p - y[i];
        hess[i] = Math.max(p * (1 - p), 1e-4);
        loss += -(y[i] * Math.log(p + 1e-9) + (1 - y[i]) * Math.log(1 - p + 1e-9));
      }
      model.trainLoss.push(loss / n);
      if (loss / n < bestLoss - 0.001) {
        bestLoss = loss / n;
        noImprove = 0;
      } else {
        noImprove++;
        if (noImprove >= 5 && model.trees.length >= 10) break;
      }
      const tree = buildGradTree(X, grad, hess, subIdx, this.maxDepth, this.lambda);
      model.trees.push(tree);
      model.treeLoss.push(loss / n);
      for (let i = 0; i < n; i++) {
        scores[i] += this.learningRate * predictTree(tree, X[i]);
      }
    }
    return model;
  }

  /** Entrena 3 sub-modelos para horizontes 1, 5 y 15 barras. `returns` debe estar alineado con X. */
  train(X: number[][], returns: number[]): void {
    if (X.length < 30 || returns.length < X.length + 1) {
      this.lastTrainAt = new Date().toLocaleTimeString("es-ES");
      return;
    }
    this.nFeatures = X[0].length;
    this.horizons = [emptyHorizon(1), emptyHorizon(5), emptyHorizon(15)];
    for (let h = 0; h < this.horizons.length; h++) {
      const horizon = this.horizons[h].horizon;
      const y = this.computeHorizonTargets(returns, horizon);
      this.horizons[h] = this.fitHorizon(X, y, horizon);
    }
    this.updateFeatureImportance();
    this.samplesTrained = X.length;
    this.lastTrainAt = new Date().toLocaleTimeString("es-ES");
  }

  private predictHorizon(horizon: number, x: number[]): number {
    const m = this.horizons.find((h) => h.horizon === horizon);
    if (!m || m.trees.length === 0) return 0.5;
    let s = m.baseScore;
    for (const t of m.trees) s += this.learningRate * predictTree(t, x);
    return sigmoid(s);
  }

  private totalTreeCount(): number {
    return this.horizons.reduce((a, h) => a + h.trees.length, 0);
  }

  private logLossForTree(model: HorizonModel, treeIndex: number, X: number[][], y: number[]): number {
    let loss = 0;
    for (let i = 0; i < X.length; i++) {
      let s = model.baseScore;
      for (let t = 0; t < model.trees.length; t++) {
        if (t === treeIndex) continue;
        s += this.learningRate * predictTree(model.trees[t], X[i]);
      }
      const p = sigmoid(s);
      loss += -(y[i] * Math.log(p + 1e-9) + (1 - y[i]) * Math.log(1 - p + 1e-9));
    }
    return loss / X.length;
  }

  private pruneWorstTrees(X: number[][], returns: number[]): void {
    while (this.totalTreeCount() > this.maxTrees + 20) {
      let worst: { horizonIdx: number; treeIdx: number; loss: number } | null = null;
      for (let hi = 0; hi < this.horizons.length; hi++) {
        const model = this.horizons[hi];
        if (model.trees.length <= 1) continue;
        const y = this.computeHorizonTargets(returns, model.horizon);
        for (let ti = 0; ti < model.trees.length; ti++) {
          const loss = this.logLossForTree(model, ti, X, y);
          if (!worst || loss < worst.loss) {
            worst = { horizonIdx: hi, treeIdx: ti, loss };
          }
        }
      }
      if (!worst) break;
      const model = this.horizons[worst.horizonIdx];
      model.trees.splice(worst.treeIdx, 1);
      model.treeLoss.splice(worst.treeIdx, 1);
    }
  }

  private updateFeatureImportance(): void {
    const acc = new Array(this.nFeatures).fill(0);
    for (const h of this.horizons) {
      for (const t of h.trees) countFeatureUsage(t, acc);
    }
    const total = acc.reduce((a, b) => a + b, 0) || 1;
    this.featureImportance = acc.map((v) => v / total);
  }

  retrainIncremental(X: number[][], returns: number[], extraTrees = 8): void {
    if (X.length < 30 || returns.length < X.length + 1 || this.totalTreeCount() === 0) {
      this.train(X, returns);
      return;
    }
    this.nFeatures = X[0].length;
    const n = X.length;
    const subSize = Math.round(n * this.subsample);
    this.learningRate = Math.max(this.baseLearningRate * 0.2, this.learningRate * 0.97);
    for (let hi = 0; hi < this.horizons.length; hi++) {
      const model = this.horizons[hi];
      const horizon = model.horizon;
      const y = this.computeHorizonTargets(returns, horizon);
      const scores = new Array(n).fill(model.baseScore);
      for (const t of model.trees) {
        for (let i = 0; i < n; i++) scores[i] += this.learningRate * predictTree(t, X[i]);
      }
      for (let t = 0; t < extraTrees; t++) {
        const subIdx: number[] = [];
        const used = new Set<number>();
        for (let k = 0; k < subSize; k++) {
          let r: number;
          do {
            r = Math.floor(Math.random() * n);
          } while (used.has(r));
          used.add(r);
          subIdx.push(r);
        }
        const grad: number[] = new Array(n);
        const hess: number[] = new Array(n);
        let loss = 0;
        for (let i = 0; i < n; i++) {
          const p = sigmoid(scores[i]);
          grad[i] = p - y[i];
          hess[i] = Math.max(p * (1 - p), 1e-4);
          loss += -(y[i] * Math.log(p + 1e-9) + (1 - y[i]) * Math.log(1 - p + 1e-9));
        }
        const tree = buildGradTree(X, grad, hess, subIdx, this.maxDepth, this.lambda);
        model.trees.push(tree);
        model.treeLoss.push(loss / n);
        for (let i = 0; i < n; i++) {
          scores[i] += this.learningRate * predictTree(tree, X[i]);
        }
      }
    }
    this.pruneWorstTrees(X, returns);
    this.updateFeatureImportance();
    this.samplesTrained = X.length;
    this.lastTrainAt = new Date().toLocaleTimeString("es-ES");
  }

  predictProba(x: number[]): number {
    if (this.totalTreeCount() === 0) {
      this.lastProbability = 0.5;
      return 0.5;
    }
    const p1 = this.predictHorizon(1, x);
    const p5 = this.predictHorizon(5, x);
    const p15 = this.predictHorizon(15, x);
    this.lastProbability = clamp(0.5 * p1 + 0.3 * p5 + 0.2 * p15, 0.05, 0.95);
    return this.lastProbability;
  }

  snapshot(signal: "buy" | "sell" | "hold", featNames: string[]): ModelSnapshot {
    const allLoss = this.horizons.flatMap((h) => h.trainLoss);
    return {
      probability: this.lastProbability,
      signal,
      confidence: 0.5,
      sampleCount: this.samplesTrained,
      trainProgress: clamp(this.totalTreeCount() / (this.maxTrees * 3), 0, 1),
      lastTrainAt: this.lastTrainAt || "—",
      extras: {
        trees: this.totalTreeCount(),
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
      lossHistory: [...allLoss].slice(-50),
    };
  }

  get treeCount(): number {
    return this.totalTreeCount();
  }

  toJSON(): unknown {
    return {
      maxTrees: this.maxTrees,
      maxDepth: this.maxDepth,
      lambda: this.lambda,
      learningRate: this.learningRate,
      baseLearningRate: this.baseLearningRate,
      subsample: this.subsample,
      aggressiveness: this.aggressiveness,
      lastProbability: this.lastProbability,
      buyThreshold: this.buyThreshold,
      sellThreshold: this.sellThreshold,
      nFeatures: this.nFeatures,
      featureImportance: [...this.featureImportance],
      samplesTrained: this.samplesTrained,
      lastTrainAt: this.lastTrainAt,
      horizons: this.horizons.map((h) => ({
        horizon: h.horizon,
        baseScore: h.baseScore,
        trainLoss: [...h.trainLoss],
        treeLoss: [...h.treeLoss],
        trees: h.trees.map((t) => treeToJson(t)),
      })),
    };
  }

  static fromJSON(data: unknown): GradientBoostingModel {
    const d = data as Record<string, unknown>;
    const m = new GradientBoostingModel();
    if (typeof d.maxTrees === "number") m.maxTrees = d.maxTrees;
    if (typeof d.maxDepth === "number") m.maxDepth = d.maxDepth;
    if (typeof d.lambda === "number") m.lambda = d.lambda;
    if (typeof d.learningRate === "number") m.learningRate = d.learningRate;
    if (typeof d.baseLearningRate === "number") m.baseLearningRate = d.baseLearningRate;
    if (typeof d.subsample === "number") m.subsample = d.subsample;
    if (typeof d.aggressiveness === "number") m.aggressiveness = d.aggressiveness;
    if (typeof d.lastProbability === "number") m.lastProbability = d.lastProbability;
    if (typeof d.buyThreshold === "number") m.buyThreshold = d.buyThreshold;
    if (typeof d.sellThreshold === "number") m.sellThreshold = d.sellThreshold;
    if (typeof d.nFeatures === "number") m.nFeatures = d.nFeatures;
    if (Array.isArray(d.featureImportance)) m.featureImportance = d.featureImportance.map((v) => Number(v));
    if (typeof d.samplesTrained === "number") m.samplesTrained = d.samplesTrained;
    if (typeof d.lastTrainAt === "string") m.lastTrainAt = d.lastTrainAt;
    if (Array.isArray(d.horizons)) {
      m.horizons = d.horizons.map((hData: unknown) => {
        const h = hData as Record<string, unknown>;
        const horizon = typeof h.horizon === "number" ? h.horizon : 1;
        const model = emptyHorizon(horizon);
        if (typeof h.baseScore === "number") model.baseScore = h.baseScore;
        if (Array.isArray(h.trainLoss)) model.trainLoss = h.trainLoss.map((v) => Number(v));
        if (Array.isArray(h.treeLoss)) model.treeLoss = h.treeLoss.map((v) => Number(v));
        if (Array.isArray(h.trees)) model.trees = h.trees.map((t) => treeFromJson(t)!).filter(Boolean) as TreeNode[];
        return model;
      });
    }
    return m;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// 3. StatisticalModel — Señales robustas: z-score, mean-reversion, momentum
// y tendencia. Umbral adaptativo con ventana 50.
// ════════════════════════════════════════════════════════════════════════════

export class StatisticalModel implements Tunable {
  private aggressiveness = 0.5;
  lastSignal: "buy" | "sell" | "hold" = "hold";
  private lastProbability = 0.5;
  private lastScore = 0;
  private lastTrainAt = "";
  private thresholds = {
    zScore: 1.5,
    zMeanReversion: 1.5,
  };
  private readonly WINDOW = 200;
  private winBuf: number[] = [];
  private winMean = 0;
  private winM2 = 0;
  private winCount = 0;
  private recentHits = 0;
  private recentTotal = 0;
  private adaptiveOffset = 0;
  private readonly ADAPTIVE_WINDOW = 50;

  setAggression(level: AggressionLevel): void {
    this.aggressiveness = aggressionToProfile(level);
    const agg = this.aggressiveness;
    this.thresholds.zScore = 1.5 - agg * 0.8;
    this.thresholds.zMeanReversion = 1.8 - agg * 0.8;
  }

  predict(
    features: number[],
    lastOutcome?: boolean
  ): { signal: "buy" | "sell" | "hold"; probability: number; composite: number } {
    const [
      ret1,
      ,
      ,
      ,
      ,
      mom5,
      mom15,
      ,
      ema21_50,
      price_ema50,
    ] = features;

    // Rolling Welford
    if (this.winCount >= this.WINDOW) {
      const old = this.winBuf[0];
      const deltaOld = old - this.winMean;
      this.winMean -= deltaOld / this.WINDOW;
      const deltaNew = old - this.winMean;
      this.winM2 -= deltaOld * deltaNew;
      this.winBuf.shift();
    } else {
      this.winCount++;
    }
    const delta = ret1 - this.winMean;
    this.winMean += delta / this.winCount;
    const delta2 = ret1 - this.winMean;
    this.winM2 += delta * delta2;
    const variance = this.winCount > 1 ? this.winM2 / (this.winCount - 1) : 1e-6;
    const sd = Math.sqrt(variance) || 1e-6;
    const zScore = (ret1 - this.winMean) / sd;
    this.winBuf.push(ret1);

    // Adaptive threshold con ventana 50
    if (lastOutcome !== undefined) {
      this.recentTotal++;
      if (lastOutcome) this.recentHits++;
      if (this.recentTotal >= this.ADAPTIVE_WINDOW) {
        const hitRate = this.recentHits / this.recentTotal;
        this.adaptiveOffset = (hitRate - 0.5) * 0.15;
        this.recentHits = 0;
        this.recentTotal = 0;
      }
    }

    // 1. Mean-reversion: señal contraria al z-score cuando |z| > umbral
    const meanRev =
      Math.abs(zScore) > this.thresholds.zMeanReversion ? -Math.sign(zScore) : 0;

    // 2. Momentum: mom5 y mom15 alineados
    const momSignal = mom5 * mom15 > 0 ? Math.sign(mom5) : 0;

    // 3. Tendencia: ema21_50 + price_ema50
    const trendSignal = Math.sign(ema21_50 + price_ema50);

    // 4. Z-score direccional: señal del propio z-score (trend-following)
    const zSignal = Math.abs(zScore) > this.thresholds.zScore ? Math.sign(zScore) : 0;

    const composite = meanRev + momSignal + trendSignal + zSignal * 0.5;
    this.lastScore = composite;

    const baseThreshold = 0.45 + (1 - this.aggressiveness) * 0.25;
    const threshold = clamp(baseThreshold - this.adaptiveOffset, 0.2, 0.7);
    let signal: "buy" | "sell" | "hold" = "hold";
    if (composite > threshold) signal = "buy";
    else if (composite < -threshold) signal = "sell";
    const prob = clamp(sigmoid(composite * 2.5), 0.05, 0.95);
    this.lastProbability = prob;
    this.lastSignal = signal;
    return { signal, probability: prob, composite };
  }

  snapshot(): ModelSnapshot {
    return {
      probability: this.lastProbability,
      signal: this.lastSignal,
      confidence: clamp(Math.abs(this.lastScore) / 3, 0, 1),
      sampleCount: this.winCount,
      trainProgress: clamp(this.winCount / 200, 0, 1),
      lastTrainAt: this.lastTrainAt || "Continuo",
      extras: {
        zScore: this.thresholds.zScore,
        zMeanReversion: this.thresholds.zMeanReversion,
        lastComposite: this.lastScore,
        aggressiveness: this.aggressiveness,
        adaptiveOffset: this.adaptiveOffset,
      },
    };
  }

  toJSON(): unknown {
    return {
      aggressiveness: this.aggressiveness,
      lastSignal: this.lastSignal,
      lastProbability: this.lastProbability,
      lastScore: this.lastScore,
      lastTrainAt: this.lastTrainAt,
      thresholds: { ...this.thresholds },
      winBuf: [...this.winBuf],
      winMean: this.winMean,
      winM2: this.winM2,
      winCount: this.winCount,
      recentHits: this.recentHits,
      recentTotal: this.recentTotal,
      adaptiveOffset: this.adaptiveOffset,
    };
  }

  static fromJSON(data: unknown): StatisticalModel {
    const d = data as Record<string, unknown>;
    const m = new StatisticalModel();
    if (typeof d.aggressiveness === "number") m.aggressiveness = d.aggressiveness;
    if (d.lastSignal === "buy" || d.lastSignal === "sell" || d.lastSignal === "hold")
      m.lastSignal = d.lastSignal;
    if (typeof d.lastProbability === "number") m.lastProbability = d.lastProbability;
    if (typeof d.lastScore === "number") m.lastScore = d.lastScore;
    if (typeof d.lastTrainAt === "string") m.lastTrainAt = d.lastTrainAt;
    if (d.thresholds && typeof d.thresholds === "object") {
      const t = d.thresholds as Record<string, number>;
      if (typeof t.zScore === "number") m.thresholds.zScore = t.zScore;
      if (typeof t.zMeanReversion === "number") m.thresholds.zMeanReversion = t.zMeanReversion;
    }
    if (Array.isArray(d.winBuf)) m.winBuf = d.winBuf.map((v) => Number(v));
    if (typeof d.winMean === "number") m.winMean = d.winMean;
    if (typeof d.winM2 === "number") m.winM2 = d.winM2;
    if (typeof d.winCount === "number") m.winCount = d.winCount;
    if (typeof d.recentHits === "number") m.recentHits = d.recentHits;
    if (typeof d.recentTotal === "number") m.recentTotal = d.recentTotal;
    if (typeof d.adaptiveOffset === "number") m.adaptiveOffset = d.adaptiveOffset;
    return m;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// 4. RandomForestModel — Regresión de retornos continuos con bagging,
// OOB early stopping y confianza real basada en varianza entre árboles.
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
  private lastX: number[] = [];
  private lastTrainAt = "";
  private oobMse = 0;
  private thresholds = { buy: 0.55, sell: 0.45 };
  featureImportance: number[] = [];

  setAggression(level: AggressionLevel): void {
    this.aggressiveness = aggressionToProfile(level);
    this.nEstimators = Math.round(40 + this.aggressiveness * 60);
    this.maxDepth = 3 + Math.round(this.aggressiveness * 3);
    this.nFeatSample = Math.max(2, Math.round(3 + this.aggressiveness * 3));
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
    let oobSq = 0;
    let oobCount = 0;
    for (let t = 0; t < this.nEstimators; t++) {
      const sampleIdx: number[] = [];
      for (let i = 0; i < n; i++) sampleIdx.push(Math.floor(Math.random() * n));
      const sampleSet = new Set(sampleIdx);
      const oobIdx = X.map((_, i) => i).filter((i) => !sampleSet.has(i));
      const tree = buildRegTree(X, y, sampleIdx, this.maxDepth, 4, this.nFeatSample);
      this.trees.push(tree);
      countFeatureUsage(tree, acc);
      if (oobIdx.length > 0) {
        for (const i of oobIdx) {
          const pred = predictTree(tree, X[i]);
          const err = pred - y[i];
          oobSq += err * err;
          oobCount++;
        }
      }
    }
    const total = acc.reduce((a, b) => a + b, 0) || 1;
    this.featureImportance = acc.map((v) => v / total);
    this.oobMse = oobCount > 0 ? oobSq / oobCount : 0;
    this.samplesTrained = X.length;
    this.lastTrainAt = new Date().toLocaleTimeString("es-ES");
  }

  private predictions(x: number[]): number[] {
    return this.trees.map((t) => predictTree(t, x));
  }

  predictProba(x: number[]): number {
    this.lastX = x;
    if (this.trees.length === 0) {
      this.lastProbability = 0.5;
      return 0.5;
    }
    const preds = this.predictions(x);
    const meanPred = preds.reduce((a, b) => a + b, 0) / preds.length;
    this.lastProbability = clamp(sigmoid(meanPred * 10), 0.05, 0.95);
    return this.lastProbability;
  }

  confidence(x: number[]): number {
    const preds = this.predictions(x);
    const mean = preds.reduce((a, b) => a + b, 0) / preds.length;
    const meanAbs = preds.reduce((a, b) => a + Math.abs(b), 0) / preds.length;
    const variance = preds.reduce((a, b) => a + (b - mean) * (b - mean), 0) / preds.length;
    const std = Math.sqrt(variance);
    return clamp(1 - std / (meanAbs + 1e-6), 0, 1);
  }

  addTrees(X: number[][], y: number[], extraTrees = 8): void {
    if (X.length < 30 || this.trees.length === 0) {
      this.train(X, y);
      return;
    }
    const n = X.length;
    let bestMse = this.oobMse;
    let noImprove = 0;
    for (let t = 0; t < extraTrees; t++) {
      if (this.trees.length >= this.nEstimators) break;
      const sampleIdx: number[] = [];
      for (let i = 0; i < n; i++) sampleIdx.push(Math.floor(Math.random() * n));
      const used = new Set(sampleIdx);
      const tree = buildRegTree(X, y, sampleIdx, this.maxDepth, 4, this.nFeatSample);
      this.trees.push(tree);
      const oobIdx = X.map((_, i) => i).filter((i) => !used.has(i));
      if (oobIdx.length > 0) {
        let sq = 0;
        for (const i of oobIdx) {
          const pred = this.predictions(X[i]).reduce((a, b) => a + b, 0) / this.trees.length;
          const err = pred - y[i];
          sq += err * err;
        }
        const mseNow = sq / oobIdx.length;
        if (mseNow < bestMse - 0.0001) {
          bestMse = mseNow;
          noImprove = 0;
        } else {
          noImprove++;
          if (noImprove >= 3) break;
        }
      }
    }
    this.oobMse = bestMse;
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
    const confidence =
      this.trees.length > 0 && this.lastX.length > 0
        ? this.confidence(this.lastX)
        : this.trees.length > 0
          ? this.confidence(Array(this.nFeatures).fill(0))
          : 0;
    return {
      probability: this.lastProbability,
      signal: this.pickSignal(this.lastProbability),
      confidence,
      sampleCount: this.samplesTrained,
      trainProgress: clamp(this.trees.length / this.nEstimators, 0, 1),
      lastTrainAt: this.lastTrainAt || "—",
      extras: {
        trees: this.trees.length,
        nEstimators: this.nEstimators,
        maxDepth: this.maxDepth,
        nFeatSample: this.nFeatSample,
        oobMse: this.oobMse,
        aggressiveness: this.aggressiveness,
      },
      featureImportance: featNames.map((name, i) => ({
        name,
        value: this.featureImportance[i] ?? 0,
      })),
    };
  }

  toJSON(): unknown {
    return {
      trees: this.trees.map((t) => treeToJson(t)),
      aggressiveness: this.aggressiveness,
      nEstimators: this.nEstimators,
      maxDepth: this.maxDepth,
      nFeatures: this.nFeatures,
      nFeatSample: this.nFeatSample,
      samplesTrained: this.samplesTrained,
      lastProbability: this.lastProbability,
      lastX: [...this.lastX],
      lastTrainAt: this.lastTrainAt,
      oobMse: this.oobMse,
      thresholds: { ...this.thresholds },
      featureImportance: [...this.featureImportance],
    };
  }

  static fromJSON(data: unknown): RandomForestModel {
    const d = data as Record<string, unknown>;
    const m = new RandomForestModel();
    if (Array.isArray(d.trees)) m.trees = d.trees.map((t) => treeFromJson(t)!).filter(Boolean) as TreeNode[];
    if (typeof d.aggressiveness === "number") m.aggressiveness = d.aggressiveness;
    if (typeof d.nEstimators === "number") m.nEstimators = d.nEstimators;
    if (typeof d.maxDepth === "number") m.maxDepth = d.maxDepth;
    if (typeof d.nFeatures === "number") m.nFeatures = d.nFeatures;
    if (typeof d.nFeatSample === "number") m.nFeatSample = d.nFeatSample;
    if (typeof d.samplesTrained === "number") m.samplesTrained = d.samplesTrained;
    if (typeof d.lastProbability === "number") m.lastProbability = d.lastProbability;
    if (Array.isArray(d.lastX)) m.lastX = d.lastX.map((v) => Number(v));
    if (typeof d.lastTrainAt === "string") m.lastTrainAt = d.lastTrainAt;
    if (typeof d.oobMse === "number") m.oobMse = d.oobMse;
    if (d.thresholds && typeof d.thresholds === "object") {
      const t = d.thresholds as Record<string, number>;
      if (typeof t.buy === "number") m.thresholds.buy = t.buy;
      if (typeof t.sell === "number") m.thresholds.sell = t.sell;
    }
    if (Array.isArray(d.featureImportance)) m.featureImportance = d.featureImportance.map((v) => Number(v));
    return m;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// 5. GRUModel — GRU real con puertas de actualización y reinicio, BPTT
// completo, LayerNorm por paso, dropout y Adam + weight decay.
// ════════════════════════════════════════════════════════════════════════════

interface GruStepCache {
  x: number[];
  hPrev: Float64Array;
  z: Float64Array;
  r: Float64Array;
  n: Float64Array;
  hRaw: Float64Array;
  hLn: Float64Array;
  hNew: Float64Array;
  aZ: Float64Array;
  aR: Float64Array;
  aN: Float64Array;
  lnMean: number;
  lnInvStd: number;
  mask: Float64Array | null;
}

export class GRUModel implements Tunable {
  private hiddenSize = 12;
  private seqLength = 20;
  private inputSize = 10;
  private dropout = 0.2;

  private Wz!: Float64Array;
  private Wr!: Float64Array;
  private Wn!: Float64Array;
  private Uz!: Float64Array;
  private Ur!: Float64Array;
  private Un!: Float64Array;
  private bz!: Float64Array;
  private br!: Float64Array;
  private bn!: Float64Array;
  private Wout!: Float64Array;
  private bout = 0;
  private lnGamma!: Float64Array;
  private lnBeta!: Float64Array;

  private aggressiveness = 0.5;
  private learningRate = 0.005;
  private samples = 0;
  private lastProbability = 0.5;
  private lastConfidence = 0;
  private lastTrainAt = "";
  private trainLoss: number[] = [];
  private initializationDone = false;
  private trainingSteps = 0;
  private weightDecay = 1e-5;
  private featureImportanceRaw: number[] = [];
  buyThreshold = 0.55;
  sellThreshold = 0.45;

  private adam: {
    mWz: Float64Array; vWz: Float64Array;
    mWr: Float64Array; vWr: Float64Array;
    mWn: Float64Array; vWn: Float64Array;
    mUz: Float64Array; vUz: Float64Array;
    mUr: Float64Array; vUr: Float64Array;
    mUn: Float64Array; vUn: Float64Array;
    mbz: Float64Array; vbz: Float64Array;
    mbr: Float64Array; vbr: Float64Array;
    mbn: Float64Array; vbn: Float64Array;
    mWout: Float64Array; vWout: Float64Array;
    mbout: number; vbout: number;
    mLnGamma: Float64Array; vLnGamma: Float64Array;
    mLnBeta: Float64Array; vLnBeta: Float64Array;
    t: number;
  } | null = null;

  private gradNormEMA = 1.0;

  setAggression(level: AggressionLevel): void {
    this.aggressiveness = aggressionToProfile(level);
    this.learningRate = 0.002 + this.aggressiveness * 0.006;
    const _h = Math.round(10 + this.aggressiveness * 8);
    const _s = Math.round(20 + this.aggressiveness * 6);
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
    const x = inputSize;

    const init2D = (rows: number, cols: number): Float64Array => {
      const arr = new Float64Array(rows * cols);
      const scale = Math.sqrt(2 / (rows + cols));
      for (let i = 0; i < arr.length; i++) arr[i] = randomGaussian() * scale;
      return arr;
    };

    this.Wz = init2D(h, x);
    this.Wr = init2D(h, x);
    this.Wn = init2D(h, x);
    this.Uz = init2D(h, h);
    this.Ur = init2D(h, h);
    this.Un = init2D(h, h);
    this.bz = new Float64Array(h);
    this.br = new Float64Array(h);
    this.bn = new Float64Array(h);
    this.Wout = new Float64Array(h);
    const scaleOut = Math.sqrt(1 / h);
    for (let i = 0; i < h; i++) this.Wout[i] = randomGaussian() * scaleOut;
    this.bout = 0;

    this.lnGamma = new Float64Array(h);
    this.lnBeta = new Float64Array(h);
    for (let i = 0; i < h; i++) this.lnGamma[i] = 1.0;

    this.adam = {
      mWz: new Float64Array(h * x), vWz: new Float64Array(h * x),
      mWr: new Float64Array(h * x), vWr: new Float64Array(h * x),
      mWn: new Float64Array(h * x), vWn: new Float64Array(h * x),
      mUz: new Float64Array(h * h), vUz: new Float64Array(h * h),
      mUr: new Float64Array(h * h), vUr: new Float64Array(h * h),
      mUn: new Float64Array(h * h), vUn: new Float64Array(h * h),
      mbz: new Float64Array(h), vbz: new Float64Array(h),
      mbr: new Float64Array(h), vbr: new Float64Array(h),
      mbn: new Float64Array(h), vbn: new Float64Array(h),
      mWout: new Float64Array(h), vWout: new Float64Array(h),
      mbout: 0, vbout: 0,
      mLnGamma: new Float64Array(h), vLnGamma: new Float64Array(h),
      mLnBeta: new Float64Array(h), vLnBeta: new Float64Array(h),
      t: 0,
    };
    this.initializationDone = true;
  }

  private toSubInput(f: number[]): number[] {
    return [
      f[0] ?? 0,
      f[1] ?? 0,
      f[2] ?? 0,
      f[3] ?? 0,
      f[4] ?? 0,
      f[5] ?? 0,
      f[6] ?? 0,
      f[7] ?? 0,
      f[8] ?? 0,
      f[9] ?? 0,
    ];
  }

  private layerNormForward(x: Float64Array): { y: Float64Array; mean: number; invStd: number } {
    const n = x.length;
    const eps = 1e-5;
    let mean = 0;
    for (let i = 0; i < n; i++) mean += x[i];
    mean /= n;
    let variance = 0;
    for (let i = 0; i < n; i++) variance += (x[i] - mean) * (x[i] - mean);
    variance /= n;
    const invStd = 1 / Math.sqrt(variance + eps);
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      y[i] = this.lnGamma[i] * (x[i] - mean) * invStd + this.lnBeta[i];
    }
    return { y, mean, invStd };
  }

  private layerNormBackward(
    dh: Float64Array,
    hRaw: Float64Array,
    invStd: number
  ): { dhRaw: Float64Array; dGamma: Float64Array; dBeta: Float64Array } {
    const n = dh.length;
    const eps = 1e-5;
    const variance = 1 / (invStd * invStd) - eps;
    let mean = 0;
    for (let i = 0; i < n; i++) mean += hRaw[i];
    mean /= n;
    const dxHat = new Float64Array(n);
    const dGamma = new Float64Array(n);
    const dBeta = new Float64Array(n);
    let dVar = 0;
    let dMean = 0;
    for (let i = 0; i < n; i++) {
      const xhat = (hRaw[i] - mean) * invStd;
      dxHat[i] = dh[i] * this.lnGamma[i];
      dGamma[i] = dh[i] * xhat;
      dBeta[i] = dh[i];
      dVar += dxHat[i] * (hRaw[i] - mean) * (-0.5) * Math.pow(variance + eps, -1.5);
      dMean += dxHat[i] * (-invStd);
    }
    dMean += dVar * (-2 / n) * Array.from(hRaw).reduce((s, v) => s + (v - mean), 0);
    const dhRaw = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      dhRaw[i] = dxHat[i] * invStd + dVar * 2 * (hRaw[i] - mean) / n + dMean / n;
    }
    return { dhRaw, dGamma, dBeta };
  }

  private gruStep(x: number[], hPrev: Float64Array, training: boolean): GruStepCache {
    const h = this.hiddenSize;
    const xSize = this.inputSize;
    const z = new Float64Array(h);
    const r = new Float64Array(h);
    const aZ = new Float64Array(h);
    const aR = new Float64Array(h);

    for (let j = 0; j < h; j++) {
      let az = this.bz[j];
      let ar = this.br[j];
      for (let k = 0; k < xSize; k++) {
        az += this.Wz[j * xSize + k] * x[k];
        ar += this.Wr[j * xSize + k] * x[k];
      }
      for (let k = 0; k < h; k++) {
        az += this.Uz[j * h + k] * hPrev[k];
        ar += this.Ur[j * h + k] * hPrev[k];
      }
      aZ[j] = az;
      aR[j] = ar;
      z[j] = sigmoid(az);
      r[j] = sigmoid(ar);
    }

    const n = new Float64Array(h);
    const aN = new Float64Array(h);
    for (let j = 0; j < h; j++) {
      let an = this.bn[j];
      for (let k = 0; k < xSize; k++) an += this.Wn[j * xSize + k] * x[k];
      for (let k = 0; k < h; k++) an += this.Un[j * h + k] * (r[k] * hPrev[k]);
      aN[j] = an;
      n[j] = Math.tanh(an);
    }

    const hRaw = new Float64Array(h);
    for (let j = 0; j < h; j++) {
      hRaw[j] = (1 - z[j]) * n[j] + z[j] * hPrev[j];
    }

    const { y: hLn, mean, invStd } = this.layerNormForward(hRaw);

    let mask: Float64Array | null = null;
    let hNew = hLn;
    if (training) {
      mask = new Float64Array(h);
      hNew = new Float64Array(h);
      for (let j = 0; j < h; j++) {
        mask[j] = Math.random() > this.dropout ? 1 / (1 - this.dropout) : 0;
        hNew[j] = hLn[j] * mask[j];
      }
    }

    return {
      x,
      hPrev,
      z,
      r,
      n,
      hRaw,
      hLn,
      hNew,
      aZ,
      aR,
      aN,
      lnMean: mean,
      lnInvStd: invStd,
      mask,
    };
  }

  private forward(seq: number[][], training = false): { output: number; cache: GruStepCache[] } {
    const h = this.hiddenSize;
    let hPrev: Float64Array = new Float64Array(h);
    const cache: GruStepCache[] = [];
    for (let t = 0; t < seq.length; t++) {
      const step = this.gruStep(seq[t], hPrev, training);
      cache.push(step);
      hPrev = step.hNew;
    }
    const last = cache[cache.length - 1];
    let logit = this.bout;
    for (let j = 0; j < h; j++) logit += this.Wout[j] * last.hNew[j];
    return { output: sigmoid(logit), cache };
  }

  private bptt(_seq: number[][], target: number, cache: GruStepCache[]): number {
    const h = this.hiddenSize;
    const xSize = this.inputSize;
    const T = cache.length;
    const a = this.adam!;

    const last = cache[T - 1];
    let logit = this.bout;
    for (let j = 0; j < h; j++) logit += this.Wout[j] * last.hNew[j];
    const outProb = sigmoid(logit);
    const dLogit = outProb - target;

    const gradWout = new Float64Array(h);
    for (let j = 0; j < h; j++) gradWout[j] = dLogit * last.hNew[j];
    let gradBout = dLogit;

    const dhNorm = new Float64Array(h);
    for (let j = 0; j < h; j++) dhNorm[j] = dLogit * this.Wout[j];

    const { dhRaw: dhAfterLN, dGamma: lastDGamma, dBeta: lastDBeta } =
      this.layerNormBackward(dhNorm, last.hRaw, last.lnInvStd);

    let dhNext = new Float64Array(h);
    for (let j = 0; j < h; j++) dhNext[j] = dhAfterLN[j];

    const gradWz = new Float64Array(h * xSize);
    const gradWr = new Float64Array(h * xSize);
    const gradWn = new Float64Array(h * xSize);
    const gradUz = new Float64Array(h * h);
    const gradUr = new Float64Array(h * h);
    const gradUn = new Float64Array(h * h);
    const gradBz = new Float64Array(h);
    const gradBr = new Float64Array(h);
    const gradBn = new Float64Array(h);
    const gradLnGamma = new Float64Array(h);
    const gradLnBeta = new Float64Array(h);

    for (let j = 0; j < h; j++) {
      gradLnGamma[j] += lastDGamma[j];
      gradLnBeta[j] += lastDBeta[j];
    }

    for (let t = T - 1; t >= 0; t--) {
      const c = cache[t];

      if (c.mask) {
        for (let j = 0; j < h; j++) dhNext[j] *= c.mask[j];
      }

      const { dhRaw: dhFromLn, dGamma: dg, dBeta: db } =
        this.layerNormBackward(dhNext, c.hRaw, c.lnInvStd);
      for (let j = 0; j < h; j++) {
        gradLnGamma[j] += dg[j];
        gradLnBeta[j] += db[j];
      }

      const dh = dhFromLn;
      const dz = new Float64Array(h);
      const dn = new Float64Array(h);
      for (let j = 0; j < h; j++) {
        dz[j] = dh[j] * (c.hPrev[j] - c.n[j]);
        dn[j] = dh[j] * (1 - c.z[j]);
      }

      const daZ = new Float64Array(h);
      const daR = new Float64Array(h);
      const daN = new Float64Array(h);
      for (let j = 0; j < h; j++) {
        daZ[j] = dz[j] * c.z[j] * (1 - c.z[j]);
        daN[j] = dn[j] * (1 - c.n[j] * c.n[j]);
      }

      for (let j = 0; j < h; j++) {
        gradBn[j] += daN[j];
        for (let k = 0; k < xSize; k++) gradWn[j * xSize + k] += daN[j] * c.x[k];
        for (let k = 0; k < h; k++) gradUn[j * h + k] += daN[j] * c.r[k] * c.hPrev[k];
      }

      const dr = new Float64Array(h);
      for (let k = 0; k < h; k++) {
        let s = 0;
        for (let j = 0; j < h; j++) s += daN[j] * this.Un[j * h + k];
        dr[k] = s * c.hPrev[k];
      }
      for (let j = 0; j < h; j++) {
        daR[j] = dr[j] * c.r[j] * (1 - c.r[j]);
      }

      for (let j = 0; j < h; j++) {
        gradBz[j] += daZ[j];
        for (let k = 0; k < xSize; k++) gradWz[j * xSize + k] += daZ[j] * c.x[k];
        for (let k = 0; k < h; k++) gradUz[j * h + k] += daZ[j] * c.hPrev[k];
      }

      for (let j = 0; j < h; j++) {
        gradBr[j] += daR[j];
        for (let k = 0; k < xSize; k++) gradWr[j * xSize + k] += daR[j] * c.x[k];
        for (let k = 0; k < h; k++) gradUr[j * h + k] += daR[j] * c.hPrev[k];
      }

      const dhPrev = new Float64Array(h);
      for (let j = 0; j < h; j++) dhPrev[j] = dh[j] * c.z[j];
      for (let k = 0; k < h; k++) {
        let s = 0;
        for (let j = 0; j < h; j++) s += daZ[j] * this.Uz[j * h + k];
        dhPrev[k] += s;
      }
      for (let k = 0; k < h; k++) {
        let s = 0;
        for (let j = 0; j < h; j++) s += daR[j] * this.Ur[j * h + k];
        dhPrev[k] += s;
      }
      for (let k = 0; k < h; k++) {
        let s = 0;
        for (let j = 0; j < h; j++) s += daN[j] * this.Un[j * h + k];
        dhPrev[k] += s * c.r[k];
      }

      dhNext = dhPrev;
    }

    let gNorm = 0;
    const accum = [
      gradWz, gradWr, gradWn, gradUz, gradUr, gradUn, gradBz, gradBr, gradBn,
      gradWout, gradLnGamma, gradLnBeta,
    ];
    for (const g of accum) {
      for (let i = 0; i < g.length; i++) gNorm += g[i] * g[i];
    }
    gNorm += gradBout * gradBout;
    gNorm = Math.sqrt(gNorm);
    this.gradNormEMA = 0.95 * this.gradNormEMA + 0.05 * gNorm;
    const clipThresh = Math.max(1.0, 2.0 * this.gradNormEMA);
    const gScale = gNorm > clipThresh ? clipThresh / gNorm : 1.0;
    if (gScale < 1.0) {
      for (const g of accum) {
        for (let i = 0; i < g.length; i++) g[i] *= gScale;
      }
      gradBout *= gScale;
    }

    a.t++;
    const lr = this.learningRate;
    const b1 = 0.9;
    const b2 = 0.999;
    const eps = 1e-8;
    const bc1 = 1 - Math.pow(b1, a.t);
    const bc2 = 1 - Math.pow(b2, a.t);

    const adamUpdate = (
      grad: Float64Array,
      m: Float64Array,
      v: Float64Array,
      param: Float64Array,
      wd = 0.0
    ) => {
      for (let i = 0; i < param.length; i++) {
        const g = grad[i] + wd * param[i];
        m[i] = b1 * m[i] + (1 - b1) * g;
        v[i] = b2 * v[i] + (1 - b2) * g * g;
        param[i] -= lr * (m[i] / bc1) / (Math.sqrt(v[i] / bc2) + eps);
      }
    };

    adamUpdate(gradWz, a.mWz, a.vWz, this.Wz, this.weightDecay);
    adamUpdate(gradWr, a.mWr, a.vWr, this.Wr, this.weightDecay);
    adamUpdate(gradWn, a.mWn, a.vWn, this.Wn, this.weightDecay);
    adamUpdate(gradUz, a.mUz, a.vUz, this.Uz, this.weightDecay);
    adamUpdate(gradUr, a.mUr, a.vUr, this.Ur, this.weightDecay);
    adamUpdate(gradUn, a.mUn, a.vUn, this.Un, this.weightDecay);
    adamUpdate(gradBz, a.mbz, a.vbz, this.bz, 0);
    adamUpdate(gradBr, a.mbr, a.vbr, this.br, 0);
    adamUpdate(gradBn, a.mbn, a.vbn, this.bn, 0);
    adamUpdate(gradWout, a.mWout, a.vWout, this.Wout, this.weightDecay);
    adamUpdate(gradLnGamma, a.mLnGamma, a.vLnGamma, this.lnGamma, 0);
    adamUpdate(gradLnBeta, a.mLnBeta, a.vLnBeta, this.lnBeta, 0);

    a.mbout = b1 * a.mbout + (1 - b1) * gradBout;
    a.vbout = b2 * a.vbout + (1 - b2) * gradBout * gradBout;
    this.bout -= lr * (a.mbout / bc1) / (Math.sqrt(a.vbout / bc2) + eps);

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
      const fwd = this.forward(seq, true);
      totalLoss += this.bptt(seq, target, fwd.cache);
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
    const fwd = this.forward(seq, false);
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
    if (this.initializationDone) {
      const h = this.hiddenSize;
      const xSize = this.inputSize;
      const imp = new Array(xSize).fill(0);
      for (let j = 0; j < h; j++) {
        for (let k = 0; k < xSize; k++) {
          imp[k] +=
            Math.abs(this.Wz[j * xSize + k]) +
            Math.abs(this.Wr[j * xSize + k]) +
            Math.abs(this.Wn[j * xSize + k]);
        }
      }
      const total = imp.reduce((a, b) => a + b, 0) || 1;
      this.featureImportanceRaw = imp.map((v) => v / total);
    }
    const featNames = _featNames ?? [
      "Retorno 1", "Retorno 3", "RSI-14", "MACD hist", "Volatilidad 20",
      "Momentum 5", "Momentum 15", "EMA9/EMA21", "EMA21/EMA50", "Precio/EMA50",
    ];
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
        gradNormEMA: this.gradNormEMA,
        adamT: this.adam?.t ?? 0,
      },
      featureImportance: featNames.slice(0, this.featureImportanceRaw.length).map((name, i) => ({
        name,
        value: this.featureImportanceRaw[i] ?? 0,
      })),
      lossHistory: [...this.trainLoss].slice(-50),
    };
  }

  toJSON(): unknown {
    return {
      hiddenSize: this.hiddenSize,
      seqLength: this.seqLength,
      inputSize: this.inputSize,
      dropout: this.dropout,
      Wz: Array.from(this.Wz ?? []),
      Wr: Array.from(this.Wr ?? []),
      Wn: Array.from(this.Wn ?? []),
      Uz: Array.from(this.Uz ?? []),
      Ur: Array.from(this.Ur ?? []),
      Un: Array.from(this.Un ?? []),
      bz: Array.from(this.bz ?? []),
      br: Array.from(this.br ?? []),
      bn: Array.from(this.bn ?? []),
      Wout: Array.from(this.Wout ?? []),
      bout: this.bout,
      lnGamma: Array.from(this.lnGamma ?? []),
      lnBeta: Array.from(this.lnBeta ?? []),
      aggressiveness: this.aggressiveness,
      learningRate: this.learningRate,
      samples: this.samples,
      lastProbability: this.lastProbability,
      lastConfidence: this.lastConfidence,
      lastTrainAt: this.lastTrainAt,
      trainLoss: [...this.trainLoss],
      initializationDone: this.initializationDone,
      trainingSteps: this.trainingSteps,
      weightDecay: this.weightDecay,
      featureImportanceRaw: [...this.featureImportanceRaw],
      buyThreshold: this.buyThreshold,
      sellThreshold: this.sellThreshold,
      gradNormEMA: this.gradNormEMA,
      adam: this.adam
        ? {
            mWz: Array.from(this.adam.mWz),
            vWz: Array.from(this.adam.vWz),
            mWr: Array.from(this.adam.mWr),
            vWr: Array.from(this.adam.vWr),
            mWn: Array.from(this.adam.mWn),
            vWn: Array.from(this.adam.vWn),
            mUz: Array.from(this.adam.mUz),
            vUz: Array.from(this.adam.vUz),
            mUr: Array.from(this.adam.mUr),
            vUr: Array.from(this.adam.vUr),
            mUn: Array.from(this.adam.mUn),
            vUn: Array.from(this.adam.vUn),
            mbz: Array.from(this.adam.mbz),
            vbz: Array.from(this.adam.vbz),
            mbr: Array.from(this.adam.mbr),
            vbr: Array.from(this.adam.vbr),
            mbn: Array.from(this.adam.mbn),
            vbn: Array.from(this.adam.vbn),
            mWout: Array.from(this.adam.mWout),
            vWout: Array.from(this.adam.vWout),
            mbout: this.adam.mbout,
            vbout: this.adam.vbout,
            mLnGamma: Array.from(this.adam.mLnGamma),
            vLnGamma: Array.from(this.adam.vLnGamma),
            mLnBeta: Array.from(this.adam.mLnBeta),
            vLnBeta: Array.from(this.adam.vLnBeta),
            t: this.adam.t,
          }
        : null,
    };
  }

  static fromJSON(data: unknown): GRUModel {
    const d = data as Record<string, unknown>;
    const m = new GRUModel();
    if (typeof d.hiddenSize === "number") m.hiddenSize = d.hiddenSize;
    if (typeof d.seqLength === "number") m.seqLength = d.seqLength;
    if (typeof d.inputSize === "number") m.inputSize = d.inputSize;
    if (typeof d.dropout === "number") m.dropout = d.dropout;
    if (typeof d.aggressiveness === "number") m.aggressiveness = d.aggressiveness;
    if (typeof d.learningRate === "number") m.learningRate = d.learningRate;
    if (typeof d.samples === "number") m.samples = d.samples;
    if (typeof d.lastProbability === "number") m.lastProbability = d.lastProbability;
    if (typeof d.lastConfidence === "number") m.lastConfidence = d.lastConfidence;
    if (typeof d.lastTrainAt === "string") m.lastTrainAt = d.lastTrainAt;
    if (Array.isArray(d.trainLoss)) m.trainLoss = d.trainLoss.map((v) => Number(v));
    if (typeof d.initializationDone === "boolean") m.initializationDone = d.initializationDone;
    if (typeof d.trainingSteps === "number") m.trainingSteps = d.trainingSteps;
    if (typeof d.weightDecay === "number") m.weightDecay = d.weightDecay;
    if (Array.isArray(d.featureImportanceRaw)) m.featureImportanceRaw = d.featureImportanceRaw.map((v) => Number(v));
    if (typeof d.buyThreshold === "number") m.buyThreshold = d.buyThreshold;
    if (typeof d.sellThreshold === "number") m.sellThreshold = d.sellThreshold;
    if (typeof d.gradNormEMA === "number") m.gradNormEMA = d.gradNormEMA;

    const toArr = (v: unknown): Float64Array | undefined =>
      Array.isArray(v) ? new Float64Array(v.map((n) => Number(n))) : undefined;

    if (m.initializationDone) {
      const h = m.hiddenSize;
      const x = m.inputSize;
      m.Wz = toArr(d.Wz) ?? new Float64Array(h * x);
      m.Wr = toArr(d.Wr) ?? new Float64Array(h * x);
      m.Wn = toArr(d.Wn) ?? new Float64Array(h * x);
      m.Uz = toArr(d.Uz) ?? new Float64Array(h * h);
      m.Ur = toArr(d.Ur) ?? new Float64Array(h * h);
      m.Un = toArr(d.Un) ?? new Float64Array(h * h);
      m.bz = toArr(d.bz) ?? new Float64Array(h);
      m.br = toArr(d.br) ?? new Float64Array(h);
      m.bn = toArr(d.bn) ?? new Float64Array(h);
      m.Wout = toArr(d.Wout) ?? new Float64Array(h);
      m.lnGamma = toArr(d.lnGamma) ?? new Float64Array(h);
      m.lnBeta = toArr(d.lnBeta) ?? new Float64Array(h);
      for (let i = 0; i < h; i++) if (m.lnGamma[i] === 0) m.lnGamma[i] = 1;
    }
    if (typeof d.bout === "number") m.bout = d.bout;

    if (d.adam && typeof d.adam === "object" && m.initializationDone) {
      const a = d.adam as Record<string, unknown>;
      const h = m.hiddenSize;
      const x = m.inputSize;
      m.adam = {
        mWz: toArr(a.mWz) ?? new Float64Array(h * x),
        vWz: toArr(a.vWz) ?? new Float64Array(h * x),
        mWr: toArr(a.mWr) ?? new Float64Array(h * x),
        vWr: toArr(a.vWr) ?? new Float64Array(h * x),
        mWn: toArr(a.mWn) ?? new Float64Array(h * x),
        vWn: toArr(a.vWn) ?? new Float64Array(h * x),
        mUz: toArr(a.mUz) ?? new Float64Array(h * h),
        vUz: toArr(a.vUz) ?? new Float64Array(h * h),
        mUr: toArr(a.mUr) ?? new Float64Array(h * h),
        vUr: toArr(a.vUr) ?? new Float64Array(h * h),
        mUn: toArr(a.mUn) ?? new Float64Array(h * h),
        vUn: toArr(a.vUn) ?? new Float64Array(h * h),
        mbz: toArr(a.mbz) ?? new Float64Array(h),
        vbz: toArr(a.vbz) ?? new Float64Array(h),
        mbr: toArr(a.mbr) ?? new Float64Array(h),
        vbr: toArr(a.vbr) ?? new Float64Array(h),
        mbn: toArr(a.mbn) ?? new Float64Array(h),
        vbn: toArr(a.vbn) ?? new Float64Array(h),
        mWout: toArr(a.mWout) ?? new Float64Array(h),
        vWout: toArr(a.vWout) ?? new Float64Array(h),
        mbout: typeof a.mbout === "number" ? a.mbout : 0,
        vbout: typeof a.vbout === "number" ? a.vbout : 0,
        mLnGamma: toArr(a.mLnGamma) ?? new Float64Array(h),
        vLnGamma: toArr(a.vLnGamma) ?? new Float64Array(h),
        mLnBeta: toArr(a.mLnBeta) ?? new Float64Array(h),
        vLnBeta: toArr(a.vLnBeta) ?? new Float64Array(h),
        t: typeof a.t === "number" ? a.t : 0,
      };
    }
    return m;
  }
}
