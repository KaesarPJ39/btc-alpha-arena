// ── Risk management: stops, sizing, drawdown, correlation ────────────────────

import { clamp } from "./indicators";

export interface RiskProfile {
  /** Fraction of equity to risk on a single trade (e.g. 0.01 = 1%) */
  riskPerTrade: number;
  /** ATR multiplier for stop-loss */
  stopLossAtr: number;
  /** ATR multiplier for take-profit */
  takeProfitAtr: number;
  /** ATR multiplier for trailing stop */
  trailingStopAtr: number;
  /** Hard max drawdown before pausing a model (e.g. 0.15 = 15%) */
  maxDrawdown: number;
  /** Pause duration in ticks after max drawdown is hit */
  pauseTicks: number;
  /** Reduce size when N or more models agree on the same direction */
  concentrationThreshold: number;
  /** Size multiplier when concentration is detected */
  concentrationFactor: number;
  /** Slippage simulation range for backtests/live paper trading (fraction) */
  slippage: number;
}

export const DEFAULT_RISK_PROFILE: RiskProfile = {
  riskPerTrade: 0.01,
  stopLossAtr: 1.5,
  takeProfitAtr: 2.5,
  trailingStopAtr: 2.0,
  maxDrawdown: 0.15,
  pauseTicks: 100,
  concentrationThreshold: 4,
  concentrationFactor: 0.5,
  slippage: 0.0003,
};

export interface Position {
  side: "long" | "short";
  entryPrice: number;
  qty: number;
  stopLoss: number;
  takeProfit: number;
  trailingStop: number;
}

export class RiskManager {
  profile: RiskProfile;

  constructor(profile: Partial<RiskProfile> = {}) {
    this.profile = { ...DEFAULT_RISK_PROFILE, ...profile };
  }

  /** Average True Range from close-only data (simplified, no high/low) */
  static atr(closes: number[], period = 14): number {
    if (closes.length < 2) return 0;
    const tr: number[] = [];
    for (let i = 1; i < closes.length; i++) {
      tr.push(Math.abs(closes[i] - closes[i - 1]));
    }
    if (tr.length < period) {
      return tr.reduce((a, b) => a + b, 0) / Math.max(1, tr.length);
    }
    let sum = 0;
    for (let i = tr.length - period; i < tr.length; i++) sum += tr[i];
    return sum / period;
  }

  /** Volatility-adjusted position size in USD (notional) */
  positionSize(
    equity: number,
    price: number,
    atr: number,
    confidence: number,
    modelVolatility: number
  ): number {
    if (atr <= 0 || price <= 0) return 0;
    const riskAmount = equity * this.profile.riskPerTrade;
    const stopDistance = atr * this.profile.stopLossAtr;
    const notional = riskAmount / (stopDistance / price);

    // Scale by model confidence
    const confidenceAdj = clamp(confidence, 0.2, 1);
    // Reduce size in high volatility regimes
    const volAdj = clamp(1 - modelVolatility * 1.5, 0.3, 1);

    return notional * confidenceAdj * volAdj;
  }

  stopLoss(price: number, atr: number, direction: "long" | "short"): number {
    const dist = atr * this.profile.stopLossAtr;
    return direction === "long" ? price - dist : price + dist;
  }

  takeProfit(price: number, atr: number, direction: "long" | "short"): number {
    const dist = atr * this.profile.takeProfitAtr;
    return direction === "long" ? price + dist : price - dist;
  }

  /** Returns new trailing stop level or null if unchanged */
  updateTrailingStop(
    position: Position,
    currentPrice: number,
    atr: number
  ): number | null {
    if (position.side === "long") {
      const newStop = currentPrice - atr * this.profile.trailingStopAtr;
      if (newStop > position.trailingStop) return newStop;
    } else {
      const newStop = currentPrice + atr * this.profile.trailingStopAtr;
      if (newStop < position.trailingStop) return newStop;
    }
    return null;
  }

  /** True if the position should be closed by stop/take-profit */
  shouldClose(position: Position, price: number): { close: boolean; reason: string } | null {
    if (position.side === "long") {
      if (price <= position.stopLoss) return { close: true, reason: "stop-loss" };
      if (price >= position.takeProfit) return { close: true, reason: "take-profit" };
      if (price <= position.trailingStop) return { close: true, reason: "trailing-stop" };
    } else {
      if (price >= position.stopLoss) return { close: true, reason: "stop-loss" };
      if (price <= position.takeProfit) return { close: true, reason: "take-profit" };
      if (price >= position.trailingStop) return { close: true, reason: "trailing-stop" };
    }
    return null;
  }

  /** Kelly fraction (fractional) capped to avoid overbetting */
  static kellyFraction(winRate: number, avgWin: number, avgLoss: number): number {
    if (avgLoss <= 0 || winRate <= 0 || winRate >= 1) return 0;
    const b = avgWin / avgLoss; // payoff odds
    const kelly = (winRate * b - (1 - winRate)) / b;
    return clamp(kelly * 0.25, 0, 0.5); // quarter-Kelly, max 50%
  }

  /** Returns a concentration multiplier based on number of models aligned */
  concentrationAdjustment(alignedCount: number): number {
    if (alignedCount >= this.profile.concentrationThreshold) {
      return this.profile.concentrationFactor;
    }
    return 1;
  }

  /** Apply random slippage to a price for realistic simulation */
  applySlippage(price: number, side: "buy" | "sell"): number {
    const slip = (Math.random() - 0.5) * 2 * this.profile.slippage;
    return side === "buy" ? price * (1 + slip) : price * (1 - slip);
  }
}
