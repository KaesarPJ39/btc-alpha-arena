// ── Tipos y configuración central del BTC Alpha Arena ────────────────────────

export type ModelId = "rl" | "xgb" | "stat" | "rf" | "lstm";

export type AggressionLevel = 0 | 1 | 2 | 3 | 4;

export const AGGRESSION_LABELS = [
  "Conservador",
  "Moderado",
  "Neutral",
  "Agresivo",
  "Temerario",
] as const;

export const AGGRESSION_SHORT = ["Cons.", "Mod.", "Neu.", "Agre.", "Temer."] as const;

export interface ModelMeta {
  id: ModelId;
  name: string;
  shortName: string;
  category: string;
  description: string;
  fullName: string;
  colorHex: string;
  tailwindText: string;
  tailwindBadge: string;
  tailwindBar: string;
  tailwindRing: string;
}

export const MODELS: Record<ModelId, ModelMeta> = {
  rl: {
    id: "rl",
    name: "Agente RL",
    shortName: "RL",
    category: "Aprendizaje por refuerzo",
    description: "Q-Learning tabular online con ε-greedy",
    fullName: "Q-Learning · aprendizaje por refuerzo online",
    colorHex: "#10b981",
    tailwindText: "text-emerald-400",
    tailwindBadge: "bg-emerald-500/10 text-emerald-300 border-emerald-500/30",
    tailwindBar: "[&>div]:bg-emerald-500",
    tailwindRing: "ring-emerald-500/30",
  },
  xgb: {
    id: "xgb",
    name: "XGBoost",
    shortName: "XGB",
    category: "Gradient Boosting",
    description: "Gradient Boosting con árboles de regresión",
    fullName: "Gradient Boosting · 40 árboles · log-loss",
    colorHex: "#8b5cf6",
    tailwindText: "text-violet-400",
    tailwindBadge: "bg-violet-500/10 text-violet-300 border-violet-500/30",
    tailwindBar: "[&>div]:bg-violet-500",
    tailwindRing: "ring-violet-500/30",
  },
  stat: {
    id: "stat",
    name: "Statistical",
    shortName: "STAT",
    category: "Modelo estadístico",
    description: "Pruebas y métricas estadísticas para decisión",
    fullName: "Statistical Model · pruebas de hipótesis + Z-score compuesto",
    colorHex: "#06b6d4",
    tailwindText: "text-cyan-400",
    tailwindBadge: "bg-cyan-500/10 text-cyan-300 border-cyan-500/30",
    tailwindBar: "[&>div]:bg-cyan-500",
    tailwindRing: "ring-cyan-500/30",
  },
  rf: {
    id: "rf",
    name: "Random Forest",
    shortName: "RF",
    category: "Bagging de árboles",
    description: "Bosque aleatorio con bootstrap sampling",
    fullName: "Random Forest · 60 árboles con bagging",
    colorHex: "#f59e0b",
    tailwindText: "text-amber-400",
    tailwindBadge: "bg-amber-500/10 text-amber-300 border-amber-500/30",
    tailwindBar: "[&>div]:bg-amber-500",
    tailwindRing: "ring-amber-500/30",
  },
  lstm: {
    id: "lstm",
    name: "LSTM",
    shortName: "LSTM",
    category: "Red neuronal recurrente",
    description: "Red LSTM con puertas de memoria",
    fullName: "LSTM · red neuronal recurrente para series temporales",
    colorHex: "#ec4899",
    tailwindText: "text-pink-400",
    tailwindBadge: "bg-pink-500/10 text-pink-300 border-pink-500/30",
    tailwindBar: "[&>div]:bg-pink-500",
    tailwindRing: "ring-pink-500/30",
  },
};

export const MODEL_IDS: ModelId[] = ["rl", "xgb", "stat", "rf", "lstm"];

/**
 * Mapea el nivel de agresividad (0..4) → perfil continuo [0..1].
 * Conservador 0 → 0.15, Temerario 4 → 0.95.
 */
export function aggressionToProfile(level: AggressionLevel): number {
  return 0.15 + (level / 4) * 0.8;
}
