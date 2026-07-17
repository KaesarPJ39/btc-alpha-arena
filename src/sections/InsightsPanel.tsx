import { useMemo } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EngineSnapshot } from "@/lib/engine";

interface Props {
  snap: EngineSnapshot;
}

export function InsightsPanel({ snap }: Props) {
  const rlInfo = snap.modelInfo.rl;
  const rewards = useMemo(
    () => (rlInfo.lossHistory ?? []).map((v, i) => ({ i, v: -v })),
    [rlInfo.lossHistory]
  );
  const xgbFi = useMemo(
    () => [...(snap.modelInfo.xgb.featureImportance ?? [])].sort((a, b) => b.value - a.value),
    [snap.modelInfo.xgb.featureImportance]
  );
  const rfFi = useMemo(
    () => [...(snap.modelInfo.rf.featureImportance ?? [])].sort((a, b) => b.value - a.value),
    [snap.modelInfo.rf.featureImportance]
  );
  const gruFi = useMemo(
    () => [...(snap.modelInfo.gru.featureImportance ?? [])].sort((a, b) => b.value - a.value),
    [snap.modelInfo.gru.featureImportance]
  );
  const statInfo = snap.modelInfo.stat;

  return (
    <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
      {/* XGBoost FI */}
      <FeatureImportanceCard
        title="XGBoost · importancia de variables"
        hint="Frecuencia de split por feature en el ensemble"
        colorHex="#8b5cf6"
        accent="#8b5cf6"
        fallback="#4c3a7a"
        data={xgbFi}
        stats={[
          { label: "Árboles", value: `${snap.modelInfo.xgb.extras.trees ?? 0}` },
          { label: "Muestras", value: `${snap.modelInfo.xgb.sampleCount}` },
          { label: "Reentrenado", value: snap.modelInfo.xgb.lastTrainAt },
        ]}
        footer={
          <div className="rounded-xl border border-violet-500/20 bg-violet-500/5 px-3 py-2 text-[11px]">
            <span className="text-muted-foreground">P(subida) próxima barra: </span>
            <span className={`font-bold tabular-nums ${snap.modelInfo.xgb.probability >= 0.5 ? "text-emerald-400" : "text-rose-400"}`}>
              {(snap.modelInfo.xgb.probability * 100).toFixed(1)}%
            </span>
            <ProbBar p={snap.modelInfo.xgb.probability} toneHex="#8b5cf6" />
          </div>
        }
      />

      {/* RL rewards */}
      <div className="glass-card rounded-2xl p-5">
        <div className="mb-1 flex items-center gap-2">
          <span className="h-2 w-2 rounded-full" style={{ background: "#10b981" }} />
          <h3 className="text-sm font-bold tracking-tight">Agente RL · recompensa acumulada</h3>
        </div>
        <p className="mb-4 text-[11px] text-muted-foreground">
          Curva de aprendizaje Q-Learning (recompensa = Δ equity por paso, moldeada por Sharpe)
        </p>
        <div className="h-[220px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={rewards} margin={{ top: 4, right: 8, bottom: 0, left: 4 }}>
              <defs>
                <linearGradient id="gRew" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#10b981" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(220 22% 14%)" vertical={false} />
              <XAxis dataKey="i" hide />
              <YAxis stroke="hsl(215 16% 45%)" fontSize={10} tickLine={false} axisLine={false} width={48} />
              <Tooltip
                formatter={(v: number) => [v.toFixed(1), "Recompensa Σ"]}
                labelFormatter={() => ""}
                contentStyle={{ background: "hsl(222 40% 8%)", border: "1px solid hsl(220 22% 16%)", borderRadius: 12, fontSize: 11 }}
              />
              <Area type="monotone" dataKey="v" stroke="#10b981" strokeWidth={1.8} fill="url(#gRew)" dot={false} isAnimationActive={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-4 grid grid-cols-3 gap-3 border-t border-border/60 pt-4 text-[11px]">
          <SimpleStat label="Estados Q" value={`${rlInfo.extras.qStates ?? 0}`} />
          <SimpleStat label="Actualizaciones" value={`${rlInfo.sampleCount}`} />
          <SimpleStat label="ε exploración" value={(rlInfo.extras.epsilon ?? 0).toFixed(3)} />
        </div>
        <div className="mt-3 rounded-xl border border-emerald-500/20 bg-emerald-500/5 px-3 py-2 text-[11px]">
          <span className="text-muted-foreground">Última acción: </span>
          <span className="font-bold text-emerald-300">{rlInfo.signal.toUpperCase()}</span>
          <span className="ml-3 text-muted-foreground">Confianza: </span>
          <span className="font-bold tabular-nums text-emerald-300">{Math.round(rlInfo.confidence * 100)}%</span>
        </div>
      </div>

      {/* Statistical: pruebas */}
      <div className="glass-card rounded-2xl p-5">
        <div className="mb-1 flex items-center gap-2">
          <span className="h-2 w-2 rounded-full" style={{ background: "#06b6d4" }} />
          <h3 className="text-sm font-bold tracking-tight">Statistical · pruebas de hipótesis</h3>
        </div>
        <p className="mb-4 text-[11px] text-muted-foreground">
          Umbrales por prueba (Z-momentum, Z-mean reversion, R², vol) y score compuesto
        </p>
        <div className="grid grid-cols-2 gap-3 text-[11px]">
          <Box label="Z-Momentum umbral" value={(statInfo.extras.zMomentum ?? 0).toFixed(2)} />
          <Box label="Z-MeanRev umbral" value={(statInfo.extras.zMeanReversion ?? 0).toFixed(2)} />
          <Box label="R² Tendencia umbral" value={(statInfo.extras.r2TrendMin ?? 0).toFixed(2)} />
          <Box label="Penalización vol" value={(statInfo.extras.volatilityPenalty ?? 0).toFixed(2)} />
          <Box label="Último score compuesto" value={(statInfo.extras.lastComposite ?? 0).toFixed(2)} accent="#06b6d4" />
          <Box label="Muestras vistas" value={`${statInfo.sampleCount}`} />
        </div>
        <div className="mt-3 rounded-xl border border-cyan-500/20 bg-cyan-500/5 px-3 py-2 text-[11px]">
          <span className="text-muted-foreground">Señal actual: </span>
          <span className="font-bold text-cyan-300">{statInfo.signal.toUpperCase()}</span>
          <span className="ml-3 text-muted-foreground">Confianza: </span>
          <span className="font-bold tabular-nums text-cyan-300">{Math.round(statInfo.confidence * 100)}%</span>
          <ProbBar p={statInfo.probability} toneHex="#06b6d4" />
        </div>
      </div>

      {/* Random Forest FI */}
      <FeatureImportanceCard
        title="Random Forest · importancia de variables"
        hint="Frecuencia de split promediada en el bosque · bagging con Gini"
        colorHex="#f59e0b"
        accent="#f59e0b"
        fallback="#7a5818"
        data={rfFi}
        stats={[
          { label: "Árboles", value: `${snap.modelInfo.rf.extras.trees ?? 0}` },
          { label: "Profundidad", value: `${snap.modelInfo.rf.extras.maxDepth ?? 0}` },
          { label: "OOB estimate", value: `${((snap.modelInfo.rf.extras.oobEstimate ?? 0) * 100).toFixed(1)}%` },
        ]}
        footer={
          <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 px-3 py-2 text-[11px]">
            <span className="text-muted-foreground">P(subida) por mayoría: </span>
            <span className={`font-bold tabular-nums ${snap.modelInfo.rf.probability >= 0.5 ? "text-emerald-400" : "text-rose-400"}`}>
              {(snap.modelInfo.rf.probability * 100).toFixed(1)}%
            </span>
            <ProbBar p={snap.modelInfo.rf.probability} toneHex="#f59e0b" />
          </div>
        }
      />

      {/* GRU: parámetros */}
      <div className="glass-card rounded-2xl p-5">
        <div className="mb-1 flex items-center gap-2">
          <span className="h-2 w-2 rounded-full" style={{ background: "#ec4899" }} />
          <h3 className="text-sm font-bold tracking-tight">GRU · red neuronal recurrente</h3>
        </div>
        <p className="mb-4 text-[11px] text-muted-foreground">
          Estado interno y progreso de entrenamiento (BPTT truncado, secuencia = {snap.modelInfo.gru.extras.seqLength ?? 0} pasos)
        </p>
        <div className="grid grid-cols-2 gap-3 text-[11px]">
          <Box label="Hidden size" value={`${snap.modelInfo.gru.extras.hiddenSize ?? 0}`} />
          <Box label="Seq length" value={`${snap.modelInfo.gru.extras.seqLength ?? 0}`} />
          <Box label="LR" value={(snap.modelInfo.gru.extras.learningRate ?? 0).toFixed(4)} />
          <Box label="λ (L2)" value={(snap.modelInfo.gru.extras.lambda ?? 0).toExponential(1)} />
          <Box label="Pasos de entrenamiento" value={`${snap.modelInfo.gru.extras.trainingSteps ?? 0}`} />
          <Box label="Muestras" value={`${snap.modelInfo.gru.sampleCount}`} />
        </div>
        <div className="mt-3 rounded-xl border border-pink-500/20 bg-pink-500/5 px-3 py-2 text-[11px]">
          <span className="text-muted-foreground">P(subida) por GRU: </span>
          <span className={`font-bold tabular-nums ${snap.modelInfo.gru.probability >= 0.5 ? "text-emerald-400" : "text-rose-400"}`}>
            {(snap.modelInfo.gru.probability * 100).toFixed(1)}%
          </span>
          <ProbBar p={snap.modelInfo.gru.probability} toneHex="#ec4899" />
        </div>
      </div>

      {/* GRU FI (sintética) */}
      <FeatureImportanceCard
        title="GRU · importancia estimada de inputs"
        hint="Basado en frecuencia de activación (aproximación)"
        colorHex="#ec4899"
        accent="#ec4899"
        fallback="#7a3a5a"
        data={gruFi}
        stats={[
          { label: "Hidden", value: `${snap.modelInfo.gru.extras.hiddenSize ?? 0}` },
          { label: "Pasos", value: `${snap.modelInfo.gru.extras.trainingSteps ?? 0}` },
          { label: "Reentrenado", value: snap.modelInfo.gru.lastTrainAt },
        ]}
      />
    </div>
  );
}

function FeatureImportanceCard(props: {
  title: string;
  hint: string;
  colorHex: string;
  accent: string;
  fallback: string;
  data: { name: string; value: number }[];
  stats: { label: string; value: string }[];
  footer?: React.ReactNode;
}) {
  return (
    <div className="glass-card rounded-2xl p-5">
      <div className="mb-1 flex items-center gap-2">
        <span className="h-2 w-2 rounded-full" style={{ background: props.colorHex }} />
        <h3 className="text-sm font-bold tracking-tight">{props.title}</h3>
      </div>
      <p className="mb-4 text-[11px] text-muted-foreground">{props.hint}</p>
      {props.data.length > 0 ? (
        <div className="h-[200px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={props.data} layout="vertical" margin={{ top: 0, right: 12, bottom: 0, left: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(220 22% 14%)" horizontal={false} />
              <XAxis type="number" tickFormatter={(v: number) => `${Math.round(v * 100)}%`} stroke="hsl(215 16% 45%)" fontSize={10} tickLine={false} axisLine={false} />
              <YAxis type="category" dataKey="name" width={92} stroke="hsl(215 16% 45%)" fontSize={10} tickLine={false} axisLine={false} />
              <Tooltip
                formatter={(v: number) => [`${(v * 100).toFixed(1)}%`, "Importancia"]}
                contentStyle={{ background: "hsl(222 40% 8%)", border: "1px solid hsl(220 22% 16%)", borderRadius: 12, fontSize: 11 }}
              />
              <Bar dataKey="value" radius={[0, 4, 4, 0]} isAnimationActive={false}>
                {props.data.map((_, i) => (
                  <Cell key={i} fill={i < 3 ? props.accent : props.fallback} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="flex h-[200px] items-center justify-center text-[11px] text-muted-foreground">
          Sin datos todavía
        </div>
      )}
      <div className="mt-4 grid grid-cols-3 gap-3 border-t border-border/60 pt-4 text-[11px]">
        {props.stats.map((s) => (
          <SimpleStat key={s.label} label={s.label} value={s.value} />
        ))}
      </div>
      {props.footer}
    </div>
  );
}

function Box({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="rounded-xl border border-border/40 bg-secondary/20 px-3 py-2">
      <p className="text-[10px] text-muted-foreground">{label}</p>
      <p className="mt-0.5 font-semibold tabular-nums" style={accent ? { color: accent } : undefined}>
        {value}
      </p>
    </div>
  );
}

function SimpleStat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-muted-foreground">{label}</p>
      <p className="font-semibold tabular-nums">{value}</p>
    </div>
  );
}

function ProbBar({ p, toneHex }: { p: number; toneHex: string }) {
  return (
    <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-secondary">
      <div
        className="h-full rounded-full transition-all duration-500"
        style={{ width: `${Math.round(p * 100)}%`, background: toneHex }}
      />
    </div>
  );
}

