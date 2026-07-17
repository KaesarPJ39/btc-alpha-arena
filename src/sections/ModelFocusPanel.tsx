import type { AgentMetrics, ModelUiState } from "@/lib/engine";
import { fmtUSD } from "@/lib/format";
import { MODELS, type ModelId } from "@/lib/registry";

interface Props {
  modelId: ModelId;
  info: ModelUiState;
  metrics: AgentMetrics;
}

export function ModelFocusPanel({ modelId, info, metrics }: Props) {
  const meta = MODELS[modelId];
  const extras = Object.entries(info.extras);
  const fi = info.featureImportance ?? [];

  return (
    <div className="glass-card rounded-2xl p-5">
      <div className="mb-3 flex items-center gap-2">
        <span className="h-2 w-2 rounded-full" style={{ background: meta.colorHex }} />
        <h3 className="text-sm font-bold tracking-tight">Ficha interna · {meta.name}</h3>
      </div>

      {/* Señal y probabilidad */}
      <div className="grid grid-cols-3 gap-3 border-b border-border/40 pb-4">
        <Box label="Señal" value={info.signal.toUpperCase()} accent={meta.colorHex} />
        <Box label="Confianza" value={`${Math.round(info.confidence * 100)}%`} accent={meta.colorHex} />
        <Box label="P(subida)" value={`${(info.probability * 100).toFixed(1)}%`} accent={meta.colorHex} />
      </div>

      {/* Métricas de cartera breves */}
      <div className="mt-4 grid grid-cols-3 gap-3 border-b border-border/40 pb-4 text-[11px]">
        <Box label="Cash" value={fmtUSD(metrics.cash)} />
        <Box label="BTC" value={`${metrics.btc.toFixed(5)}`} />
        <Box label="Exposición" value={`${Math.round(metrics.exposure * 100)}%`} />
        <Box label="Sharpe" value={metrics.sharpe.toFixed(2)} />
        <Box label="Max DD" value={`-${metrics.maxDD.toFixed(2)}%`} />
        <Box label="Win rate" value={metrics.realizedCloses ? `${metrics.winRate.toFixed(0)}%` : "—"} />
      </div>

      {/* Hiperparámetros internos */}
      {extras.length > 0 && (
        <div className="mt-4">
          <p className="mb-2 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
            Hiperparámetros internos
          </p>
          <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-[11px] sm:grid-cols-3">
            {extras.map(([k, v]) => (
              <div key={k} className="flex items-center justify-between gap-2 rounded-md bg-secondary/30 px-2 py-1">
                <span className="text-muted-foreground">{prettyKey(k)}</span>
                <span className="tabular-nums font-semibold" style={{ color: meta.colorHex }}>
                  {formatVal(k, v)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Feature importance si existe */}
      {fi.length > 0 && (
        <div className="mt-4 border-t border-border/40 pt-3">
          <p className="mb-2 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
            Importancia de variables
          </p>
          <div className="space-y-1.5">
            {[...fi].sort((a, b) => b.value - a.value).slice(0, 6).map((f) => (
              <div key={f.name} className="flex items-center gap-2 text-[11px]">
                <span className="w-28 truncate text-muted-foreground">{f.name}</span>
                <div className="flex-1 overflow-hidden rounded-full bg-secondary">
                  <div
                    className="h-1.5 rounded-full transition-all"
                    style={{ width: `${Math.round(f.value * 100)}%`, background: meta.colorHex }}
                  />
                </div>
                <span className="w-10 text-right tabular-nums">{(f.value * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Loss history si existe */}
      {info.lossHistory && info.lossHistory.length > 0 && (
        <div className="mt-3 border-t border-border/40 pt-3">
          <p className="mb-2 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
            Evolución de loss
          </p>
          <p className="text-[11px] text-muted-foreground">
            Último: <span className="tabular-nums font-semibold">{info.lossHistory[info.lossHistory.length - 1].toFixed(4)}</span>
            {" · "}
            ({info.lossHistory.length} muestras)
          </p>
        </div>
      )}

      <p className="mt-4 rounded-md border border-border/40 bg-secondary/20 px-2 py-1.5 text-[10px] text-muted-foreground">
        Muestras vistas: <span className="tabular-nums">{info.sampleCount.toLocaleString("es-ES")}</span>
        {" · "}
        Último reentrenamiento: <span className="tabular-nums">{info.lastTrainAt}</span>
      </p>
    </div>
  );
}

function Box({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div>
      <p className="text-[10px] text-muted-foreground">{label}</p>
      <p className="mt-0.5 text-sm font-bold tabular-nums" style={accent ? { color: accent } : undefined}>
        {value}
      </p>
    </div>
  );
}

const KEY_LABELS: Record<string, string> = {
  epsilon: "ε (explor.)",
  qStates: "Estados Q",
  alpha: "α (aprend.)",
  gamma: "γ (desc.)",
  aggressiveness: "Agresividad",
  trees: "Árboles",
  maxTrees: "Máx árboles",
  maxDepth: "Profundidad",
  learningRate: "Learning rate",
  lambda: "λ (L2)",
  zMomentum: "Z mom. umbral",
  zMeanReversion: "Z mean-rev",
  r2TrendMin: "R² mín",
  volatilityPenalty: "Pen. volat.",
  lastComposite: "Score comp.",
  nEstimators: "n estimators",
  nFeatSample: "n feat/sample",
  oobEstimate: "OOB est.",
  hiddenSize: "Hidden size",
  seqLength: "Sequence len",
  inputSize: "Input size",
  trainingSteps: "Pasos entren.",
};

function prettyKey(k: string): string {
  return KEY_LABELS[k] ?? k;
}

function formatVal(k: string, v: number): string {
  if (k === "learningRate" || k === "alpha") return v.toFixed(4);
  if (k === "aggressiveness" || k === "oobEstimate") return v.toFixed(2);
  if (k === "epsilon") return v.toFixed(3);
  if (k === "lambda") return v.toExponential(1);
  if (k === "lastComposite") return v.toFixed(3);
  return v.toLocaleString("es-ES", { maximumFractionDigits: 1 });
}
