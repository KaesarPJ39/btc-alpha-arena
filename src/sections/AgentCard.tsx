import { BrainCircuit, LineChart, Sigma, TreePine, Network, TrendingDown, TrendingUp, type LucideIcon } from "lucide-react";
import type { AgentMetrics } from "@/lib/engine";
import { LOAN_PRINCIPAL } from "@/lib/engine";
import { fmtBTC, fmtPct, fmtUSD } from "@/lib/format";
import { Progress } from "@/components/ui/progress";
import { MODELS, type ModelId } from "@/lib/registry";

const ICONS: Record<ModelId, LucideIcon> = {
  rl: BrainCircuit,
  xgb: LineChart,
  stat: Sigma,
  rf: TreePine,
  lstm: Network,
};

interface Props {
  variant: ModelId;
  metrics: AgentMetrics;
  subtitle?: string;
  signalLabel?: string;
  signalValue?: string;
  signalTone?: "buy" | "sell" | "hold";
  leader?: boolean;
  aggression?: number;
}

export function AgentCard({
  variant,
  metrics: m,
  subtitle,
  signalLabel = "Señal",
  signalValue,
  signalTone = "hold",
  leader = false,
  aggression,
}: Props) {
  const meta = MODELS[variant];
  const Icon = ICONS[variant];
  const positive = m.netReturn >= 0;

  return (
    <div className={`glass-card relative overflow-hidden rounded-2xl p-5 ring-1 ${leader ? meta.tailwindRing : "ring-transparent"}`}>
      {leader && (
        <span className={`absolute right-4 top-4 rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${meta.tailwindBadge}`}>
          Líder
        </span>
      )}
      <div className="flex items-center gap-3">
        <div className={`flex h-10 w-10 items-center justify-center rounded-xl bg-secondary ${meta.tailwindText}`}>
          <Icon className="h-5 w-5" />
        </div>
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-bold tracking-tight">{meta.name}</h3>
          <p className="truncate text-[11px] text-muted-foreground">
            {subtitle ?? meta.description}
          </p>
        </div>
        {aggression !== undefined && (
          <span className={`rounded-md border px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${meta.tailwindBadge}`}>
            {["C", "M", "N", "A", "T"][aggression]}
          </span>
        )}
      </div>

      <div className="mt-4 flex items-end justify-between">
        <div>
          <p className="text-[11px] uppercase tracking-wider text-muted-foreground">Equity neto</p>
          <p className="text-2xl font-bold tabular-nums tracking-tight">{fmtUSD(m.netEquity)}</p>
        </div>
        <div className={`flex items-center gap-1 text-sm font-semibold tabular-nums ${positive ? "text-emerald-400" : "text-rose-400"}`}>
          {positive ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
          {fmtPct(m.netReturn)}
        </div>
      </div>

      <div className="mt-4">
        <div className="mb-1 flex justify-between text-[11px] text-muted-foreground">
          <span>Exposición en BTC</span>
          <span className="tabular-nums">{Math.round(m.exposure * 100)}%</span>
        </div>
        <Progress value={m.exposure * 100} className={`h-1.5 bg-secondary ${meta.tailwindBar}`} />
      </div>

      <div className="mt-4 grid grid-cols-3 gap-x-3 gap-y-3 border-t border-border/60 pt-4 text-[11px]">
        <Metric label="Efectivo" value={fmtUSD(m.cash)} />
        <Metric label="En BTC" value={fmtUSD(m.btcValue)} title={fmtBTC(m.btc)} />
        <Metric label="Interés devengado" value={fmtUSD(m.interest, 2)} tone="warn" />
        <Metric label="Sharpe" value={m.sharpe.toFixed(2)} />
        <Metric label="Max DD" value={`-${m.maxDD.toFixed(2)}%`} tone="bad" />
        <Metric label="Win rate" value={m.realizedCloses ? `${m.winRate.toFixed(0)}%` : "—"} />
      </div>

      {signalValue && (
        <div className="mt-4 flex items-center justify-between rounded-xl border border-border/60 bg-secondary/40 px-3 py-2">
          <span className="text-[11px] text-muted-foreground">{signalLabel}</span>
          <span
            className={`rounded-md px-2 py-0.5 text-[11px] font-bold ${
              signalTone === "buy"
                ? "bg-emerald-500/15 text-emerald-300"
                : signalTone === "sell"
                  ? "bg-rose-500/15 text-rose-300"
                  : "bg-secondary text-muted-foreground"
            }`}
          >
            {signalValue}
          </span>
        </div>
      )}

      <p className="mt-3 text-[10px] leading-relaxed text-muted-foreground/80">
        Capital inicial: crédito de {fmtUSD(LOAN_PRINCIPAL)} · {m.trades} operaciones ({m.buys}C / {m.sells}V)
      </p>
    </div>
  );
}

function Metric({ label, value, tone, title }: { label: string; value: string; tone?: "warn" | "bad"; title?: string }) {
  return (
    <div title={title}>
      <p className="text-muted-foreground">{label}</p>
      <p
        className={`mt-0.5 font-semibold tabular-nums ${
          tone === "warn" ? "text-amber-300" : tone === "bad" ? "text-rose-300" : "text-foreground"
        }`}
      >
        {value}
      </p>
    </div>
  );
}
