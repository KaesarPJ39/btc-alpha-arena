import { Landmark, Percent, Scale, Timer } from "lucide-react";
import type { EngineSnapshot } from "@/lib/engine";
import { LOAN_APR, LOAN_PRINCIPAL } from "@/lib/engine";
import { fmtUSD } from "@/lib/format";
import { MODEL_IDS, MODELS, type ModelId } from "@/lib/registry";

interface Props {
  snap: EngineSnapshot;
}

export function CreditPanel({ snap }: Props) {
  const nets = MODEL_IDS.map((id) => snap.agents[id].netEquity - LOAN_PRINCIPAL);
  const maxNet = Math.max(...nets);
  const winner = nets.indexOf(maxNet) >= 0 ? MODEL_IDS[nets.indexOf(maxNet)] : null;
  const totalInterest = MODEL_IDS.reduce((a, id) => a + snap.agents[id].interest, 0);

  return (
    <div className="glass-card rounded-2xl p-5">
      <div className="mb-4 flex items-center gap-2">
        <Landmark className="h-4 w-4 text-amber-400" />
        <div>
          <h3 className="text-sm font-bold tracking-tight">Simulación de crédito</h3>
          <p className="text-[11px] text-muted-foreground">
            Cada modelo opera con un préstamo apalancado de {fmtUSD(LOAN_PRINCIPAL)} al {(LOAN_APR * 100).toFixed(1)}% TAE
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <CreditStat icon={Landmark} label="Principal por modelo" value={fmtUSD(LOAN_PRINCIPAL)} />
        <CreditStat icon={Percent} label="Tipo anual (TAE)" value={`${(LOAN_APR * 100).toFixed(2)}%`} />
        <CreditStat icon={Timer} label="Interés total devengado" value={fmtUSD(totalInterest, 2)} tone="warn" />
        <CreditStat icon={Timer} label="Modelos activos" value={`${MODEL_IDS.length}`} />
      </div>

      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
        {MODEL_IDS.map((id, i) => (
          <Settlement
            key={id}
            id={id}
            net={nets[i]}
            interest={snap.agents[id].interest}
            winner={winner === id}
          />
        ))}
      </div>

      <div className="mt-4 flex items-start gap-2 rounded-xl border border-amber-500/20 bg-amber-500/5 px-3 py-2.5 text-[11px] leading-relaxed text-muted-foreground">
        <Scale className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-400" />
        <p>
          <span className="font-semibold text-amber-300">Condiciones del crédito:</span> el interés se devenga
          de forma continua sobre el principal ({fmtUSD(LOAN_PRINCIPAL)}) y se descuenta del equity. Para ser
          rentable, cada estrategia debe superar el coste financiero del préstamo más las comisiones de trading
          (0.10% por operación). "Neto tras devolver préstamo" muestra el beneficio real una vez repagado el principal.
        </p>
      </div>
    </div>
  );
}

function CreditStat({
  icon: Icon,
  label,
  value,
  tone,
}: {
  icon: typeof Landmark;
  label: string;
  value: string;
  tone?: "warn";
}) {
  return (
    <div className="rounded-xl border border-border/60 bg-secondary/30 px-3 py-2.5">
      <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
        <Icon className="h-3 w-3" />
        {label}
      </div>
      <p className={`mt-1 text-sm font-bold tabular-nums ${tone === "warn" ? "text-amber-300" : ""}`}>{value}</p>
    </div>
  );
}

function Settlement({
  id,
  net,
  interest,
  winner,
}: {
  id: ModelId;
  net: number;
  interest: number;
  winner: boolean;
}) {
  const meta = MODELS[id];
  return (
    <div className="rounded-xl border border-border/60 bg-secondary/20 px-4 py-3">
      <div className="flex items-center justify-between">
        <span className={`text-[11px] font-semibold ${meta.tailwindText}`}>{meta.name}</span>
        {winner && (
          <span className="rounded-full bg-amber-500/15 px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider text-amber-300">
            Mejor neto
          </span>
        )}
      </div>
      <p className={`mt-1 text-lg font-bold tabular-nums ${net >= 0 ? "text-emerald-300" : "text-rose-300"}`}>
        {net >= 0 ? "+" : ""}
        {fmtUSD(net)}
      </p>
      <p className="text-[10px] text-muted-foreground">
        Neto tras préstamo · interés: {fmtUSD(interest, 2)}
      </p>
    </div>
  );
}
