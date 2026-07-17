import { Swords } from "lucide-react";
import type { EngineSnapshot } from "@/lib/engine";
import { EquityChart } from "@/sections/EquityChart";
import { PriceChart } from "@/sections/PriceChart";
import { InsightsPanel } from "@/sections/InsightsPanel";
import { CreditPanel } from "@/sections/CreditPanel";
import { TradesTable } from "@/sections/TradesTable";
import { TradeLog, RiskDashboard } from "@/sections/TradeLog";
import { AgentCard } from "@/sections/AgentCard";
import { fmtPct } from "@/lib/format";
import { MODEL_IDS, MODELS } from "@/lib/registry";

interface Props {
  snap: EngineSnapshot;
}

export function ArenaPage({ snap }: Props) {
  const bhBench =
    snap.equitySeries.length > 0
      ? (snap.equitySeries[snap.equitySeries.length - 1].bh / 100_000 - 1) * 100
      : 0;
  const retornos = MODEL_IDS.map((id) => snap.agents[id].netReturn);
  const maxRet = Math.max(...retornos);
  const leaderIdx = retornos.indexOf(maxRet);
  const leaderId = leaderIdx >= 0 ? MODEL_IDS[leaderIdx] : null;

  return (
    <div className="space-y-5">
      {/* Marcador comparativo */}
      <section className="glass-card overflow-hidden rounded-2xl">
        <div className="flex flex-wrap items-center gap-4 px-6 py-4">
          <div className="flex h-11 w-11 items-center justify-center rounded-full border border-amber-500/40 bg-amber-500/10">
            <Swords className="h-5 w-5 text-amber-400" />
          </div>
          <div className="min-w-0 flex-1">
            <h2 className="text-sm font-bold tracking-tight">Arena · Batalla de 5 modelos</h2>
            <p className="text-[11px] text-muted-foreground">
              Q-Learning · Gradient Boosting · Statistical · Random Forest · GRU · comparando en vivo contra Buy & Hold
            </p>
          </div>
          <div className="text-right">
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Líder actual</p>
            <p className={`text-lg font-bold ${leaderId ? MODELS[leaderId].tailwindText : "text-foreground"}`}>
              {leaderId ? MODELS[leaderId].name : "—"}
            </p>
            <p className="text-[11px] tabular-nums text-muted-foreground">
              {leaderId ? `${fmtPct(Math.abs(maxRet))} neto · BH: ${fmtPct(bhBench)}` : "Sin datos"}
            </p>
          </div>
        </div>
        <div className="border-t border-border/40 px-6 py-3">
          <div className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-3 lg:grid-cols-5">
            {MODEL_IDS.map((id) => {
              const m = snap.agents[id];
              const positive = m.netReturn >= 0;
              const isLeader = id === leaderId;
              return (
                <div
                  key={id}
                  className={`rounded-xl border px-3 py-2 ${
                    isLeader ? "border-amber-500/40 bg-amber-500/5" : "border-border/40 bg-secondary/20"
                  }`}
                >
                  <p className={`truncate text-[11px] font-semibold ${MODELS[id].tailwindText}`}>
                    {MODELS[id].shortName}
                    {isLeader && " ★"} 
                  </p>
                  <p className={`mt-0.5 text-base font-bold tabular-nums ${positive ? "text-emerald-400" : "text-rose-400"}`}>
                    {fmtPct(m.netReturn)}
                  </p>
                  <p className="text-[10px] text-muted-foreground tabular-nums">
                    Sharpe {m.sharpe.toFixed(2)} · {m.trades} ops
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Tarjetas de modelo (5) */}
      <section className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
        {MODEL_IDS.map((id) => (
          <AgentCard
            key={id}
            variant={id}
            metrics={snap.agents[id]}
            subtitle={MODELS[id].fullName}
            signalLabel="Señal actual"
            signalValue={snap.modelInfo[id].signal.toUpperCase()}
            signalTone={snap.modelInfo[id].signal}
            leader={id === leaderId}
            aggression={snap.aggressionPerModel[id]}
          />
        ))}
        {/* Celda extra: Buy & Hold */}
        <div className="glass-card rounded-2xl p-5 ring-1 ring-slate-500/10">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-secondary text-slate-400">
              <span className="text-sm font-bold">BH</span>
            </div>
            <div>
              <h3 className="text-sm font-bold tracking-tight">Buy & Hold</h3>
              <p className="text-[11px] text-muted-foreground">Benchmark · compra y mantiene</p>
            </div>
          </div>
          <div className="mt-4 flex items-end justify-between">
            <div>
              <p className="text-[11px] uppercase tracking-wider text-muted-foreground">Equity BH</p>
              <p className="text-2xl font-bold tabular-nums tracking-tight">
                {snap.equitySeries.length > 0
                  ? `$${snap.equitySeries[snap.equitySeries.length - 1].bh.toLocaleString("en-US", { maximumFractionDigits: 0 })}`
                  : "—"}
              </p>
            </div>
            <p className={`text-sm font-semibold tabular-nums ${bhBench >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
              {fmtPct(bhBench)}
            </p>
          </div>
          <p className="mt-4 text-[10px] leading-relaxed text-muted-foreground/80">
            Compara en static con $100,000 en BTC al inicio. Pago del mismo interés del crédito del 9.5% TAE.
          </p>
        </div>
      </section>

      {/* Gráfico de equity */}
      <EquityChart data={snap.equitySeries} />

      {/* Precio + trades */}
      <section className="grid grid-cols-1 gap-5 xl:grid-cols-[1.2fr_1fr]">
        <PriceChart data={snap.equitySeries} trades={snap.trades} />
        <TradesTable trades={snap.trades} />
      </section>

      {/* Log de operaciones y dashboard de riesgo */}
      <TradeLog snap={snap} />
      <RiskDashboard snap={snap} />

      {/* Crédito */}
      <CreditPanel snap={snap} />

      {/* Interior de los modelos */}
      <InsightsPanel snap={snap} />

      <footer className="border-t border-border/50 pt-4 pb-6 text-center text-[10px] leading-relaxed text-muted-foreground">
        <p>
          BTC Alpha Arena — Simulación con precios reales de BTC/USD vía API pública.
          Ahora con backend Node.js + WebSocket, risk management y persistencia de modelos.
        </p>
        <p className="mt-1">
          No constituye asesoramiento financiero. Rentabilidades pasadas no garantizan resultados futuros.
        </p>
      </footer>
    </div>
  );
}
