import { ArrowDownLeft, ArrowUpRight, ShieldAlert, Target, TrendingUp } from "lucide-react";
import type { EngineSnapshot } from "@/lib/engine";
import { fmtUSD, fmtTimeShort } from "@/lib/format";
import { MODELS, MODEL_IDS } from "@/lib/registry";

interface Props {
  snap: EngineSnapshot;
}

function reasonIcon(reason: string) {
  if (reason.includes("stop-loss")) return <ShieldAlert className="h-3.5 w-3.5 text-rose-400" />;
  if (reason.includes("take-profit")) return <Target className="h-3.5 w-3.5 text-emerald-400" />;
  if (reason.includes("trailing")) return <TrendingUp className="h-3.5 w-3.5 text-amber-400" />;
  return null;
}

export function TradeLog({ snap }: Props) {
  const trades = snap.trades.slice(0, 50);

  return (
    <section className="glass-card rounded-2xl p-5">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-bold tracking-tight">Log de operaciones recientes</h3>
          <p className="text-[11px] text-muted-foreground">Últimas 50 señales ejecutadas por los 5 modelos</p>
        </div>
        <div className="text-right text-[10px] text-muted-foreground">
          Total ops: <span className="font-medium text-foreground">{snap.trades.length}</span>
        </div>
      </div>
      <div className="max-h-[400px] overflow-auto pr-1">
        <table className="w-full text-[11px]">
          <thead className="sticky top-0 bg-card/95 text-muted-foreground">
            <tr>
              <th className="py-2 text-left font-medium">Hora</th>
              <th className="py-2 text-left font-medium">Modelo</th>
              <th className="py-2 text-left font-medium">Side</th>
              <th className="py-2 text-right font-medium">Precio</th>
              <th className="py-2 text-right font-medium">Qty</th>
              <th className="py-2 text-right font-medium">Valor</th>
              <th className="py-2 text-left font-medium">Razón</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/40">
            {trades.map((t, i) => (
              <tr key={`${t.ts}-${t.agent}-${i}`} className="hover:bg-secondary/30">
                <td className="py-2 tabular-nums text-muted-foreground">{fmtTimeShort(t.ts)}</td>
                <td className="py-2">
                  <span className="font-semibold" style={{ color: MODELS[t.agent].colorHex }}>
                    {MODELS[t.agent].shortName}
                  </span>
                </td>
                <td className="py-2">
                  <span className={`flex items-center gap-1 font-semibold ${t.side === "BUY" ? "text-emerald-400" : "text-rose-400"}`}>
                    {t.side === "BUY" ? <ArrowDownLeft className="h-3.5 w-3.5" /> : <ArrowUpRight className="h-3.5 w-3.5" />}
                    {t.side}
                  </span>
                </td>
                <td className="py-2 text-right tabular-nums">{fmtUSD(t.price, 1)}</td>
                <td className="py-2 text-right tabular-nums">{t.qty.toFixed(6)}</td>
                <td className="py-2 text-right tabular-nums">{fmtUSD(t.value)}</td>
                <td className="py-2">
                  <span className="flex items-center gap-1.5">
                    {reasonIcon(t.reason)}
                    <span className="truncate max-w-[140px] text-muted-foreground">{t.reason}</span>
                  </span>
                </td>
              </tr>
            ))}
            {trades.length === 0 && (
              <tr>
                <td colSpan={7} className="py-8 text-center text-muted-foreground">
                  Aún no hay operaciones registradas.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function RiskDashboard({ snap }: Props) {
  return (
    <section className="glass-card rounded-2xl p-5">
      <h3 className="mb-4 text-sm font-bold tracking-tight">Dashboard de riesgo</h3>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {MODEL_IDS.map((id) => {
          const m = snap.agents[id];
          const dd = m.currentDD ?? 0;
          return (
            <div key={id} className="rounded-xl border border-border/40 bg-secondary/20 p-3">
              <div className="mb-2 flex items-center gap-2">
                <span className="h-2 w-2 rounded-full" style={{ background: MODELS[id].colorHex }} />
                <span className="text-[11px] font-semibold">{MODELS[id].shortName}</span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-[10px]">
                <div>
                  <p className="text-muted-foreground">Exposición</p>
                  <p className="font-medium tabular-nums">{(m.exposure * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Drawdown</p>
                  <p className={`font-medium tabular-nums ${dd > 10 ? "text-rose-400" : ""}`}>{dd.toFixed(1)}%</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Pos. abiertas</p>
                  <p className="font-medium tabular-nums">{m.openPositions ?? 0}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Win rate</p>
                  <p className="font-medium tabular-nums">{m.winRate.toFixed(1)}%</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
