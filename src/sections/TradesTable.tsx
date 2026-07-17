import { ArrowDownRight, ArrowUpRight, ScrollText } from "lucide-react";
import type { Trade } from "@/lib/engine";
import { fmtTime, fmtUSD } from "@/lib/format";
import { MODELS } from "@/lib/registry";

interface Props {
  trades: Trade[];
}

export function TradesTable({ trades }: Props) {
  return (
    <div className="glass-card rounded-2xl p-5">
      <div className="mb-4 flex items-center gap-2">
        <ScrollText className="h-4 w-4 text-slate-400" />
        <div>
          <h3 className="text-sm font-bold tracking-tight">Registro de operaciones</h3>
          <p className="text-[11px] text-muted-foreground">Ejecuciones simuladas más recientes (0.10% comisión)</p>
        </div>
      </div>
      <div className="max-h-[300px] overflow-y-auto pr-1">
        {trades.length === 0 ? (
          <p className="py-8 text-center text-[12px] text-muted-foreground">
            Aún no hay operaciones — los modelos están calibrando…
          </p>
        ) : (
          <table className="w-full text-[11px]">
            <thead className="sticky top-0 bg-card/95 backdrop-blur">
              <tr className="text-left text-[10px] uppercase tracking-wider text-muted-foreground">
                <th className="pb-2 pr-2 font-medium">Hora</th>
                <th className="pb-2 pr-2 font-medium">Modelo</th>
                <th className="pb-2 pr-2 font-medium">Lado</th>
                <th className="pb-2 pr-2 text-right font-medium">Cantidad</th>
                <th className="pb-2 pr-2 text-right font-medium">Precio</th>
                <th className="pb-2 pr-2 text-right font-medium">Valor</th>
                <th className="pb-2 text-right font-medium">Motivo</th>
              </tr>
            </thead>
            <tbody>
              {trades.slice(0, 24).map((tr, i) => (
                <tr key={`${tr.ts}-${tr.agent}-${i}`} className="border-t border-border/40">
                  <td className="py-1.5 pr-2 tabular-nums text-muted-foreground">{fmtTime(tr.ts)}</td>
                  <td className="py-1.5 pr-2">
                    <span
                      className={`rounded border px-1.5 py-0.5 text-[10px] font-bold ${MODELS[tr.agent].tailwindBadge}`}
                    >
                      {MODELS[tr.agent].shortName}
                    </span>
                  </td>
                  <td className="py-1.5 pr-2">
                    <span
                      className={`flex w-fit items-center gap-0.5 font-semibold ${
                        tr.side === "BUY" ? "text-emerald-400" : "text-rose-400"
                      }`}
                    >
                      {tr.side === "BUY" ? (
                        <ArrowUpRight className="h-3 w-3" />
                      ) : (
                        <ArrowDownRight className="h-3 w-3" />
                      )}
                      {tr.side === "BUY" ? "Compra" : "Venta"}
                    </span>
                  </td>
                  <td className="py-1.5 pr-2 text-right tabular-nums">{tr.qty.toFixed(5)}</td>
                  <td className="py-1.5 pr-2 text-right tabular-nums">{fmtUSD(tr.price, 2)}</td>
                  <td className="py-1.5 pr-2 text-right tabular-nums">{fmtUSD(tr.value)}</td>
                  <td className="py-1.5 text-right text-[10px] text-muted-foreground">{tr.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
