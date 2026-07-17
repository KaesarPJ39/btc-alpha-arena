import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EquityPoint, Trade } from "@/lib/engine";
import { fmtTimeShort, fmtUSD } from "@/lib/format";
import { MODEL_IDS, MODELS } from "@/lib/registry";

interface Props {
  data: EquityPoint[];
  trades: Trade[];
}

export function PriceChart({ data, trades }: Props) {
  // Últimas ~240 barras para que se aprecien los marcadores
  const chart = useMemo(() => data.slice(-240), [data]);

  const markers = useMemo(() => {
    if (chart.length === 0) return [];
    const t0 = chart[0].t;
    const t1 = chart[chart.length - 1].t;
    const byTime = new Map(chart.map((p) => [p.t, p.price]));
    // casar cada trade con la barra más cercana
    return trades
      .filter((tr) => tr.ts >= t0 && tr.ts <= t1 + 60_000)
      .slice(0, 30)
      .map((tr) => {
        let bestT = tr.ts;
        let bestD = Infinity;
        for (const t of byTime.keys()) {
          const d = Math.abs(t - tr.ts);
          if (d < bestD) {
            bestD = d;
            bestT = t;
          }
        }
        return { ...tr, t: bestT, y: byTime.get(bestT) ?? tr.price };
      });
  }, [chart, trades]);

  const prices = chart.map((p) => p.price);
  const min = Math.min(...prices);
  const max = Math.max(...prices);
  const pad = Math.max((max - min) * 0.1, 5);

  return (
    <div className="glass-card rounded-2xl p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-bold tracking-tight">BTC/USD · ejecuciones</h3>
          <p className="text-[11px] text-muted-foreground">
            Precio real con marcadores de compra/venta de los 5 modelos
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3 text-[10px] text-muted-foreground">
          {MODEL_IDS.map((id) => (
            <span key={id} className="flex items-center gap-1">
              <MarkerDot color={MODELS[id].colorHex} /> {MODELS[id].shortName}
            </span>
          ))}
          <span>▲ compra · ▼ venta</span>
        </div>
      </div>
      <div className="h-[240px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chart} margin={{ top: 6, right: 8, bottom: 0, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(220 22% 14%)" vertical={false} />
            <XAxis
              dataKey="t"
              tickFormatter={fmtTimeShort}
              stroke="hsl(215 16% 45%)"
              fontSize={10}
              tickLine={false}
              axisLine={false}
              minTickGap={60}
            />
            <YAxis
              domain={[min - pad, max + pad]}
              tickFormatter={(v: number) => `$${(v / 1000).toFixed(1)}k`}
              stroke="hsl(215 16% 45%)"
              fontSize={10}
              tickLine={false}
              axisLine={false}
              width={56}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const p = payload[0].payload as EquityPoint;
                return (
                  <div className="rounded-xl border border-border bg-popover/95 px-3 py-2 text-[11px] shadow-xl backdrop-blur">
                    <p className="text-muted-foreground">{fmtTimeShort(p.t)}</p>
                    <p className="font-bold tabular-nums text-amber-400">{fmtUSD(p.price, 2)}</p>
                  </div>
                );
              }}
            />
            <Line type="monotone" dataKey="price" stroke="#f59e0b" strokeWidth={1.8} dot={false} isAnimationActive={false} />
            {markers.map((m, i) => (
              <ReferenceDot
                key={`${m.ts}-${i}`}
                x={m.t}
                y={m.y}
                shape={(props: { cx?: number; cy?: number }) => {
                  const cx = props.cx ?? 0;
                  const cy = props.cy ?? 0;
                  const color = MODELS[m.agent].colorHex;
                  return m.side === "BUY" ? (
                    <path d={`M ${cx} ${cy - 5} L ${cx + 5} ${cy + 4} L ${cx - 5} ${cy + 4} Z`} fill={color} stroke="#0b1220" strokeWidth={1} />
                  ) : (
                    <path d={`M ${cx} ${cy + 5} L ${cx + 5} ${cy - 4} L ${cx - 5} ${cy - 4} Z`} fill={color} stroke="#0b1220" strokeWidth={1} />
                  );
                }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function MarkerDot({ color }: { color: string }) {
  return <span className="inline-block h-2 w-2 rounded-full" style={{ background: color }} />;
}
