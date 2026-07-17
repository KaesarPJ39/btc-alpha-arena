import { useMemo } from "react";
import {
  Area,
  Line,
  ComposedChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Legend,
} from "recharts";
import type { EquityPoint } from "@/lib/engine";
import { LOAN_PRINCIPAL } from "@/lib/engine";
import { fmtTimeShort, fmtUSD } from "@/lib/format";
import { MODEL_IDS, MODELS } from "@/lib/registry";

interface Props {
  data: EquityPoint[];
}

export function EquityChart({ data }: Props) {
  const chart = useMemo(() => {
    const maxPts = 500;
    const step = Math.max(1, Math.ceil(data.length / maxPts));
    return data.filter((_, i) => i % step === 0 || i === data.length - 1);
  }, [data]);

  const liveStart = useMemo(() => chart.find((p) => p.live)?.t, [chart]);
  const keys = useMemo(() => {
    const values = chart.flatMap((p) => MODEL_IDS.map((id) => p[id]));
    return { min: Math.min(...values, LOAN_PRINCIPAL), max: Math.max(...values, LOAN_PRINCIPAL) };
  }, [chart]);
  const pad = Math.max((keys.max - keys.min) * 0.15, 50);

  return (
    <div className="glass-card rounded-2xl p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-bold tracking-tight">Equity neto en tiempo real</h3>
          <p className="text-[11px] text-muted-foreground">
            Valor de cartera tras intereses del crédito · backtest 1m + ejecución en vivo
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3 text-[10px] text-muted-foreground">
          {MODEL_IDS.map((id) => (
            <span key={id} className="flex items-center gap-1.5">
              <span className="inline-block h-0.5 w-3 rounded" style={{ background: MODELS[id].colorHex }} />
              {MODELS[id].shortName}
            </span>
          ))}
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-0.5 w-3 rounded bg-slate-500" /> BH
          </span>
        </div>
      </div>
      <div className="h-[340px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chart} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
            <defs>
              {MODEL_IDS.map((id) => (
                <linearGradient key={id} id={`grad-${id}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={MODELS[id].colorHex} stopOpacity={0.25} />
                  <stop offset="100%" stopColor={MODELS[id].colorHex} stopOpacity={0} />
                </linearGradient>
              ))}
            </defs>
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
              domain={[keys.min - pad, keys.max + pad]}
              tickCount={6}
              tickFormatter={(v: number) => `$${(v / 1000).toFixed(2)}k`}
              stroke="hsl(215 16% 45%)"
              fontSize={10}
              tickLine={false}
              axisLine={false}
              width={62}
            />
            <Tooltip content={<EquityTooltip />} />
            <Legend wrapperStyle={{ display: "none" }} />
            <ReferenceLine
              y={LOAN_PRINCIPAL}
              stroke="hsl(215 16% 40%)"
              strokeDasharray="6 4"
              label={{ value: "Capital inicial", position: "insideTopLeft", fontSize: 10, fill: "hsl(215 16% 50%)" }}
            />
            {liveStart && (
              <ReferenceLine
                x={liveStart}
                stroke="#f59e0b"
                strokeDasharray="4 4"
                label={{ value: "EN VIVO", position: "top", fontSize: 9, fill: "#f59e0b", fontWeight: 700 }}
              />
            )}
            <Line type="monotone" dataKey="bh" stroke="#64748b" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            {MODEL_IDS.map((id, i) => (
              <Area
                key={id}
                type="monotone"
                dataKey={id}
                stroke={MODELS[id].colorHex}
                strokeWidth={i === 0 ? 2.2 : 1.6}
                fill={`url(#grad-${id})`}
                dot={false}
                isAnimationActive={false}
              />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function EquityTooltip({ active, payload }: { active?: boolean; payload?: { payload: EquityPoint }[] }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  const row = (label: string, v: number, color: string) => (
    <div className="flex items-center justify-between gap-6 text-[11px]">
      <span className="flex items-center gap-1.5 text-muted-foreground">
        <span className="h-2 w-2 rounded-full" style={{ background: color }} />
        {label}
      </span>
      <span className="font-semibold tabular-nums" style={{ color }}>
        {fmtUSD(v)}
      </span>
    </div>
  );
  return (
    <div className="space-y-1.5 rounded-xl border border-border bg-popover/95 px-3 py-2.5 shadow-xl backdrop-blur">
      <p className="text-[10px] text-muted-foreground">
        {new Date(p.t).toLocaleString("es-ES", { hour: "2-digit", minute: "2-digit", second: "2-digit", day: "2-digit", month: "short" })}
        {p.live && <span className="ml-1.5 font-bold text-amber-400">LIVE</span>}
      </p>
      {MODEL_IDS.map((id) => row(MODELS[id].name, p[id], MODELS[id].colorHex))}
      {row("Buy & Hold", p.bh, "#64748b")}
      <div className="border-t border-border/60 pt-1 text-[10px] text-muted-foreground">
        BTC {fmtUSD(p.price, 2)}
      </div>
    </div>
  );
}
