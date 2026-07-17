import { useCallback, useMemo } from "react";
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
} from "recharts";
import type { EngineSnapshot, EquityPoint, Trade } from "@/lib/engine";
import { LOAN_PRINCIPAL } from "@/lib/engine";
import { fmtPct, fmtUSD, fmtTimeShort } from "@/lib/format";
import { AgentCard } from "@/sections/AgentCard";
import { PriceChart } from "@/sections/PriceChart";
import { TradesTable } from "@/sections/TradesTable";
import { Tuner } from "@/sections/Tuner";
import { ModelFocusPanel } from "@/sections/ModelFocusPanel";
import {
  MODELS,
  MODEL_IDS,
  AGGRESSION_LABELS,
  type ModelId,
} from "@/lib/registry";

interface Props {
  snap: EngineSnapshot;
  modelId: ModelId;
  onTune: (id: ModelId, level: 0 | 1 | 2 | 3 | 4) => void;
}

export function ModelPage({ snap, modelId, onTune }: Props) {
  const meta = MODELS[modelId];
  const m = snap.agents[modelId];
  const info = snap.modelInfo[modelId];
  const aggression = snap.aggressionPerModel[modelId];

  const trades = useMemo(
    () => snap.trades.filter((t) => t.agent === modelId) as unknown as Trade[],
    [snap.trades, modelId]
  );

  const handleTune = useCallback(
    (level: 0 | 1 | 2 | 3 | 4) => onTune(modelId, level),
    [modelId, onTune]
  );

  const netAfterLoan = m.netEquity - LOAN_PRINCIPAL;
  const positive = m.netReturn >= 0;

  return (
    <div className="space-y-5">
      <section className="glass-card rounded-2xl p-5">
        <div className="flex flex-wrap items-center gap-4">
          <div className={`flex h-12 w-12 items-center justify-center rounded-xl bg-secondary ${meta.tailwindText}`}>
            <span className="text-base font-bold">{meta.shortName}</span>
          </div>
          <div className="min-w-0 flex-1">
            <h2 className="text-sm font-bold tracking-tight">{meta.name}</h2>
            <p className="text-[11px] text-muted-foreground">{meta.fullName}</p>
          </div>
          <div className="text-right">
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Retorno neto</p>
            <p className={`text-2xl font-bold tabular-nums ${positive ? "text-emerald-400" : "text-rose-400"}`}>
              {fmtPct(m.netReturn)}
            </p>
            <p className="text-[11px] tabular-nums text-muted-foreground">
              Neto tras préstamo: {netAfterLoan >= 0 ? "+" : ""}{fmtUSD(netAfterLoan)}
            </p>
          </div>
        </div>
      </section>

      <Tuner
        modelId={modelId}
        current={aggression}
        onChange={handleTune}
        scaleLabel={AGGRESSION_LABELS[aggression]}
      />

      <section className="grid grid-cols-1 gap-5 xl:grid-cols-2">
        <AgentCard
          variant={modelId}
          metrics={m}
          subtitle={meta.description}
          signalLabel="Señal actual"
          signalValue={info.signal.toUpperCase()}
          signalTone={info.signal}
          aggression={aggression}
        />
        <ModelFocusPanel modelId={modelId} info={info} metrics={m} />
      </section>

      <SingleEquityCard snap={snap} modelId={modelId} />

      <section className="grid grid-cols-1 gap-5 xl:grid-cols-[1.2fr_1fr]">
        <PriceChart data={snap.equitySeries} trades={trades} />
        <TradesTable trades={trades} />
      </section>

      <RankingCard snap={snap} modelId={modelId} />
    </div>
  );
}

function SingleEquityCard({ snap, modelId }: { snap: EngineSnapshot; modelId: ModelId }) {
  const meta = MODELS[modelId];
  const chart = useMemo(() => {
    const data = snap.equitySeries;
    const maxPts = 500;
    const step = Math.max(1, Math.ceil(data.length / maxPts));
    return data.filter((_, i) => i % step === 0 || i === data.length - 1);
  }, [snap.equitySeries]);

  const values = chart.flatMap((p) => [p[modelId], p.bh]);
  const min = Math.min(...values, LOAN_PRINCIPAL);
  const max = Math.max(...values, LOAN_PRINCIPAL);
  const pad = Math.max((max - min) * 0.12, 50);
  const liveStart = chart.find((p) => p.live)?.t;

  return (
    <div className="glass-card rounded-2xl p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-bold tracking-tight">{meta.name} vs Buy & Hold</h3>
          <p className="text-[11px] text-muted-foreground">Equity neto de este modelo comparado con el benchmark</p>
        </div>
        <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-0.5 w-4 rounded" style={{ background: meta.colorHex }} />
            {meta.shortName}
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-0.5 w-4 rounded bg-slate-500" /> BH
          </span>
        </div>
      </div>
      <div className="h-[300px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chart} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
            <defs>
              <linearGradient id={`grad-${modelId}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={meta.colorHex} stopOpacity={0.30} />
                <stop offset="100%" stopColor={meta.colorHex} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(220 22% 14%)" vertical={false} />
            <XAxis dataKey="t" tickFormatter={fmtTimeShort} stroke="hsl(215 16% 45%)" fontSize={10} tickLine={false} axisLine={false} minTickGap={60} />
            <YAxis domain={[min - pad, max + pad]} tickCount={6} tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`} stroke="hsl(215 16% 45%)" fontSize={10} tickLine={false} axisLine={false} width={52} />
            <Tooltip
              content={((props: { active?: boolean; payload?: { payload: EquityPoint }[] }) => {
                if (!props.active || !props.payload?.length) return null;
                const p = props.payload[0].payload;
                return (
                  <div className="rounded-xl border border-border bg-popover/95 px-3 py-2 text-[11px] shadow-xl backdrop-blur">
                    <p className="text-[10px] text-muted-foreground">
                      {new Date(p.t).toLocaleString("es-ES", { hour: "2-digit", minute: "2-digit" })}
                    </p>
                    <p className="font-bold tabular-nums" style={{ color: meta.colorHex }}>
                      {fmtUSD(p[modelId])}
                    </p>
                    <p className="text-[10px] tabular-nums text-muted-foreground">BH: {fmtUSD(p.bh)}</p>
                  </div>
                );
              }) as never}
            />
            <ReferenceLine y={LOAN_PRINCIPAL} stroke="hsl(215 16% 40%)" strokeDasharray="6 4" />
            {liveStart && (
              <ReferenceLine x={liveStart} stroke="#f59e0b" strokeDasharray="4 4" label={{ value: "EN VIVO", position: "top", fontSize: 9, fill: "#f59e0b" }} />
            )}
            <Line type="monotone" dataKey="bh" stroke="#64748b" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            <Area type="monotone" dataKey={modelId} stroke={meta.colorHex} strokeWidth={2} fill={`url(#grad-${modelId})`} dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function RankingCard({ snap, modelId }: { snap: EngineSnapshot; modelId: ModelId }) {
  return (
    <section className="glass-card rounded-2xl p-5">
      <h3 className="mb-3 text-sm font-bold tracking-tight">Ranking en la arena</h3>
      <div className="space-y-2">
        {[...MODEL_IDS]
          .map((id) => ({ id, ret: snap.agents[id].netReturn }))
          .sort((a, b) => b.ret - a.ret)
          .map((row, i) => {
            const rm = MODELS[row.id];
            const isMe = row.id === modelId;
            return (
              <div
                key={row.id}
                className={`flex items-center justify-between rounded-lg border px-3 py-2 ${
                  isMe ? `${rm.tailwindRing} bg-secondary/40` : "border-border/40 bg-secondary/20"
                }`}
              >
                <span className="flex items-center gap-2 text-[12px]">
                  <span className="w-6 font-bold text-muted-foreground">#{i + 1}</span>
                  <span className="h-2 w-2 rounded-full" style={{ background: rm.colorHex }} />
                  <span className={`font-semibold ${isMe ? rm.tailwindText : ""}`}>{rm.name}</span>
                </span>
                <span className={`tabular-nums text-sm font-bold ${row.ret >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                  {fmtPct(row.ret)}
                </span>
              </div>
            );
          })}
      </div>
    </section>
  );
}

// EOF
