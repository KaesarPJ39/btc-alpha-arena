import { useEffect, useRef, useState } from "react";
import { Activity, Bitcoin, CircleDot, Pause, Play, Wifi } from "lucide-react";
import type { MarketStats } from "@/lib/engine";
import { fmtPct, fmtTime, fmtUSD } from "@/lib/format";
import { Button } from "@/components/ui/button";

interface Props {
  market: MarketStats;
  status: string;
  statusDetail: string;
  running: boolean;
  onToggle: () => void;
}

export function Header({ market, status, statusDetail, running, onToggle }: Props) {
  const prevPrice = useRef(0);
  const [flash, setFlash] = useState<"up" | "down" | null>(null);

  useEffect(() => {
    if (prevPrice.current && market.price !== prevPrice.current) {
      setFlash(market.price > prevPrice.current ? "up" : "down");
      prevPrice.current = market.price;
      const t = setTimeout(() => setFlash(null), 650);
      return () => clearTimeout(t);
    }
    prevPrice.current = market.price;
  }, [market.price]);

  const up = market.change24h >= 0;

  return (
    <header className="sticky top-0 z-40 border-b border-border/70 bg-background/80 backdrop-blur-xl">
      <div className="mx-auto flex max-w-[1500px] flex-wrap items-center gap-x-6 gap-y-3 px-5 py-3">
        {/* Marca */}
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-400 to-emerald-600 shadow-lg shadow-emerald-500/20">
            <Activity className="h-5 w-5 text-emerald-950" strokeWidth={2.5} />
          </div>
          <div>
            <h1 className="text-[15px] font-bold leading-tight tracking-tight">
              BTC Alpha Arena
            </h1>
            <p className="text-[11px] leading-tight text-muted-foreground">
              Reinforcement Learning vs XGBoost · Trading en vivo
            </p>
          </div>
        </div>

        {/* Precio en vivo */}
        <div className="flex items-center gap-3 rounded-xl border border-border/70 bg-card/60 px-4 py-2">
          <Bitcoin className="h-5 w-5 text-amber-400" />
          <div>
            <div className="flex items-baseline gap-2">
              <span
                className={`text-xl font-bold tabular-nums tracking-tight ${
                  flash === "up" ? "tick-flash-up" : flash === "down" ? "tick-flash-down" : ""
                }`}
              >
                {market.price > 0 ? fmtUSD(market.price, 2) : "—"}
              </span>
              <span
                className={`text-xs font-semibold tabular-nums ${
                  up ? "text-emerald-400" : "text-rose-400"
                }`}
              >
                {fmtPct(market.change24h)} 24h
              </span>
            </div>
            <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
              <Wifi className="h-3 w-3" />
              <span>{market.source}</span>
              <span>·</span>
              <span>{market.lastUpdate ? fmtTime(market.lastUpdate) : "…"}</span>
            </div>
          </div>
        </div>

        {/* Estado */}
        <div className="ml-auto flex items-center gap-4">
          <div className="hidden items-center gap-2 md:flex">
            <span
              className={`h-2 w-2 rounded-full ${
                status === "live"
                  ? "bg-emerald-400 pulse-dot"
                  : status === "error"
                    ? "bg-rose-500"
                    : "bg-amber-400 animate-pulse"
              }`}
            />
            <div className="text-right">
              <p className="text-[11px] font-medium leading-tight">
                {status === "live" ? "Mercado en vivo" : status === "error" ? "Error" : "Preparando"}
              </p>
              <p className="text-[10px] leading-tight text-muted-foreground">{statusDetail}</p>
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={onToggle}
            className="gap-1.5 border-border/80 bg-card/60 text-xs hover:bg-accent"
          >
            {running ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
            {running ? "Pausar" : "Reanudar"}
          </Button>
        </div>
      </div>

      {/* Barra de stats 24h */}
      {market.high24h > 0 && (
        <div className="border-t border-border/50 bg-card/30">
          <div className="mx-auto flex max-w-[1500px] flex-wrap items-center gap-x-8 gap-y-1 px-5 py-1.5 text-[11px] text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <CircleDot className="h-3 w-3 text-amber-400" /> BTC/USDT
            </span>
            <span>
              Máx 24h <span className="font-medium text-foreground">{fmtUSD(market.high24h, 0)}</span>
            </span>
            <span>
              Mín 24h <span className="font-medium text-foreground">{fmtUSD(market.low24h, 0)}</span>
            </span>
            <span>
              Volumen 24h{" "}
              <span className="font-medium text-foreground">
                ${(market.volume24h / 1e9).toFixed(2)}B
              </span>
            </span>
          </div>
        </div>
      )}
    </header>
  );
}
