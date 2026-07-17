import { useEffect, useRef, useState } from "react";
import { Activity, Bitcoin, CircleDot, KeyRound, Pause, Play, RotateCcw, Wifi } from "lucide-react";
import type { MarketStats } from "@/lib/engine";
import { fmtPct, fmtTime, fmtUSD } from "@/lib/format";
import { Button } from "@/components/ui/button";

const RESET_PASSWORD = "admin27";

interface Props {
  market: MarketStats;
  status: string;
  statusDetail: string;
  running: boolean;
  onToggle: () => void;
  onReset: () => void;
}

export function Header({ market, status, statusDetail, running, onToggle, onReset }: Props) {
  const prevPrice = useRef(0);
  const [flash, setFlash] = useState<"up" | "down" | null>(null);
  const [showReset, setShowReset] = useState(false);
  const [pwd, setPwd] = useState("");
  const [pwdError, setPwdError] = useState(false);
  const pwdRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (prevPrice.current && market.price !== prevPrice.current) {
      setFlash(market.price > prevPrice.current ? "up" : "down");
      prevPrice.current = market.price;
      const t = setTimeout(() => setFlash(null), 650);
      return () => clearTimeout(t);
    }
    prevPrice.current = market.price;
  }, [market.price]);

  useEffect(() => {
    if (showReset) {
      setPwd("");
      setPwdError(false);
      setTimeout(() => pwdRef.current?.focus(), 50);
    }
  }, [showReset]);

  const handleReset = () => {
    if (pwd === RESET_PASSWORD) {
      setShowReset(false);
      onReset();
    } else {
      setPwdError(true);
      setPwd("");
      pwdRef.current?.focus();
    }
  };

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
              Q-Learning · XGBoost · Statistical · Random Forest · RNN · BTC/USD en vivo
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
        <div className="ml-auto flex items-center gap-3">
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
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowReset(true)}
            className="gap-1.5 border-rose-500/30 bg-card/60 text-xs text-rose-400 hover:bg-rose-500/10 hover:text-rose-300"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Reset
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

      {/* Diálogo de reset con contraseña */}
      {showReset && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="glass-card mx-4 w-full max-w-sm rounded-2xl border border-rose-500/20 p-6 shadow-2xl">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-rose-500/10">
                <KeyRound className="h-5 w-5 text-rose-400" />
              </div>
              <div>
                <h3 className="text-sm font-bold">Reiniciar simulación</h3>
                <p className="text-[11px] text-muted-foreground">
                  Se borrarán todos los datos de los 5 modelos
                </p>
              </div>
            </div>
            <p className="mb-3 text-[11px] leading-relaxed text-muted-foreground/80">
              Esto reiniciará equity, trades, historial de modelos y volverá al estado inicial con $100,000 por cuenta. La acción no se puede deshacer.
            </p>
            <input
              ref={pwdRef}
              type="password"
              placeholder="Contraseña"
              value={pwd}
              onChange={(e) => { setPwd(e.target.value); setPwdError(false); }}
              onKeyDown={(e) => e.key === "Enter" && handleReset()}
              className={`mb-3 w-full rounded-lg border bg-card/60 px-3 py-2 text-sm outline-none transition-colors placeholder:text-muted-foreground/50 focus:ring-1 ${
                pwdError
                  ? "border-rose-500/60 focus:border-rose-500 focus:ring-rose-500/30"
                  : "border-border/60 focus:border-emerald-500 focus:ring-emerald-500/30"
              }`}
            />
            {pwdError && (
              <p className="mb-2 text-[11px] font-medium text-rose-400">Contraseña incorrecta</p>
            )}
            <div className="flex justify-end gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowReset(false)}
                className="text-xs"
              >
                Cancelar
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={handleReset}
                className="gap-1.5 border-rose-500/40 bg-rose-500/10 text-xs text-rose-400 hover:bg-rose-500/20"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                Reiniciar
              </Button>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
