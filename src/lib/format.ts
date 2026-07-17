export function fmtUSD(v: number, decimals = 0): string {
  const abs = Math.abs(v);
  const d = abs >= 1000 ? decimals : 2;
  return (
    (v < 0 ? "-$" : "$") +
    abs.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d })
  );
}

export function fmtPct(v: number, decimals = 2): string {
  return `${v >= 0 ? "+" : ""}${v.toFixed(decimals)}%`;
}

export function fmtBTC(v: number): string {
  return `${v.toFixed(5)} BTC`;
}

export function fmtCompact(v: number): string {
  if (v >= 1e9) return `$${(v / 1e9).toFixed(2)}B`;
  if (v >= 1e6) return `$${(v / 1e6).toFixed(2)}M`;
  if (v >= 1e3) return `$${(v / 1e3).toFixed(1)}K`;
  return fmtUSD(v);
}

export function fmtTime(ts: number): string {
  return new Date(ts).toLocaleTimeString("es-ES", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function fmtTimeShort(ts: number): string {
  return new Date(ts).toLocaleTimeString("es-ES", { hour: "2-digit", minute: "2-digit" });
}
