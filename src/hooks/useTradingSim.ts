import { useEffect, useRef, useState, useCallback } from "react";
import { TradingEngine, type EngineSnapshot } from "@/lib/engine";
import type { ModelId, AggressionLevel } from "@/lib/registry";

export function useTradingSim() {
  const engineRef = useRef<TradingEngine | null>(null);
  const [snap, setSnap] = useState<EngineSnapshot | null>(null);

  const startEngine = useCallback(() => {
    const engine = new TradingEngine();
    engineRef.current = engine;
    engine.onUpdate(() => setSnap(engine.snapshot()));
    void engine.start();
    return engine;
  }, []);

  useEffect(() => {
    const engine = startEngine();
    let cancelled = false;
    void engine.start().then(() => {
      if (cancelled) engine.stop();
    });
    return () => {
      cancelled = true;
      engine.stop();
      engineRef.current = null;
    };
  }, [startEngine]);

  const toggleRunning = useCallback(() => {
    engineRef.current?.toggleRunning();
  }, []);

  const setAggression = useCallback((id: ModelId, level: AggressionLevel) => {
    engineRef.current?.setAggression(id, level);
  }, []);

  const reset = useCallback(() => {
    engineRef.current?.stop();
    startEngine();
  }, [startEngine]);

  return { snap, toggleRunning, setAggression, reset };
}
