import { useEffect, useRef, useState, useCallback } from "react";
import { TradingEngine, type EngineSnapshot, GradientBoostingModel, RandomForestModel, GRUModel } from "@btc-arena/core";
import type { ModelId, AggressionLevel } from "@btc-arena/core";
import { useWebSocket } from "./useWebSocket";
import TrainingWorker from "../workers/training.worker.ts?worker";

const WS_URL = import.meta.env.VITE_WS_URL ?? "ws://localhost:4000";

export function useTradingSim() {
  const ws = useWebSocket();
  const engineRef = useRef<TradingEngine | null>(null);
  const [localSnap, setLocalSnap] = useState<EngineSnapshot | null>(null);
  const [useLocal, setUseLocal] = useState(false);

  // Fallback to local engine if WebSocket is not connected after 4 seconds
  useEffect(() => {
    if (ws.connected) {
      queueMicrotask(() => setUseLocal(false));
      return;
    }
    const timeout = setTimeout(() => {
      if (!ws.connected && !ws.snap) {
        setUseLocal(true);
        const engine = new TradingEngine();
        engineRef.current = engine;
        engine.onUpdate(() => setLocalSnap(engine.snapshot()));
        void engine.start().then(() => {
          // engine keeps running
        });
      }
    }, 4000);
    return () => clearTimeout(timeout);
  }, [ws.connected, ws.snap]);

  // Web Worker retraining for local mode
  useEffect(() => {
    if (!useLocal) return;
    const engine = engineRef.current;
    if (!engine) return;

    const worker = new TrainingWorker();
    worker.onmessage = (e: MessageEvent<{ type: string; model: "xgb" | "rf" | "gru"; state: unknown }>) => {
      const msg = e.data;
      if (msg.type !== "trained") return;
      try {
        if (msg.model === "xgb") engine.replaceModel("xgb", GradientBoostingModel.fromJSON(msg.state));
        else if (msg.model === "rf") engine.replaceModel("rf", RandomForestModel.fromJSON(msg.state));
        else if (msg.model === "gru") engine.replaceModel("gru", GRUModel.fromJSON(msg.state));
      } catch {
        // ignore malformed worker state
      }
    };

    const interval = setInterval(() => {
      const data = engine.getTrainingData(400);
      if (!data) return;
      worker.postMessage({ type: "train", model: "xgb", X: data.X, y: data.returns, aggression: engine.aggressionPerModel.xgb });
      worker.postMessage({ type: "train", model: "rf", X: data.X, y: data.returns, aggression: engine.aggressionPerModel.rf });
      worker.postMessage({ type: "train", model: "gru", X: data.X, y: data.returns, aggression: engine.aggressionPerModel.gru, steps: 20 });
    }, 30_000);

    return () => {
      clearInterval(interval);
      worker.terminate();
    };
  }, [useLocal]);

  // Cleanup local engine on unmount
  useEffect(() => {
    return () => {
      engineRef.current?.stop();
      engineRef.current = null;
    };
  }, []);

  const toggleRunning = useCallback(() => {
    if (useLocal) {
      engineRef.current?.toggleRunning();
    } else {
      ws.toggleRunning();
    }
  }, [useLocal, ws]);

  const setAggression = useCallback((id: ModelId, level: AggressionLevel) => {
    if (useLocal) {
      engineRef.current?.setAggression(id, level);
    } else {
      ws.setAggression(id, level);
    }
  }, [useLocal, ws]);

  const reset = useCallback(async () => {
    if (useLocal) {
      await engineRef.current?.reset();
    } else {
      await ws.reset();
    }
  }, [useLocal, ws]);

  const exportModels = useCallback(async () => {
    if (useLocal) {
      const state = engineRef.current?.exportState();
      return state ?? null;
    }
    const res = await fetch(`${import.meta.env.VITE_API_URL ?? "http://localhost:4000"}/api/export`);
    return (await res.json()) as import("@btc-arena/core").PersistedEngineState;
  }, [useLocal]);

  const importModels = useCallback(async (state: unknown) => {
    const typed = state as import("@btc-arena/core").PersistedEngineState;
    if (useLocal) {
      engineRef.current?.importState(typed);
      return;
    }
    await fetch(`${import.meta.env.VITE_API_URL ?? "http://localhost:4000"}/api/import`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(typed),
    });
  }, [useLocal]);

  return {
    snap: useLocal ? localSnap : ws.snap,
    toggleRunning,
    setAggression,
    reset,
    exportModels,
    importModels,
    connected: ws.connected,
    mode: useLocal ? "local" : ws.mode,
    wsUrl: WS_URL,
  };
}
