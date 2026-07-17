import { useEffect, useRef, useState, useCallback } from "react";
import type { EngineSnapshot, ModelId, AggressionLevel } from "@btc-arena/core";

const WS_URL = import.meta.env.VITE_WS_URL ?? "ws://localhost:4000";

interface WsMessage {
  type: "snapshot";
  data: EngineSnapshot;
}

export interface TradingSimAPI {
  snap: EngineSnapshot | null;
  toggleRunning: () => void;
  setAggression: (id: ModelId, level: AggressionLevel) => void;
  reset: () => Promise<void>;
  connected: boolean;
  mode: "websocket" | "local" | "loading";
}

export function useWebSocket(): TradingSimAPI {
  const wsRef = useRef<WebSocket | null>(null);
  const [snap, setSnap] = useState<EngineSnapshot | null>(null);
  const [connected, setConnected] = useState(false);
  const [mode, setMode] = useState<"websocket" | "local" | "loading">("loading");

  useEffect(() => {
    let cancelled = false;
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      if (cancelled) return;
      setConnected(true);
      setMode("websocket");
    };

    ws.onmessage = (event) => {
      if (cancelled) return;
      try {
        const msg = JSON.parse(String(event.data)) as WsMessage;
        if (msg.type === "snapshot") setSnap(msg.data);
      } catch {
        // ignore malformed messages
      }
    };

    ws.onclose = () => {
      if (cancelled) return;
      setConnected(false);
    };

    ws.onerror = () => {
      if (cancelled) return;
      setConnected(false);
    };

    return () => {
      cancelled = true;
      ws.close();
      wsRef.current = null;
    };
  }, []);

  const send = useCallback((type: string, payload?: unknown) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type, payload }));
    }
  }, []);

  const toggleRunning = useCallback(() => {
    send("toggle");
  }, [send]);

  const setAggression = useCallback((id: ModelId, level: AggressionLevel) => {
    send("setAggression", { id, level });
  }, [send]);

  const reset = useCallback(async () => {
    send("reset");
  }, [send]);

  return { snap, toggleRunning, setAggression, reset, connected, mode };
}
