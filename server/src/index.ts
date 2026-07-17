// ── BTC Arena trading server ────────────────────────────────────────────────
// Express + WebSocket backend. Runs the TradingEngine from @btc-arena/core,
// broadcasts snapshots to connected frontends, persists models and logs trades.

import express from "express";
import { createServer } from "http";
import { WebSocketServer, WebSocket } from "ws";
import { existsSync, mkdirSync, writeFileSync, readFileSync, appendFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
import {
  TradingEngine,
  type EngineSnapshot,
  type PersistedEngineState,
  type ModelId,
  MODEL_IDS,
} from "@btc-arena/core";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const DATA_DIR = resolve(__dirname, "../data");
const STATE_FILE = resolve(DATA_DIR, "engine-state.json");
const LOG_FILE = resolve(DATA_DIR, "trades.csv");

const PORT = Number(process.env.PORT ?? 4000);
const BROADCAST_MS = 3000;

function ensureDataDir(): void {
  if (!existsSync(DATA_DIR)) mkdirSync(DATA_DIR, { recursive: true });
}

function logHeader(): void {
  if (!existsSync(LOG_FILE)) {
    writeFileSync(LOG_FILE, "timestamp,agent,side,price,qty,value,fee,reason\n");
  }
}

function logTrade(trade: EngineSnapshot["trades"][number]): void {
  ensureDataDir();
  logHeader();
  const line = `${new Date(trade.ts).toISOString()},${trade.agent},${trade.side},${trade.price},${trade.qty},${trade.value},${trade.fee},"${trade.reason.replace(/"/g, "'")}"\n`;
  appendFileSync(LOG_FILE, line);
}

function loadState(): PersistedEngineState | null {
  if (!existsSync(STATE_FILE)) return null;
  try {
    const raw = readFileSync(STATE_FILE, "utf-8");
    return JSON.parse(raw) as PersistedEngineState;
  } catch {
    return null;
  }
}

function saveState(engine: TradingEngine): void {
  ensureDataDir();
  try {
    const state = engine.exportState();
    writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
  } catch (err) {
    console.error("Failed to save engine state:", err);
  }
}

const app = express();
const server = createServer(app);
const wss = new WebSocketServer({ server });

app.use(express.json({ limit: "10mb" }));
app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") {
    res.sendStatus(204);
    return;
  }
  next();
});

const engine = new TradingEngine();

app.get("/health", (_req, res) => {
  res.json({ ok: true, status: engine.status, ticks: engine.ticks, connections: wss.clients.size });
});

app.get("/api/state", (_req, res) => {
  res.json(engine.snapshot());
});

app.post("/api/aggression/:id", (req, res) => {
  const id = req.params.id as ModelId;
  const level = Number(req.body.level);
  if (!MODEL_IDS.includes(id) || !Number.isInteger(level) || level < 0 || level > 4) {
    res.status(400).json({ error: "Invalid model or level" });
    return;
  }
  engine.setAggression(id, level as 0 | 1 | 2 | 3 | 4);
  res.json({ ok: true });
});

app.post("/api/toggle", (_req, res) => {
  engine.toggleRunning();
  res.json({ ok: true, running: engine.running });
});

app.post("/api/reset", async (_req, res) => {
  res.json({ ok: true, message: "Reset initiated" });
  await engine.reset();
});

app.get("/api/export", (_req, res) => {
  const state = engine.exportState();
  res.setHeader("Content-Disposition", "attachment; filename=btc-arena-models.json");
  res.setHeader("Content-Type", "application/json");
  res.json(state);
});

app.post("/api/import", (req, res) => {
  const state = req.body as PersistedEngineState;
  try {
    engine.importState(state);
    saveState(engine);
    res.json({ ok: true });
  } catch (err) {
    res.status(400).json({ error: String(err) });
  }
  return;
});

const clients = new Set<WebSocket>();

wss.on("connection", (ws) => {
  clients.add(ws);
  ws.send(JSON.stringify({ type: "snapshot", data: engine.snapshot() }));
  ws.on("close", () => clients.delete(ws));
  ws.on("message", (raw) => {
    try {
      const msg = JSON.parse(String(raw)) as { type: string; payload?: unknown };
      if (msg.type === "setAggression") {
        const { id, level } = msg.payload as { id: ModelId; level: 0 | 1 | 2 | 3 | 4 };
        engine.setAggression(id, level);
      } else if (msg.type === "toggle") {
        engine.toggleRunning();
      } else if (msg.type === "reset") {
        void engine.reset();
      }
    } catch {
      // ignore malformed messages
    }
  });
});

let lastLoggedTradeTs = 0;
function broadcast(): void {
  const snap = engine.snapshot();
  const msg = JSON.stringify({ type: "snapshot", data: snap });
  for (const ws of clients) {
    if (ws.readyState === WebSocket.OPEN) ws.send(msg);
  }
  // Log new trades
  for (let i = 0; i < snap.trades.length; i++) {
    const t = snap.trades[i];
    if (t.ts > lastLoggedTradeTs) {
      logTrade(t);
      if (i === 0) lastLoggedTradeTs = t.ts;
    }
  }
}

async function main(): Promise<void> {
  ensureDataDir();
  const saved = loadState();
  if (saved) {
    console.log("Loaded persisted model state");
    engine.importState(saved);
  }

  engine.onUpdate(() => {
    // Optional: throttle broadcasts could go here, but interval handles it.
  });

  void engine.start();

  setInterval(broadcast, BROADCAST_MS);
  setInterval(() => saveState(engine), 30_000);

  server.listen(PORT, () => {
    console.log(`BTC Arena server listening on http://localhost:${PORT}`);
    console.log(`WebSocket server ready on ws://localhost:${PORT}`);
  });
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
