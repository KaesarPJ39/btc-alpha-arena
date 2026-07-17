// ── Web Worker para entrenamiento de modelos fuera del hilo principal ────────
// Este worker importa @btc-arena/core y entrena modelos pesados (XGB, RF, GRU)
// cuando el frontend corre en modo local (sin backend WebSocket).

import {
  GradientBoostingModel,
  RandomForestModel,
  GRUModel,
} from "@btc-arena/core";

export type WorkerTrainRequest = {
  type: "train";
  model: "xgb" | "rf" | "gru";
  X: number[][];
  y: number[];
  aggression: 0 | 1 | 2 | 3 | 4;
  steps?: number;
};

export type WorkerTrainResponse = {
  type: "trained";
  model: "xgb" | "rf" | "gru";
  state: unknown;
};

self.onmessage = (event: MessageEvent<WorkerTrainRequest>) => {
  const msg = event.data;
  if (msg.type !== "train") return;

  try {
    let state: unknown;
    if (msg.model === "xgb") {
      const m = new GradientBoostingModel();
      m.setAggression(msg.aggression);
      m.train(msg.X, msg.y);
      state = m.toJSON();
    } else if (msg.model === "rf") {
      const m = new RandomForestModel();
      m.setAggression(msg.aggression);
      m.train(msg.X, msg.y);
      state = m.toJSON();
    } else {
      const m = new GRUModel();
      m.setAggression(msg.aggression);
      m.train(msg.X, msg.y, msg.steps ?? 60);
      state = m.toJSON();
    }
    const response: WorkerTrainResponse = { type: "trained", model: msg.model, state };
    self.postMessage(response);
  } catch (err) {
    self.postMessage({ type: "error", error: String(err) });
  }
};

export {};
