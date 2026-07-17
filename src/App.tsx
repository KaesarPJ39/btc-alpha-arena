import { Routes, Route, Navigate } from "react-router";
import { ArenaPage } from "@/pages/Arena";
import { ModelPage } from "@/pages/ModelPage";
import { DocsPage } from "@/pages/Docs";
import { useTradingSim } from "@/hooks/useTradingSim";
import { Header } from "@/sections/Header";
import { Sidebar } from "@/sections/Sidebar";

export default function App() {
  const sim = useTradingSim();
  const snap = sim.snap;

  if (!snap) {
    return (
      <div className="arena-bg flex min-h-screen items-center justify-center">
        <div className="text-center">
          <div className="mx-auto mb-4 h-10 w-10 animate-spin rounded-full border-2 border-emerald-500 border-t-transparent" />
          <p className="text-sm text-muted-foreground">Iniciando motores de trading…</p>
          <p className="mt-1 text-[11px] text-muted-foreground/70">Cargando 5 modelos · BTC/USD</p>
        </div>
      </div>
    );
  }

  return (
    <div className="arena-bg min-h-screen">
      <Header
        market={snap.market}
        status={snap.status}
        statusDetail={snap.statusDetail}
        running={snap.running}
        onToggle={sim.toggleRunning}
      />
      <div className="mx-auto flex max-w-[1500px] gap-0 px-0">
        <Sidebar />
        <main className="min-w-0 flex-1 space-y-5 px-5 py-6">
          <Routes>
            <Route path="/" element={<Navigate to="/arena" replace />} />
            <Route path="/arena" element={<ArenaPage snap={snap} />} />
            <Route path="/model/rl" element={<ModelPage snap={snap} modelId="rl" onTune={sim.setAggression} />} />
            <Route path="/model/xgb" element={<ModelPage snap={snap} modelId="xgb" onTune={sim.setAggression} />} />
            <Route path="/model/stat" element={<ModelPage snap={snap} modelId="stat" onTune={sim.setAggression} />} />
            <Route path="/model/rf" element={<ModelPage snap={snap} modelId="rf" onTune={sim.setAggression} />} />
            <Route path="/model/lstm" element={<ModelPage snap={snap} modelId="lstm" onTune={sim.setAggression} />} />
            <Route path="/docs" element={<DocsPage />} />
            <Route path="*" element={<Navigate to="/arena" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
