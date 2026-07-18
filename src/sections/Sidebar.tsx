import { NavLink } from "react-router";
import { Swords, BrainCircuit, LineChart, Sigma, TreePine, Network, Cpu, BookOpen } from "lucide-react";
import { cn } from "@/lib/utils";
import { MODEL_IDS, MODELS } from "@/lib/registry";

const NAV = [
  { to: "/arena", label: "Arena", icon: Swords, hint: "Comparativa de los 6 modelos" },
  { to: "/model/rl", label: "Agente RL", icon: BrainCircuit, hint: MODELS.rl.description },
  { to: "/model/xgb", label: "XGBoost", icon: LineChart, hint: MODELS.xgb.description },
  { to: "/model/stat", label: "Statistical", icon: Sigma, hint: MODELS.stat.description },
  { to: "/model/rf", label: "Random Forest", icon: TreePine, hint: MODELS.rf.description },
  { to: "/model/gru", label: "GRU", icon: Network, hint: MODELS.gru.description },
  { to: "/model/rvfl", label: "RVFL", icon: Cpu, hint: MODELS.rvfl.description },
  { to: "/docs", label: "Documentación", icon: BookOpen, hint: "Cómo funciona cada modelo" },
] as const;

export function Sidebar() {
  return (
    <aside className="hidden w-[244px] shrink-0 border-r border-border/50 bg-card/20 px-3 py-5 md:block">
      <nav className="sticky top-24 space-y-1">
        <p className="px-3 pb-2 text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground">
          Navegación
        </p>
        {NAV.map((item, idx) => {
          const Icon = item.icon;
          const isModel = idx > 0 && idx < NAV.length - 1;
          const modelId = isModel ? MODEL_IDS[idx - 1] : null;
          const tone = modelId ? MODELS[modelId].tailwindText : "";
          return (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                cn(
                  "group relative flex items-start gap-3 rounded-xl px-3 py-2.5 transition-colors",
                  isActive
                    ? "bg-secondary/80"
                    : "text-muted-foreground hover:bg-secondary/40 hover:text-foreground"
                )
              }
            >
              {({ isActive }) => (
                <>
                  {isActive && (
                    <span className="absolute left-0 top-1/2 h-7 w-1 -translate-y-1/2 rounded-r bg-primary" />
                  )}
                  <Icon
                    className={cn(
                      "mt-0.5 h-4 w-4 shrink-0",
                      isActive ? "text-primary" : tone
                    )}
                  />
                  <span className="min-w-0">
                    <span className="block text-[13px] font-medium leading-tight">
                      {item.label}
                    </span>
                    <span className="block truncate text-[10px] leading-tight text-muted-foreground">
                      {item.hint}
                    </span>
                  </span>
                </>
              )}
            </NavLink>
          );
        })}
      </nav>
      <div className="mt-6 rounded-xl border border-border/40 bg-secondary/30 px-3 py-3 text-[10px] leading-relaxed text-muted-foreground">
        <p className="font-semibold text-foreground/80">BTC Alpha Arena</p>
        <p className="mt-1">6 modelos ML/RRL ejecutándose en el navegador. Precios reales BTC/USD vía API pública.</p>
      </div>
    </aside>
  );
}
